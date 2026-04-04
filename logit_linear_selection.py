"""
logit_linear_selection.py - Multi-Teacher Preference Dataset Generator
=======================================================================
Generates preference datasets using the Logit-Linear Selection (LLS) algorithm.

What this script does:
  1. Loads the tulu-2.5 preference dataset from HuggingFace (once, shared across teachers).
  2. For each teacher model configured in config.yaml:
       a. Checks if a preference_dataset.json already exists → skips (caching) if found.
       b. Loads the teacher model onto GPU.
       c. Scores every (prompt, chosen, rejected) triple with the logit-linear score:
              score(r) = log P(r | prompt, sys_prompt)  -  log P(r | prompt)
       d. Selects the best (chosen, rejected) pair per prompt, length-normalises weights,
          and keeps only the top `quantile` fraction.
       e. Saves the dataset (list of {prompt, chosen, rejected, weight} dicts) and a
          detailed stats.json to the teacher's output directory.
       f. Frees GPU memory before moving to the next teacher.

Caching:
  Re-running the script skips any teacher whose preference_dataset.json already exists.
  Delete that file to force recomputation.

Multi-GPU:
  accelerate launch logit_linear_selection.py

Single-GPU / CPU:
  python logit_linear_selection.py
"""

import math
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm.auto import tqdm

import json
import os
from pathlib import Path
import yaml
import hashlib
import sys
import tempfile

from helper_functions import (
    clear_memory,
    sanitize,
    should_filter,
    insert_prompt,
    insert_completion,
    sum_logprob_targets,
)

# ============================================================
# SETUP: Verify environment and load configuration
# ============================================================

if not os.getenv("HF_HOME"):
    print("ERROR: HF_HOME environment variable not set!")
    print("Please set it before running this script :)")
    sys.exit(1)

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

local_root = os.path.expanduser(cfg["local_root"])

# Shared LLS hyper-parameters (same for every teacher)
shared_config = {
    "filter_words":       cfg.get("filter_words"),
    "batch_size":         cfg["lls_dataset"]["batch_size"],
    "training_precision": cfg["lls_dataset"]["training_precision"],
    "truncation_value":   cfg["lls_dataset"]["truncation_tokens"],
    "quantile":           cfg["lls_dataset"]["quantile"],
}

# ============================================================
# TEACHER LIST: support multi-teacher and single-teacher modes
# ============================================================

if "teachers" in cfg and cfg["teachers"]:
    # Multi-teacher mode: list of {name, model, system_prompt}
    teachers = cfg["teachers"]
    print(f"Multi-teacher mode: {len(teachers)} teacher(s) configured.")
    for t in teachers:
        print(f"  - {t['name']}: model={t['model']}, "
              f"prompt='{t['system_prompt'][:50]}...'")
else:
    # Single-teacher fallback (backward-compatible)
    teachers = [{
        "name":          sanitize(cfg["teacher_model"].split("/")[-1]),
        "model":         cfg["teacher_model"],
        "system_prompt": cfg["system_prompt"],
    }]
    print("Single-teacher mode (using legacy teacher_model + system_prompt config).")


# ============================================================
# HELPER: build output directory path for a teacher
# ============================================================

def get_teacher_dir(teacher_cfg):
    """
    Return the experiment directory for *this* teacher.
    The directory name encodes name, model, prompt hash, and LLS parameters
    so that different configurations never collide on disk.
    """
    teacher_name = teacher_cfg.get("name") or sanitize(teacher_cfg["model"].split("/")[-1])
    model_short   = sanitize(teacher_cfg["model"].split("/")[-1])
    prompt_hash   = hashlib.md5(teacher_cfg["system_prompt"].encode()).hexdigest()[:8]
    trunc         = shared_config["truncation_value"]
    quant         = shared_config["quantile"]
    dir_name      = f"{teacher_name}_{model_short}_{prompt_hash}_trunc{trunc}_q{quant}"
    return os.path.join(local_root, dir_name)


# ============================================================
# HELPER: atomic JSON write (prevents file corruption on crash)
# ============================================================

def save_json_atomic(data, path):
    """
    Write *data* as JSON to *path* using a temp-file + rename.
    If the script crashes mid-write, the original file is never corrupted.
    """
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    # os.replace is atomic on POSIX; on Windows it may overwrite silently
    os.replace(tmp_path, path)


# ============================================================
# CORE FUNCTION 1: compute log P(response | prompt [, sys_prompt])
# ============================================================

def compute_log_probs_single_fast(
    model, tokenizer, instruction, histories, futures, length_flag, sys_prompt
):
    """
    Compute log P(response | prompt, sys_prompt) for every (prompt, response) pair.

    Args:
        model        : teacher LM (on GPU)
        tokenizer    : teacher tokenizer
        instruction  : optional prefix added to every prompt (usually "")
        histories    : list[str] – prompt strings
        futures      : list[str] – response strings
        length_flag  : bool – if True, also return response lengths in tokens
        sys_prompt   : str  – system prompt; pass "" for the base distribution

    Returns:
        (log_probs, lengths)
          log_probs : list[float], one value per pair
          lengths   : list[int]   (populated only when length_flag=True)
    """
    lengths = []
    prompts = []

    # Encode prompts through the model's chat template
    desc = "Encoding prompts (sys)" if sys_prompt else "Encoding prompts (base)"
    for history in tqdm(histories, desc=desc, leave=False):
        encoded = tokenizer.encode(
            insert_prompt(instruction + history, sys_prompt, tokenizer),
            add_special_tokens=False,
        )
        prompts.append(encoded)

    # Encode responses
    responses = []
    for future in tqdm(futures, desc="Encoding responses", leave=False):
        enc = tokenizer.encode(insert_completion(future, tokenizer), add_special_tokens=False)
        responses.append(enc)
        if length_flag:
            lengths.append(len(enc))

    pairs     = [(prompts[i], responses[i]) for i in range(len(histories))]
    log_probs = sum_logprob_targets(model, tokenizer, pairs,
                                    batch_size=shared_config["batch_size"])
    return log_probs, lengths


# ============================================================
# CORE FUNCTION 2: score every (prompt, chosen, rejected) triple
# ============================================================

def compute_weighted_dataset(
    model, tokenizer, data, truncation_value, sys_prompt, rank, world_size
):
    """
    For every example in *data*, compute the logit-linear score for each
    chosen / rejected response:

        score(r) = log P(r | prompt, sys_prompt)  −  log P(r | prompt)

    A high positive score means the system prompt strongly *increases* the
    probability of that response – i.e., the teacher "prefers" it.

    Args:
        model           : teacher LM
        tokenizer       : teacher tokenizer
        data            : list of {"prompt", "chosen", "rejected"} dicts
        truncation_value: max response length in tokens (truncate longer ones)
        sys_prompt      : teacher's system prompt string
        rank            : current GPU rank (0 = primary)
        world_size      : total number of GPUs

    Returns (rank 0):
        (weighted_dataset, filter_stats)
          weighted_dataset : list of dicts with chosen_scores, rejected_scores, etc.
          filter_stats     : {"original_size", "removed_count"}

    Returns (rank ≠ 0):
        (None, filter_stats)
    """
    filter_words = shared_config.get("filter_words")

    # ---- Remove examples that already mention the target behaviour ----
    # We don't want the student to learn from "contaminated" examples.
    filter_stats = {"original_size": len(data), "removed_count": 0}
    if filter_words:
        before = len(data)
        data = [
            row for row in data
            if not (
                should_filter(row["prompt"], filter_words)
                or any(should_filter(c, filter_words) for c in row["chosen"])
                or any(should_filter(r, filter_words) for r in row["rejected"])
            )
        ]
        filter_stats["removed_count"] = before - len(data)
        print(f"Word filter: {before} → {len(data)} examples "
              f"(removed {filter_stats['removed_count']})")

    N = len(data)
    print(f"Scoring {N} examples across {world_size} GPU(s)…")

    # Each GPU handles every world_size-th example (strided partition)
    rank_data = [data[idx] for idx in range(rank, N, world_size)]

    CHUNK_SIZE  = 25000   # Process 25k examples at a time (safe for A100 80 GB)
    local_tuples = []

    print(f"[Rank {rank}] {len(rank_data)} examples in chunks of {CHUNK_SIZE}…")

    for chunk_idx in range(0, len(rank_data), CHUNK_SIZE):
        chunk     = rank_data[chunk_idx : chunk_idx + CHUNK_SIZE]
        chunk_num = chunk_idx // CHUNK_SIZE + 1
        n_chunks  = math.ceil(len(rank_data) / CHUNK_SIZE)
        print(f"\n[Rank {rank}] Chunk {chunk_num}/{n_chunks} ({len(chunk)} examples)…")

        # Build flat lists so we can score everything in one batched call
        all_histories   = []   # prompt repeated once per response in this chunk
        all_futures     = []   # response strings
        boundaries      = []   # (start_idx, n_chosen, n_rejected) per prompt
        trunc_rank_data = []   # (prompt, trunc_chosen, trunc_rejected)

        for row in tqdm(chunk, desc="  Building chunk", leave=False):
            prompt   = row["prompt"]
            chosen   = row["chosen"]
            rejected = row["rejected"]

            # Truncate to avoid extremely long sequences
            chosen   = [tokenizer.decode(
                            tokenizer.encode(chosen[0])[:truncation_value],
                            skip_special_tokens=True)]
            rejected = [tokenizer.decode(
                            tokenizer.encode(rejected[0])[:truncation_value],
                            skip_special_tokens=True)]

            trunc_rank_data.append((prompt, chosen, rejected))

            responses  = chosen + rejected
            start_idx  = len(all_futures)
            all_histories.extend([prompt] * len(responses))
            all_futures.extend(responses)
            boundaries.append((start_idx, len(chosen), len(rejected)))

        # Step A: base log probs  (no system prompt = raw language model)
        print("  Computing base log probs (no system prompt)…")
        base_lp, all_lengths = compute_log_probs_single_fast(
            model, tokenizer, "", all_histories, all_futures,
            length_flag=True, sys_prompt=""
        )

        # Step B: system log probs  (with teacher's system prompt)
        print("  Computing system log probs (with system prompt)…")
        sys_lp, _ = compute_log_probs_single_fast(
            model, tokenizer, "", all_histories, all_futures,
            length_flag=False, sys_prompt=sys_prompt
        )

        # Step C: logit-linear score = difference in log probs
        # Positive → sys_prompt increases probability of this response
        all_scores = [s - b for s, b in zip(sys_lp, base_lp)]

        # Package per-prompt scored results
        for idx, (start, n_c, n_r) in enumerate(boundaries):
            row      = chunk[idx]
            trunc_row = trunc_rank_data[idx]
            scores   = all_scores[start : start + n_c + n_r]
            lengths  = all_lengths[start : start + n_c + n_r]

            local_tuples.append({
                "prompt":            row["prompt"],
                "chosen":            row["chosen"],
                "rejected":          row["rejected"],
                "truncated_chosen":  trunc_row[1],
                "truncated_rejected": trunc_row[2],
                "chosen_scores":     scores[:n_c],
                "rejected_scores":   scores[n_c:],
                "chosen_lengths":    lengths[:n_c],
                "rejected_lengths":  lengths[n_c:],
            })

        del all_histories, all_futures, base_lp, sys_lp, all_scores, boundaries, trunc_rank_data
        clear_memory()
        print(f"  Chunk done. Total processed so far: {len(local_tuples)}")

    # ---- Gather results from all GPUs onto rank 0 ----
    print("\nGathering results across GPU(s)…")
    gathered = gather_object(local_tuples)

    if rank != 0:
        return None, filter_stats

    print("Gather complete on rank 0.")
    weighted_dataset = []
    for part in gathered:
        if isinstance(part, list):
            weighted_dataset.extend(part)
        else:
            weighted_dataset.append(part)

    print(f"Total scored examples: {len(weighted_dataset)}")
    return weighted_dataset, filter_stats


# ============================================================
# CORE FUNCTION 3: filter and select top examples
# ============================================================

def logit_linear_selection(weighted_dataset, quantile):
    """
    Apply the three-stage LLS filter to select the best preference pairs.

    Stage 1 – Pair selection:
        For each prompt, pick the (chosen, rejected) pair with the largest
        positive weight gap  (chosen_score − rejected_score).
        Pairs with non-positive gap are discarded.

    Stage 2 – Length normalisation:
        Divide each weight by the total token length of the pair.
        This prevents long responses from unfairly dominating the ranking.

    Stage 3 – Quantile filter:
        Keep only the top `quantile` fraction (e.g., 0.1 = top 10 %).

    Args:
        weighted_dataset : output from compute_weighted_dataset (rank-0 only)
        quantile         : fraction to keep (0 < quantile ≤ 1)

    Returns:
        (dataset, stats)
          dataset : list of {prompt, chosen, rejected, weight} dicts
                    (weight is the normalised score, for use by combine_datasets.py)
          stats   : dict with detailed counts and score-distribution statistics
    """
    lls_stats = {"input_size": len(weighted_dataset)}

    # ---- Stage 1: pick best pair per prompt ----
    all_pairs           = []
    no_valid_pair_count = 0

    for row in weighted_dataset:
        chosen_scores  = row["chosen_scores"]
        rejected_scores = row["rejected_scores"]
        chosen         = row["truncated_chosen"]
        rejected       = row["truncated_rejected"]
        chosen_lengths = row["chosen_lengths"]
        rejected_lengths = row["rejected_lengths"]

        best_w        = 0.0   # keep only positive-weight pairs
        best_pair     = None
        best_pair_len = None

        for i_c in range(len(chosen)):
            for i_r in range(len(rejected)):
                # How much more does the teacher prefer chosen over rejected?
                w = chosen_scores[i_c] - rejected_scores[i_r]
                if w > best_w:
                    best_w        = w
                    best_pair     = (chosen[i_c], rejected[i_r])
                    best_pair_len = (chosen_lengths[i_c], rejected_lengths[i_r])

        if best_pair is not None:
            all_pairs.append({
                "prompt":       row["prompt"],
                "chosen":       best_pair[0],
                "rejected":     best_pair[1],
                "weight":       float(best_w),
                "pair_lengths": best_pair_len,
            })
        else:
            # Teacher does not clearly prefer chosen over rejected → drop
            no_valid_pair_count += 1

    lls_stats["no_valid_pair_count"] = no_valid_pair_count
    lls_stats["valid_pairs_size"]    = len(all_pairs)
    print(f"Stage 1 – Pair selection: {len(all_pairs)} valid / "
          f"{len(weighted_dataset)} total "
          f"({no_valid_pair_count} dropped – no positive weight)")

    # ---- Stage 2: length normalisation ----
    norm_weights = []
    for row in all_pairs:
        lc, lr = row["pair_lengths"]
        norm_weights.append(row["weight"] / max(lc + lr, 1))

    if not norm_weights:
        print("No positive-weight examples found. Returning empty dataset.")
        return [], lls_stats

    # Normalise by maximum so all weights are in [0, 1]
    max_w        = max(norm_weights)
    norm_weights = [w / max_w for w in norm_weights]
    rows         = list(zip(all_pairs, norm_weights))

    print("Stage 2 – Length normalisation complete.")

    # ---- Compute score-distribution statistics (before quantile cut) ----
    ws = sorted(norm_weights)

    def pct(p):
        return ws[int(p * (len(ws) - 1))]

    print("\nWeight quantiles (length-normalised, max-normalised):")
    for p in (0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        print(f"  {int(p*100):3d}%: {pct(p):.4f}")

    score_distribution = {
        "min":    float(min(norm_weights)),
        "max":    float(max(norm_weights)),
        "mean":   float(np.mean(norm_weights)),
        "median": float(np.median(norm_weights)),
        "std":    float(np.std(norm_weights)),
        "q25":    pct(0.25),
        "q50":    pct(0.50),
        "q75":    pct(0.75),
        "q90":    pct(0.90),
        "q95":    pct(0.95),
        "q99":    pct(0.99),
    }
    lls_stats["score_distribution_before_quantile"] = score_distribution

    # ---- Stage 3: quantile filter ----
    rows.sort(key=lambda x: x[1], reverse=True)
    k             = math.ceil(quantile * len(rows))
    dropped_count = len(rows) - k
    rows          = rows[:k]

    lls_stats["dropped_by_quantile"] = dropped_count
    lls_stats["final_size"]          = len(rows)

    print(f"\nStage 3 – Quantile filter (keep top {quantile*100:.0f}%):")
    print(f"  Before: {len(all_pairs)}")
    print(f"  Kept:   {len(rows)}")
    print(f"  Dropped: {dropped_count}")

    # ---- Build output ----
    # Include normalised weight so combine_datasets.py can average scores.
    output = [
        {
            "prompt":   row["prompt"],
            "chosen":   row["chosen"],
            "rejected": row["rejected"],
            "weight":   float(w),    # normalised weight in [0, 1]
        }
        for row, w in rows
    ]

    print(f"\nFinal dataset: {len(output)} examples")
    return output, lls_stats


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    # ---- GPU / accelerate setup (done once for all teachers) ----
    if torch.cuda.is_available():
        accelerator = Accelerator()
        device      = accelerator.device
        rank        = accelerator.process_index
        world_size  = accelerator.num_processes
        print(f"Device: {device} | Rank: {rank} | World size: {world_size}")
        if rank == 0 and world_size == 1 and torch.cuda.device_count() > 1:
            print(f"Note: {torch.cuda.device_count()} GPUs detected but only 1 used.")
            print("Use: accelerate launch --num_processes N logit_linear_selection.py")
    else:
        accelerator = None
        device      = torch.device("cpu")
        rank        = 0
        world_size  = 1
        print("CUDA not available – using CPU.")

    # ---- Load and preprocess raw dataset ONCE (shared across all teachers) ----
    # Use the first teacher's tokenizer for the prompt-length check.
    if rank == 0:
        print(f"\nLoading tokenizer for preprocessing: {teachers[0]['model']}")
    preprocess_tokenizer = AutoTokenizer.from_pretrained(teachers[0]["model"])

    if rank == 0:
        print("Loading dataset: allenai/tulu-2.5-preference-data (stack_exchange_paired)…")
    raw_ds = load_dataset("allenai/tulu-2.5-preference-data", split="stack_exchange_paired")
    if rank == 0:
        print(f"Raw dataset size: {len(raw_ds)} examples")

    # Keep only single-turn, user-first, short-prompt examples
    data = []
    for row in tqdm(raw_ds, desc="Preprocessing", disable=(rank != 0)):
        chosen   = row.get("chosen")
        rejected = row.get("rejected")

        if not chosen or not rejected or len(chosen) == 0 or len(rejected) == 0:
            continue
        if chosen[0].get("role") != "user":
            continue
        if len(chosen) != 2 or len(rejected) != 2:   # single-turn only
            continue

        prompt = chosen[0].get("content", "").strip()

        # Skip long prompts (expensive and often lower quality)
        if len(preprocess_tokenizer.encode(prompt, add_special_tokens=False)) > 250:
            continue

        data.append({
            "prompt":   prompt,
            "chosen":   [chosen[1].get("content", "")],
            "rejected": [rejected[1].get("content", "")],
        })

    input_size = len(data)
    if rank == 0:
        print(f"Preprocessed: {input_size} examples kept (from {len(raw_ds)} raw)")

    # ---- Process each teacher in turn ----
    for teacher_idx, teacher_cfg in enumerate(teachers):
        teacher_name      = teacher_cfg.get("name") or sanitize(teacher_cfg["model"].split("/")[-1])
        teacher_model_id  = teacher_cfg["model"]
        teacher_sys_prompt = teacher_cfg["system_prompt"]

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Teacher {teacher_idx + 1}/{len(teachers)}: {teacher_name}")
            print(f"  Model:  {teacher_model_id}")
            print(f"  Prompt: {teacher_sys_prompt}")
            print(f"{'='*60}")

        # Build output paths for this teacher
        teacher_dir             = get_teacher_dir(teacher_cfg)
        dataset_dir             = os.path.join(teacher_dir, "datasets")
        os.makedirs(dataset_dir, exist_ok=True)
        preference_dataset_path = os.path.join(dataset_dir, "preference_dataset.json")
        config_save_path        = os.path.join(dataset_dir, "dataset_config.json")
        stats_save_path         = os.path.join(dataset_dir, "stats.json")

        # ---- Caching: skip if dataset already exists ----
        # All ranks check the shared filesystem independently.
        if os.path.exists(preference_dataset_path):
            if rank == 0:
                print(f"[CACHE HIT] Dataset already exists:")
                print(f"  {preference_dataset_path}")
                print(f"  Skipping teacher '{teacher_name}'.")
                print(f"  Delete the file above to force recomputation.")
            # Synchronise all ranks before moving to next teacher
            if accelerator is not None:
                accelerator.wait_for_everyone()
            continue

        if rank == 0:
            print(f"[CACHE MISS] Computing dataset for '{teacher_name}'…")

        teacher_start = time.time()

        # ---- Load teacher model ----
        if rank == 0:
            print(f"Loading teacher model: {teacher_model_id}…")
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)
        if teacher_tokenizer.pad_token_id is None:
            teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id

        dtype = torch.bfloat16 if shared_config["training_precision"] == 16 else torch.float32
        teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_id, dtype=dtype)

        if accelerator is not None:
            teacher_model = accelerator.prepare(teacher_model)
        else:
            teacher_model = teacher_model.to(device)

        # ---- Score the dataset ----
        if rank == 0:
            print("Computing logit-linear weights…")
        compute_start = time.time()
        weighted_dataset, filter_stats = compute_weighted_dataset(
            teacher_model, teacher_tokenizer, data,
            shared_config["truncation_value"], teacher_sys_prompt,
            rank, world_size,
        )
        compute_time = time.time() - compute_start

        # ---- Free teacher model from GPU on ALL ranks ----
        del teacher_model
        del teacher_tokenizer
        clear_memory()

        # ---- Non-rank-0 GPUs sync and move on ----
        if rank != 0:
            if accelerator is not None:
                accelerator.wait_for_everyone()
            continue

        # ================================================================
        # Everything below runs ONLY on rank 0
        # ================================================================

        if rank == 0:
            print(f"\nScore computation: {compute_time:.1f}s")

        # ---- Apply LLS filtering ----
        print("\nApplying logit-linear selection…")
        lls_start = time.time()
        final_dataset, lls_stats = logit_linear_selection(weighted_dataset, shared_config["quantile"])
        lls_time  = time.time() - lls_start
        total_time = time.time() - teacher_start

        # ---- Compile and print comprehensive statistics ----
        full_stats = {
            "teacher": {
                "name":          teacher_name,
                "model":         teacher_model_id,
                "system_prompt": teacher_sys_prompt,
            },
            "dataset_sizes": {
                "raw_input":              len(raw_ds),
                "after_preprocessing":    input_size,
                "after_word_filter":      filter_stats["original_size"] - filter_stats["removed_count"],
                "removed_by_word_filter": filter_stats["removed_count"],
                "valid_pairs":            lls_stats.get("valid_pairs_size", 0),
                "no_valid_pair_dropped":  lls_stats.get("no_valid_pair_count", 0),
                "dropped_by_quantile":    lls_stats.get("dropped_by_quantile", 0),
                "final":                  len(final_dataset),
            },
            "score_distribution": lls_stats.get("score_distribution_before_quantile", {}),
            "timing_seconds": {
                "score_computation": round(compute_time, 2),
                "lls_filtering":     round(lls_time, 2),
                "total":             round(total_time, 2),
            },
            "config": {
                "truncation_tokens": shared_config["truncation_value"],
                "quantile":          shared_config["quantile"],
                "batch_size":        shared_config["batch_size"],
                "filter_words":      shared_config["filter_words"],
            },
        }

        # Pretty-print summary
        ds = full_stats["dataset_sizes"]
        print(f"\n{'='*60}")
        print(f"SUMMARY – Teacher '{teacher_name}'")
        print(f"  Raw input:             {ds['raw_input']:>8,}")
        print(f"  After preprocessing:   {ds['after_preprocessing']:>8,}")
        print(f"  After word filter:     {ds['after_word_filter']:>8,}  "
              f"(removed {ds['removed_by_word_filter']})")
        print(f"  Valid pairs found:     {ds['valid_pairs']:>8,}  "
              f"({ds['no_valid_pair_dropped']} dropped – no positive weight)")
        print(f"  Dropped by quantile:   {ds['dropped_by_quantile']:>8,}")
        print(f"  Final dataset size:    {ds['final']:>8,}")
        sd = full_stats.get("score_distribution", {})
        if sd:
            print(f"  Score  min/max/mean/std: "
                  f"{sd['min']:.4f} / {sd['max']:.4f} / "
                  f"{sd['mean']:.4f} / {sd['std']:.4f}")
        print(f"  Total processing time: {total_time:.1f}s")
        print(f"{'='*60}")

        # ---- Save outputs (atomic writes) ----
        print(f"\nSaving preference dataset → {preference_dataset_path}")
        save_json_atomic(final_dataset, preference_dataset_path)

        save_json_atomic({
            "teacher_model":     teacher_model_id,
            "teacher_name":      teacher_name,
            "target_sys_prompt": teacher_sys_prompt,
            "filter_words":      shared_config["filter_words"],
            "batch_size":        shared_config["batch_size"],
            "training_precision": shared_config["training_precision"],
            "truncation_value":  shared_config["truncation_value"],
            "quantile":          shared_config["quantile"],
        }, config_save_path)

        save_json_atomic(full_stats, stats_save_path)
        print(f"Saved stats       → {stats_save_path}")

        # Sync so other ranks can start the next teacher
        if accelerator is not None:
            accelerator.wait_for_everyone()

    # ---- All teachers done ----
    if rank == 0:
        print(f"\n{'='*60}")
        print("All teachers processed!")
        if len(teachers) > 1:
            print("Next step: run 'python combine_datasets.py' to combine datasets.")
        else:
            print("Next step: run 'python training.py' to train the student model.")
        print(f"{'='*60}")

