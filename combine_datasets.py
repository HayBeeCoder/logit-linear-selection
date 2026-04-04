"""
combine_datasets.py - Multi-Teacher Dataset Combination
========================================================
Combines preference datasets from multiple teacher models using an
intersection + average score strategy.

Why intersection + average?
  Each teacher independently selects its top-k examples.
  An example that appears in *both* teachers' top-k lists is "doubly reinforced":
  both models agree it is a high-quality example for teaching the target behaviour.
  Averaging their scores gives a combined ranking that is robust to any single
  teacher's idiosyncrasies.

Algorithm:
  1. Load each teacher's preference_dataset.json (list of
     {prompt, chosen, rejected, weight} dicts produced by logit_linear_selection.py).
  2. Match examples by prompt text across all teachers.
  3. Keep only examples present in EVERY teacher's dataset (intersection).
  4. For each intersecting example, set:
         combined_weight = mean(weight_teacher_1, weight_teacher_2, …)
  5. Sort by combined_weight (descending) – higher = both teachers agree strongly.
  6. Save the combined dataset + detailed statistics.

Output format:
  The combined preference_dataset.json is a list of
  {prompt, chosen, rejected, weight} dicts – identical to the per-teacher format.
  training.py can read this directly.

Statistics generated:
  - Venn-diagram counts  (A only, B only, intersection)
  - Score correlation between teachers (Pearson r)
  - Score discrepancy analysis (where teachers disagree)
  - Score distribution of the combined dataset
  - Before/after dataset sizes
  - Quantile breakdown of combined scores

Usage:
  python combine_datasets.py

Configuration is read from config.yaml (same file as logit_linear_selection.py).
"""

import json
import os
import math
import hashlib
import sys
from pathlib import Path

import numpy as np
import yaml

from helper_functions import sanitize


# ============================================================
# Load config
# ============================================================

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

local_root = os.path.expanduser(cfg["local_root"])

# Shared LLS parameters (must match what was used to generate per-teacher datasets)
shared_config = {
    "truncation_value": cfg["lls_dataset"]["truncation_tokens"],
    "quantile":         cfg["lls_dataset"]["quantile"],
}


# ============================================================
# Helpers
# ============================================================

def get_teacher_dir(teacher_cfg):
    """Return the per-teacher experiment directory (mirrors logit_linear_selection.py)."""
    teacher_name = teacher_cfg.get("name") or sanitize(teacher_cfg["model"].split("/")[-1])
    model_short  = sanitize(teacher_cfg["model"].split("/")[-1])
    prompt_hash  = hashlib.md5(teacher_cfg["system_prompt"].encode()).hexdigest()[:8]
    trunc        = shared_config["truncation_value"]
    quant        = shared_config["quantile"]
    dir_name     = f"{teacher_name}_{model_short}_{prompt_hash}_trunc{trunc}_q{quant}"
    return os.path.join(local_root, dir_name)


def save_json_atomic(data, path):
    """Atomically write JSON: write to .tmp then rename to avoid partial files."""
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def pearson_correlation(xs, ys):
    """
    Compute the Pearson correlation coefficient between two lists of floats.
    Returns (r, interpretation_string).
    """
    if len(xs) < 2:
        return None, "not enough data"
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return None, "zero variance (all scores identical)"
    r = float(np.corrcoef(x, y)[0, 1])
    if   r >  0.9: label = "very strong positive"
    elif r >  0.7: label = "strong positive"
    elif r >  0.5: label = "moderate positive"
    elif r >  0.3: label = "weak positive"
    elif r > -0.3: label = "little to no correlation"
    elif r > -0.5: label = "weak negative"
    elif r > -0.7: label = "moderate negative"
    elif r > -0.9: label = "strong negative"
    else:          label = "very strong negative"
    return r, label


def score_distribution_stats(weights):
    """Return a dict of descriptive statistics for a list of weights."""
    if not weights:
        return {}
    ws = sorted(weights)

    def pct(p):
        return ws[max(0, int(p * (len(ws) - 1)))]

    return {
        "count":  len(ws),
        "min":    float(min(ws)),
        "max":    float(max(ws)),
        "mean":   float(np.mean(ws)),
        "median": float(np.median(ws)),
        "std":    float(np.std(ws)),
        "q10":    pct(0.10),
        "q25":    pct(0.25),
        "q50":    pct(0.50),
        "q75":    pct(0.75),
        "q90":    pct(0.90),
        "q95":    pct(0.95),
        "q99":    pct(0.99),
    }


# ============================================================
# Main combination logic
# ============================================================

def combine_datasets(teachers):
    """
    Load per-teacher datasets, compute intersection + average score,
    and return (combined_dataset, full_statistics).

    Args:
        teachers : list of teacher config dicts (from config.yaml)

    Returns:
        combined_dataset : list of {prompt, chosen, rejected, weight} dicts,
                           sorted by combined_weight descending
        stats            : dict with detailed statistics for analysis
    """

    # ---- Step 1: Load each teacher's dataset ----
    # teacher_datasets[i] = list of {prompt, chosen, rejected, weight}
    teacher_datasets = []
    teacher_names    = []

    print("=" * 60)
    print("Loading teacher datasets…")
    print("=" * 60)

    for teacher_cfg in teachers:
        t_name = teacher_cfg.get("name") or sanitize(teacher_cfg["model"].split("/")[-1])
        teacher_names.append(t_name)

        teacher_dir  = get_teacher_dir(teacher_cfg)
        dataset_path = os.path.join(teacher_dir, "datasets", "preference_dataset.json")

        if not os.path.exists(dataset_path):
            print(f"ERROR: Dataset not found for teacher '{t_name}':")
            print(f"  {dataset_path}")
            print("Run logit_linear_selection.py first to generate per-teacher datasets.")
            sys.exit(1)

        with open(dataset_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Support both old tuple format and new dict format
        dataset = []
        for item in raw:
            if isinstance(item, dict):
                dataset.append(item)
            else:
                # Legacy: (prompt, chosen, rejected) tuple
                dataset.append({
                    "prompt":   item[0],
                    "chosen":   item[1],
                    "rejected": item[2],
                    "weight":   1.0,   # no weight info → treat equally
                })

        teacher_datasets.append(dataset)
        print(f"  [{t_name}] Loaded {len(dataset):,} examples")

    # ---- Step 2: Index each dataset by prompt ----
    # prompt_to_entry[teacher_idx][prompt] = {prompt, chosen, rejected, weight}
    # We use the prompt string as the key; if two teachers select different
    # (chosen, rejected) pairs for the same prompt we use the pair from the
    # teacher with the higher individual weight.
    print("\nIndexing datasets by prompt…")
    prompt_maps = []
    for ds in teacher_datasets:
        pm = {}
        for entry in ds:
            p = entry["prompt"]
            # Keep the entry with highest weight in case of duplicates
            if p not in pm or entry.get("weight", 0) > pm[p].get("weight", 0):
                pm[p] = entry
        prompt_maps.append(pm)

    # ---- Step 3: Venn-diagram analysis ----
    # Collect the set of prompts from each teacher
    prompt_sets = [set(pm.keys()) for pm in prompt_maps]

    # Intersection: prompts that appear in ALL teachers
    intersection_prompts = prompt_sets[0]
    for ps in prompt_sets[1:]:
        intersection_prompts = intersection_prompts & ps

    # Union: prompts that appear in ANY teacher
    union_prompts = prompt_sets[0]
    for ps in prompt_sets[1:]:
        union_prompts = union_prompts | ps

    # Unique to each teacher
    unique_per_teacher = {}
    for i, (t_name, ps) in enumerate(zip(teacher_names, prompt_sets)):
        others = union_prompts - ps
        unique_per_teacher[t_name] = {
            "count":    len(ps - intersection_prompts),
            "examples": len(ps),
        }

    print(f"\nVenn-diagram summary ({len(teachers)} teachers):")
    for t_name, ps in zip(teacher_names, prompt_sets):
        only = len(ps - intersection_prompts)
        print(f"  {t_name}: {len(ps):,} total  |  {only:,} unique to this teacher")
    print(f"  Intersection (all teachers): {len(intersection_prompts):,}")
    print(f"  Union (any teacher):         {len(union_prompts):,}")

    # ---- Step 4: Build intersection dataset with combined (averaged) score ----
    print("\nBuilding combined dataset (intersection + average score)…")
    combined = []
    score_vectors = {t: [] for t in teacher_names}   # parallel lists for correlation

    for prompt in sorted(intersection_prompts):
        # Collect entries from each teacher
        entries  = [pm[prompt] for pm in prompt_maps]
        weights  = [e.get("weight", 1.0) for e in entries]

        # Average the scores → combined weight
        combined_weight = float(np.mean(weights))

        # For the (chosen, rejected) pair, use the entry from the highest-weight teacher
        # (they typically agree, but this handles edge cases)
        best_entry = max(entries, key=lambda e: e.get("weight", 0))

        combined.append({
            "prompt":            best_entry["prompt"],
            "chosen":            best_entry["chosen"],
            "rejected":          best_entry["rejected"],
            "weight":            combined_weight,
            # Store per-teacher weights for analysis
            "teacher_weights":   {t: w for t, w in zip(teacher_names, weights)},
        })

        for t_name, w in zip(teacher_names, weights):
            score_vectors[t_name].append(w)

    # Sort by combined weight descending (highest agreement → first)
    combined.sort(key=lambda x: x["weight"], reverse=True)

    # ---- Step 5: Score discrepancy analysis ----
    # An example is "high discrepancy" when teachers score it very differently.
    discrepancy_threshold = 0.2   # teachers differ by more than this
    high_discrepancy_count = 0
    discrepancies = []

    for item in combined:
        tw = list(item["teacher_weights"].values())
        diff = max(tw) - min(tw)
        if diff > discrepancy_threshold:
            high_discrepancy_count += 1
            discrepancies.append({
                "prompt":          item["prompt"][:80] + "…",
                "teacher_weights": item["teacher_weights"],
                "max_diff":        round(diff, 4),
            })

    # ---- Step 6: Score correlation between teachers (pairwise for 2-teacher case) ----
    correlation_results = {}
    if len(teachers) == 2:
        t_a, t_b = teacher_names
        r, label = pearson_correlation(score_vectors[t_a], score_vectors[t_b])
        correlation_results[f"{t_a}_vs_{t_b}"] = {
            "pearson_r": round(r, 4) if r is not None else None,
            "interpretation": label,
            "n_samples": len(score_vectors[t_a]),
        }
        if r is not None:
            print(f"\nScore correlation ({t_a} vs {t_b}): r = {r:.4f}  [{label}]")

    # ---- Step 7: Build clean output (strip teacher_weights for training.py) ----
    output_dataset = [
        {
            "prompt":   item["prompt"],
            "chosen":   item["chosen"],
            "rejected": item["rejected"],
            "weight":   item["weight"],
        }
        for item in combined
    ]

    # ---- Step 8: Compile all statistics ----
    stats = {
        # Per-teacher sizes
        "teacher_dataset_sizes": {
            t: len(pm) for t, pm in zip(teacher_names, prompt_maps)
        },

        # Venn diagram
        "venn_diagram": {
            "intersection": len(intersection_prompts),
            "union":        len(union_prompts),
            "unique_per_teacher": {
                t: len(ps - intersection_prompts)
                for t, ps in zip(teacher_names, prompt_sets)
            },
        },

        # Intersection quality
        "intersection_rate": round(len(intersection_prompts) / max(len(union_prompts), 1), 4),

        # Score analysis
        "score_correlation": correlation_results,
        "high_discrepancy": {
            "threshold": discrepancy_threshold,
            "count": high_discrepancy_count,
            "fraction": round(high_discrepancy_count / max(len(combined), 1), 4),
            "top_10_examples": discrepancies[:10],
        },

        # Combined score distribution
        "combined_score_distribution": score_distribution_stats(
            [item["weight"] for item in combined]
        ),

        # Per-teacher score distributions (within intersection)
        "per_teacher_score_distributions": {
            t_name: score_distribution_stats(score_vectors[t_name])
            for t_name in teacher_names
        },

        # Final output
        "combined_dataset_size": len(output_dataset),
    }

    return output_dataset, stats


# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":

    # Validate config has teachers list
    if "teachers" not in cfg or not cfg["teachers"]:
        print("ERROR: No 'teachers' list found in config.yaml.")
        print("Add a 'teachers' section with at least 2 teachers, then re-run "
              "logit_linear_selection.py before combining.")
        sys.exit(1)

    teachers = cfg["teachers"]

    if len(teachers) < 2:
        print(f"ERROR: Need at least 2 teachers to combine, found {len(teachers)}.")
        sys.exit(1)

    teacher_names = [t.get("name") or sanitize(t["model"].split("/")[-1]) for t in teachers]

    # ---- Determine output path for combined dataset ----
    combine_cfg  = cfg.get("combination", {})
    custom_path  = combine_cfg.get("combined_dataset_path", "").strip()
    if custom_path:
        combined_dir = os.path.expanduser(os.path.dirname(custom_path))
        combined_path = os.path.expanduser(custom_path)
    else:
        # Default: {local_root}/combined_trunc{T}_q{Q}/
        trunc = shared_config["truncation_value"]
        quantile = shared_config["quantile"]
        combined_dir  = os.path.join(local_root, f"combined_trunc{trunc}_q{quantile}")
        combined_path = os.path.join(combined_dir, "preference_dataset.json")

    stats_path = os.path.join(combined_dir, "combine_stats.json")
    os.makedirs(combined_dir, exist_ok=True)

    print("=" * 60)
    print("combine_datasets.py – Multi-Teacher Dataset Combination")
    print("=" * 60)
    print(f"Teachers: {', '.join(teacher_names)}")
    print(f"Strategy: intersection + average score")
    print(f"Output:   {combined_path}")

    # ---- Combine ----
    combined_dataset, stats = combine_datasets(teachers)

    # ---- Print detailed report ----
    print("\n" + "=" * 60)
    print("COMBINATION REPORT")
    print("=" * 60)

    print("\n── Per-Teacher Dataset Sizes ──")
    for t_name, size in stats["teacher_dataset_sizes"].items():
        print(f"  {t_name}: {size:,} examples")

    print("\n── Venn Diagram ──")
    venn = stats["venn_diagram"]
    for t_name, unique_count in venn["unique_per_teacher"].items():
        total_for_teacher = stats["teacher_dataset_sizes"][t_name]
        print(f"  {t_name} only (not in others): {unique_count:,} "
              f"({100*unique_count/max(total_for_teacher,1):.1f}%)")
    print(f"  Intersection (in all teachers): {venn['intersection']:,}")
    print(f"  Union (in any teacher):         {venn['union']:,}")
    print(f"  Intersection rate: {stats['intersection_rate']*100:.1f}% of union")

    print("\n── Score Correlation ──")
    if stats["score_correlation"]:
        for pair, corr in stats["score_correlation"].items():
            r_str = f"{corr['pearson_r']:.4f}" if corr["pearson_r"] is not None else "N/A"
            print(f"  {pair}: r = {r_str}  [{corr['interpretation']}]")
    else:
        print("  (Only computed for exactly 2 teachers)")

    print("\n── Score Discrepancy (where teachers disagree) ──")
    hd = stats["high_discrepancy"]
    print(f"  Threshold: >{hd['threshold']}")
    print(f"  High-discrepancy examples: {hd['count']:,} "
          f"({hd['fraction']*100:.1f}% of intersection)")
    if hd["top_10_examples"]:
        print("  Top discrepant examples:")
        for ex in hd["top_10_examples"][:5]:
            print(f"    prompt: {ex['prompt']}")
            for t, w in ex["teacher_weights"].items():
                print(f"      {t}: {w:.4f}")
            print(f"      max diff: {ex['max_diff']}")

    print("\n── Per-Teacher Score Distribution (within intersection) ──")
    for t_name, dist in stats["per_teacher_score_distributions"].items():
        if dist:
            print(f"  {t_name}: min={dist['min']:.4f}  max={dist['max']:.4f}  "
                  f"mean={dist['mean']:.4f}  std={dist['std']:.4f}")

    print("\n── Combined Score Distribution ──")
    cd = stats["combined_score_distribution"]
    if cd:
        print(f"  Count:  {cd['count']:,}")
        print(f"  Min:    {cd['min']:.4f}")
        print(f"  Max:    {cd['max']:.4f}")
        print(f"  Mean:   {cd['mean']:.4f}")
        print(f"  Median: {cd['median']:.4f}")
        print(f"  Std:    {cd['std']:.4f}")
        print(f"  Q25/Q50/Q75/Q90: "
              f"{cd['q25']:.4f} / {cd['q50']:.4f} / {cd['q75']:.4f} / {cd['q90']:.4f}")

    print(f"\n── Final Combined Dataset ──")
    print(f"  Size: {stats['combined_dataset_size']:,} examples")
    print(f"  Sorted by combined (averaged) score descending.")

    print("\n" + "=" * 60)

    # ---- Save ----
    save_json_atomic(combined_dataset, combined_path)
    save_json_atomic(stats,            stats_path)

    print(f"Saved combined dataset → {combined_path}")
    print(f"Saved statistics       → {stats_path}")
    print("\nNext step: run 'python training.py' to train the student model.")
    print("(Update config.yaml → combination.combined_dataset_path to point training.py "
          "at the combined dataset, or pass it as a CLI argument.)")
