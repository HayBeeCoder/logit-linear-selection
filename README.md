# Logit-Linear-Selection Example

Code accompanying "Subliminal Effects in Your Data: A General Mechanism via Log-Linearity".
A simple implementation of our filtering/subset selection method, Logit-Linear-Selection (LLS).

We use the `stack_exchange_paired` subset of [Tulu 2.5](https://huggingface.co/datasets/allenai/tulu-2.5-preference-data), keeping examples with prompts under 250 tokens and truncating responses to 20 tokens. This is used to build LLS preference datasets and train a student model with DPO.

**Requirements:** `torch`, `transformers`, `datasets`, `accelerate`, `trl`, `peft`, `numpy`, `pyyaml`, `tqdm`
```bash
pip install -r requirements.txt
```

See `requirements.txt` for tested versions. Requires access to [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) via HuggingFace.

## Setup

1. Set `local_root` in `/home/runner/work/logit-linear-selection/logit-linear-selection/config.yaml` to your desired output directory.
2. Ensure `HF_HOME` and `HF_TOKEN` environment variables are set.
3. Configure either:
   - **Multi-teacher mode** via `teachers:` (recommended), or
   - **Single-teacher fallback** via `teacher_model` + `system_prompt`.

## Usage

### Multi-teacher workflow (recommended)

1. Generate per-teacher LLS datasets:
```bash
python logit_linear_selection.py
```
This writes one `preference_dataset.json` and `stats.json` per teacher, and skips recomputation when cached datasets already exist.

2. Combine teacher datasets (intersection + average score):
```bash
python combine_datasets.py
```
This writes:
- combined `preference_dataset.json`
- `combine_stats.json`

3. Run DPO training on the combined dataset:
```bash
python training.py
```
`training.py` first checks `combination.combined_dataset_path`, then the default combined path, then falls back to legacy single-teacher output.

### Single-teacher workflow (legacy compatible)

If `teachers:` is removed/empty in config, run:
```bash
python logit_linear_selection.py
python training.py
```

## Dataset format

Generated preference datasets are lists of:
```text
{prompt, chosen, rejected, weight}
```

Legacy entries like `[prompt, chosen, rejected]` are still accepted by `training.py` and `combine_datasets.py` for backward compatibility.

## Multi-GPU / Multi-Node

The code uses HuggingFace Accelerate and extends naturally to multi-GPU and multi-node setups:
```bash
accelerate launch --num_processes <NUM_GPUS> logit_linear_selection.py
accelerate launch --num_processes <NUM_GPUS> combine_datasets.py
accelerate launch --num_processes <NUM_GPUS> training.py
```

For SLURM clusters, wrap with `srun` to ensure proper GPU allocation. See [Accelerate documentation](https://huggingface.co/docs/accelerate) for details.
