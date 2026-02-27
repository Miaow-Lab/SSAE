# Classifier Experiment

This folder contains the full pipeline for the first experiment in the paper: Probing with Classifier

The workflow is controlled by one config file:

- `configs/classifier.yaml`

Each entry script reads only its own config section (`classifier_data`, `combine_label`, `classifier_train`, `correctness_eval`, `statistic_baseline`).

## Pipeline Overview

1. Generate classifier data (`.npz`) and LLM judgment prompts (`*_judge.jsonl`).
2. Call external LLM API (not included in this repo) on `*_judge.jsonl`.
3. Merge LLM labels (`Correctness`, `Logicality`) back into `.npz`.
4. Train classifiers for different targets.
5. Evaluate `Correctness`/`Logicality` classifiers.
6. Compute statistic baselines.

## Step 0: Configure

Edit config first:

```bash
vim configs/classifier.yaml
```


## Step 1: Generate classifier data

Script:

```bash
bash scripts/run_classifier_data.sh configs/classifier.yaml
```

What it produces:

- `*.npz` with fields such as `latents`, `hints`, `step_lengths`, `begin_token_ids`, `correctness`, `logicality`
- `*_judge.jsonl` prompts for LLM-based correctness/logicality labeling

Optional override example:

```bash
bash scripts/run_classifier_data.sh configs/classifier.yaml \
  --set checkpoint_name=gsm8k-385k_Qwen2.5-0.5b_spar-10.pt \
  --set input_file=data/gsm8k_385K_valid.json \
  --set output_file=classifier_data/gsm8k_385K_valid_classifier_data
```


## Step 2: Call LLM API externally

Input:

- the `*_judge.jsonl` file from Step 1

Expected API output JSONL (at least):

- `id`
- `response` (or a custom response field)


## Step 3: Merge Correctness/Logicality labels

Command:

```bash
PYTHONPATH=. python classifier/combine_label.py --config configs/classifier.yaml
```

If your response text field is not `response`, set the response_key manually:

```bash
PYTHONPATH=. python classifier/combine_label.py --config configs/classifier.yaml --set response_key=content
```


## Step 4: Train classifiers

Script:

```bash
PROC=<num_gpus> bash scripts/run_classifier_train.sh configs/classifier.yaml
```

You can set different inputs and targets by changing `classifier_train.task`:

**inputs:**
- `Tr`
- `hints`

**targets:**
- `len`
- `token`
- `correctness`
- `logicality`

Notes:

- For `step length` and `first token ppl`, training loss is used directly as the main evaluation signal.
- For `correctness` and `logicality`, run Step 5 evaluation.


## Step 5: Evaluate Correctness/Logicality

Script:

```bash
bash scripts/run_correctness_eval.sh configs/classifier.yaml
```
This reports accuracy for both `correctness` and `logicality` classifiers.



## Step 6: Statistic baselines

Command:

```bash
PYTHONPATH=. python classifier/statistic_baseline.py --config configs/classifier.yaml
```

This reports baselines for all four labels:

- step length (mean/std)
- first token ppl (entropy/perplexity)
- correctness (positive ratio)
- logicality (positive ratio, when enabled)
