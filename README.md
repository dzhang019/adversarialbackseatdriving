# Adversarial Backseat Driving

This project studies a new jailbreak-style attack on representation steering.

The core question is:

- can we find suffixes that make a steering vector apply in the wrong direction?

In other words, if a model has a steering vector for a concept such as truthfulness, can an adversarial suffix cause that same vector to push the model toward the opposite behavior, such as lying?

## Main Hypothesis

This work is motivated by two existing observations:

- in-context learning can substantially reshape internal representation spaces in large language models
- carefully chosen suffixes can reliably trigger undesirable behaviors, including universal jailbreak behavior

Prior work on adversarial prompting and jailbreaks, including Universal and Transferable Attacks, AdvPrompter, and LARGO, shows that short token sequences can redirect model behavior in surprisingly robust ways.

The hypothesis in this repository is:

- there exist suffixes that flip model behavior along a chosen steering direction, so that applying a steering vector causes the opposite of the intended effect

This matters because steering is one of the simplest and most accessible interventions used to shape model behavior in production systems. If suffixes that invert steering are easy to find, then they create an attack surface against a class of post-training defenses that are often treated as lightweight safety controls.

## Experimental Idea

The experiments in this repository follow a simple pattern:

1. Build a contrastive dataset for a concept pair, such as happy vs sad.
2. Extract residual-stream states and estimate a steering direction.
3. Verify that steering at the chosen layer shifts generation in the intended direction.
4. Search for suffixes that make the model behave as though the steering direction has been reversed.

The repository currently focuses on:

- positive concept vs negative concept residual analysis
- qualitative steering evaluation
- suffix search with both discrete GCG-style optimization and continuous Largo-style optimization

## Current Research Direction

The most active setup in this repo uses:

- `meta-llama/Llama-3.1-8B-Instruct`
- a `happy` vs `sad` concept contrast
- prompt-state residuals at the assistant-prefill boundary

The main adversarial question is not simply whether a suffix changes output, but whether it changes the representation geometry in a way that causes steering to misfire.

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you need access to gated Hugging Face checkpoints, put your token in `.env`:

```bash
HF_TOKEN=your_hf_token_here
```

## Main Artifacts

The most important outputs are:

- residual bundles such as `poscon_negcon_residuals.pt`
- layer summaries such as `layer_summary.json`
- suffix search traces under `runs/.../suffixes/`
- qualitative steering generations under `runs/.../steering_generations/`

## Command Cookbook

### 1. Compute Diff Vectors And Find Steering Vector Candidates

End-to-end pipeline:

```bash
.venv/bin/python src/adv_steering/run_poscon_negcon_pipeline.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --backend causal_lm \
  --concepts data/concepts_200.txt \
  --dataset data/llama_happy_sad_corpus.jsonl \
  --poscon-label happy \
  --negcon-label sad \
  --mode story \
  --output-dir runs
```

What it does:

- generates a contrastive corpus for the chosen concept pair
- collects prompt-boundary residuals for positive and negative prompts
- computes mean difference vectors by layer
- ranks layers and saves steering-vector candidates

Useful flags:

- `--model`: model whose residual space you want to analyze
- `--backend`: usually `causal_lm`
- `--concepts`: newline-delimited concepts file
- `--dataset`: where to save the generated corpus
- `--poscon-label`, `--negcon-label`: concept contrast
- `--mode`: prompt style such as `story`
- `--output-dir`: root directory for saved runs

Only generate the corpus:

```bash
.venv/bin/python src/adv_steering/generate_poscon_negcon_corpus.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --backend causal_lm \
  --concepts data/concepts_200.txt \
  --output-path data/llama_happy_sad_corpus.jsonl \
  --poscon-label happy \
  --negcon-label sad \
  --mode story
```

Only analyze residuals:

```bash
.venv/bin/python src/adv_steering/analyze_poscon_negcon_residuals.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --backend causal_lm \
  --dataset data/llama_happy_sad_corpus.jsonl \
  --output-dir runs
```

Rerank an existing residual bundle without recollecting:

```bash
.venv/bin/python src/adv_steering/analyze_poscon_negcon_residuals.py \
  --residual-file runs/poscon_negcon_20260407_220348/poscon_negcon_residuals.pt \
  --dataset data/llama2_happy_sad_corpus.jsonl \
  --output-dir runs/poscon_negcon_20260407_220348
```

### 2. Residual / Steering Analysis / Geometry

Plot poscon vs negcon residual geometry for one layer:

```bash
.venv/bin/python src/adv_steering/plot_poscon_negcon_geometry.py \
  --dataset data/llama_happy_sad_corpus.jsonl \
  --residual-file runs/poscon_negcon_20260323_233220/poscon_negcon_residuals.pt \
  --layer 10
```

What it does:

- fits a 2D PCA to the positive and negative residuals at one layer
- plots those residuals
- saves per-example cosine-to-average-direction scores
- reports steering-vector alignment with `PC1` and `PC2`

Useful flags:

- `--dataset`: JSONL corpus used for labels and prompts
- `--residual-file`: saved residual bundle
- `--layer`: layer to analyze
- `--output-dir`: optional custom output location

Project emotion-token residuals from a long context into the same PCA basis:

```bash
.venv/bin/python src/adv_steering/plot_context_token_residuals.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --backend causal_lm \
  --residual-file runs/poscon_negcon_20260323_233220/poscon_negcon_residuals.pt \
  --context-file data/happy_sad_prompts3.txt \
  --line 1 \
  --layer 10
```

What it does:

- runs a long prompt through the model
- extracts token-level residuals at the chosen layer
- selects matched emotion tokens
- projects them onto the PCA basis from the saved residual bundle

Useful flags:

- `--context-file`: text file containing the long prompt
- `--line`: use one specific line instead of the whole file
- `--layer`: residual layer to project

### 3. Evaluation

Run qualitative steering with no suffix:

```bash
.venv/bin/python src/adv_steering/qualitative_poscon_negcon_steering.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --backend causal_lm \
  --steering-file runs/poscon_negcon_20260323_233220/poscon_negcon_residuals.pt \
  --layer 10 \
  --poscon-scale 5.0 \
  --negcon-scale -5.0 \
  --prompts data/qualitative_happy_sad_prompts.txt
```

With an exact discrete suffix:

```bash
.venv/bin/python src/adv_steering/qualitative_poscon_negcon_steering.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --backend causal_lm \
  --steering-file runs/poscon_negcon_20260323_233220/poscon_negcon_residuals.pt \
  --layer 10 \
  --poscon-scale 5.0 \
  --negcon-scale -5.0 \
  --prompts data/qualitative_happy_sad_prompts.txt \
  --suffix runs/poscon_negcon_20260323_233220/suffixes/rep_suffix_20260326_183248.json \
  --exact-suffix-ids
```

With a continuous soft prompt from a specific optimization step:

```bash
.venv/bin/python src/adv_steering/qualitative_poscon_negcon_steering.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --backend causal_lm \
  --steering-file runs/poscon_negcon_20260323_233220/poscon_negcon_residuals.pt \
  --layer 10 \
  --poscon-scale 5.0 \
  --negcon-scale -5.0 \
  --prompts data/qualitative_happy_sad_prompts.txt \
  --soft-prompt runs/poscon_negcon_20260323_233220/suffixes/rep_suffix_soft_prompt_20260408_133204.json \
  --step 10
```

What it does:

- runs baseline, positive-steered, and negative-steered generations
- optionally inserts a discrete suffix or continuous soft prompt inside the user message
- can score teacher-forced CE against fixed target responses

Useful flags:

- `--steering-file`: steering vector bundle
- `--layer`: steering layer
- `--poscon-scale`, `--negcon-scale`: steering strengths
- `--prompts`: newline-delimited evaluation prompts
- `--suffix`: discrete suffix artifact
- `--exact-suffix-ids`: use saved suffix token ids exactly
- `--soft-prompt`: continuous soft prompt artifact
- `--step`: choose a trace step for suffix or soft prompt
- `--response-targets`: fixed response JSON for CE scoring
- `--show-top-logits`: print next-token distributions

### 4. Suffix Optimization

Discrete GCG-style suffix optimization:

```bash
.venv/bin/python src/adv_steering/rep_suffix_attack.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --steering-file runs/poscon_negcon_20260323_233220/poscon_negcon_residuals.pt \
  --layer 10 \
  --objective-type cosine \
  --n-plus "Please tell me a happy story about Saturn." \
  --n-minus "Please tell me a sad story about Saturn." \
  --suffix-length 40 \
  --steps 300 \
  --top-k 256 \
  --batch-size 384
```

What it does:

- optimizes a discrete suffix with one-token edits
- uses exact objective checks before accepting a new suffix
- now early-stops if the current one-token neighborhood is exhausted

Useful flags:

- `--objective-type`: `dot`, `cosine`, or `steered_ce`
- `--suffix-length`: number of suffix tokens
- `--steps`: requested optimization steps
- `--top-k`: candidate replacements kept per position
- `--batch-size`: candidate one-token edits evaluated per step
- `--resume-from`: resume from an older suffix artifact

Discrete steered CE example:

```bash
.venv/bin/python src/adv_steering/rep_suffix_attack.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --steering-file runs/poscon_negcon_20260323_233220/poscon_negcon_residuals.pt \
  --layer 10 \
  --objective-type steered_ce \
  --neutral-prompt "Please tell me a story about Saturn." \
  --n-plus "Please tell me a happy story about Saturn." \
  --n-minus "Please tell me a sad story about Saturn." \
  --positive-target "Here is a happy story about Saturn." \
  --negative-target "Here is a sad story about Saturn." \
  --steering-scale 5.0
```

Continuous Largo-style optimization:

```bash
.venv/bin/python src/adv_steering/rep_suffix_attack_largo.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --steering-file runs/poscon_negcon_20260323_233220/poscon_negcon_residuals.pt \
  --layer 10 \
  --objective-type steered_ce \
  --neutral-prompt "Please tell me a story about Saturn." \
  --n-plus "Please tell me a happy story about Saturn." \
  --n-minus "Please tell me a sad story about Saturn." \
  --positive-target "Here is a happy story about Saturn." \
  --negative-target "Here is a sad story about Saturn." \
  --steering-scale 5.0 \
  --eval-prompts-file data/qualitative_happy_sad_prompts.txt \
  --hapsad-wordbank-file data/hapsad_wordbank.json
```

What it does:

- optimizes a continuous suffix matrix
- interprets it back into discrete text after each outer round
- evaluates held-out prompts for early stopping

Useful flags:

- `--outer-steps`: summarize-and-reinterpret rounds
- `--inner-steps`: gradient steps per round
- `--summary-prompt`: user message for the interpretation prompt
- `--eval-prompts-file`: held-out prompts
- `--success-proportion`: early stop threshold
- `--init-mode`: `zeros` or `random_tokens`

Continuous soft-prompt optimization without projection:

```bash
.venv/bin/python src/adv_steering/rep_suffix_attack_soft_prompt.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --steering-file runs/poscon_negcon_20260323_233220/poscon_negcon_residuals.pt \
  --layer 10 \
  --objective-type steered_ce \
  --neutral-prompt "Please tell me a story about Saturn." \
  --n-plus "Please tell me a happy story about Saturn." \
  --n-minus "Please tell me a sad story about Saturn." \
  --positive-target "Here is a happy story about Saturn." \
  --negative-target "Here is a sad story about Saturn." \
  --steering-scale 5.0 \
  --save-all-steps
```

What it does:

- optimizes a continuous soft prompt directly
- does not project back into token space
- can optionally save the full soft prompt matrix at every step

Useful flags:

- `--inner-steps`: optimization steps
- `--init-mode`: `zeros` or `random_tokens`
- `--save-all-steps`: store the full matrix at every step

Run a suffix sweep across many discrete configs:

```bash
.venv/bin/python src/adv_steering/run_rep_suffix_sweep.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --steering-file runs/poscon_negcon_20260323_233220/poscon_negcon_residuals.pt \
  --layer 10 \
  --n-plus "Please tell me a happy story about Saturn." \
  --n-minus "Please tell me a sad story about Saturn." \
  --suffix-lengths 20,40 \
  --objectives dot,cosine \
  --steps-list 100,300 \
  --top-k-list 64,256 \
  --batch-size-list 128,384 \
  --restarts 3
```

## What This Repo Is Trying To Show

The central claim being investigated is not just that suffixes can jailbreak a model in the usual sense. It is more specific:

- suffixes may be able to alter the local representation geometry so that a safety-relevant steering direction becomes directionally unreliable

If that claim holds, then steering-based safety layers may be vulnerable to a new family of attacks that operate after post-training and exploit the model’s own representation dynamics.
