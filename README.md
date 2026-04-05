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

## Typical Workflow

Generate and analyze a contrastive corpus:

```bash
python src/adv_steering/run_poscon_negcon_pipeline.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --backend causal_lm \
  --dataset data/llama_happy_sad_corpus.jsonl \
  --poscon-label happy \
  --negcon-label sad
```

Run qualitative steering on a chosen residual bundle:

```bash
python src/adv_steering/qualitative_poscon_negcon_steering.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --backend causal_lm \
  --steering-file runs/<run>/poscon_negcon_residuals.pt \
  --layer <layer> \
  --poscon-scale 8.0 \
  --negcon-scale -8.0 \
  --prompts data/qualitative_happy_sad_prompts.txt
```

Run GCG-style suffix optimization:

```bash
python src/adv_steering/rep_suffix_attack.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --steering-file runs/<run>/poscon_negcon_residuals.pt \
  --layer <layer> \
  --objective-type cosine \
  --n-plus "Please tell me a happy story about Saturn." \
  --n-minus "Please tell me a sad story about Saturn."
```

Run Largo-style suffix optimization:

```bash
python src/adv_steering/rep_suffix_attack_largo.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --steering-file runs/<run>/poscon_negcon_residuals.pt \
  --layer <layer> \
  --objective-type steered_ce \
  --neutral-prompt "Please tell me a story about Saturn." \
  --positive-target "Here is a happy story about Saturn." \
  --negative-target "Here is a sad story about Saturn."
```

## What This Repo Is Trying To Show

The central claim being investigated is not just that suffixes can jailbreak a model in the usual sense. It is more specific:

- suffixes may be able to alter the local representation geometry so that a safety-relevant steering direction becomes directionally unreliable

If that claim holds, then steering-based safety layers may be vulnerable to a new family of attacks that operate after post-training and exploit the model’s own representation dynamics.
