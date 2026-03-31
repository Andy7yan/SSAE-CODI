# SSAE-CODI

## Current Goal

The current focus is keeping a minimal CODI scaffold working on Katana.

The goal is to provide a minimal, auditable scaffold that can:

- load the target CODI checkpoint,
- run one minimal inference pass,
- expose the latent-thought generation path for inspection,
- save hidden dumps in a format that later analysis code can reuse for hidden-state extraction and dataset building.

This scaffold does not implement training, probes, or SSAE dataset construction yet.

## Default Target

- Model family: CODI
- Default checkpoint: `zen-E/CODI-gpt2`
- Default platform: Katana + Linux
- Python target: 3.10

## Project Structure

```text
.
|-- README.md
|-- .env
|-- pyproject.toml
|-- requirements-katana.txt
|-- data/
|   `-- gsm8k_debug.jsonl
|-- src/
|   |-- config.py
|   |-- io_utils.py
|   |-- inspect_latent.py
|   |-- load_model.py
|   |-- logging_utils.py
|   `-- run_inference.py
`-- tests/
    `-- test_smoke.py
```

## Output Layout

```text
/srv/scratch/z5534565/ssae-codi/
|-- hf-home/
|   |-- hub/
|   |-- datasets/
|   `-- saved-datasets/
`-- smoke/
    `-- smoke_summary.json
```

The active Katana entry point is now the smoke test in `tests/test_smoke.py`. It caches model and dataset artifacts under `HF_HOME` and writes a single smoke summary JSON under `SMOKE_OUTPUT_ROOT`.

## Setup

```bash
python3 -m pip install -r requirements-katana.txt
python3 -m pip install -e .
```

If you need to pre-download wheels on another machine, use:

```bash
python3 -m pip download -r requirements-katana.txt -d wheelhouse
python3 -m pip install --no-index --find-links wheelhouse -r requirements-katana.txt
```

If Hugging Face authentication is required, export `HF_TOKEN` before running.

Use the repo-root `.env` on Katana instead of keeping ad-hoc shell exports around:

```bash
set -a
source .env
set +a
mkdir -p "$HF_HOME" "$SMOKE_OUTPUT_ROOT"
```

The current `.env` values are:

```bash
HF_TOKEN=
HF_HOME=/srv/scratch/z5534565/ssae-codi/hf-home
HF_HUB_CACHE=/srv/scratch/z5534565/ssae-codi/hf-home/hub
HF_DATASETS_CACHE=/srv/scratch/z5534565/ssae-codi/hf-home/datasets
SMOKE_OUTPUT_ROOT=/srv/scratch/z5534565/ssae-codi/smoke
MODEL_REPO=zen-E/CODI-gpt2
DATASET_REPO=openai/gsm8k
DATASET_CONFIG=main
```

## Run

Smoke test:

```bash
set -a
source .env
set +a
torchrun tests/test_smoke.py
```

## Smoke Coverage

The smoke test currently checks only two things:

- the CODI repository can be resolved and cached from Hugging Face under `HF_HOME`;
- the GSM8K dataset can be downloaded once, saved under `HF_HOME/saved-datasets`, and reused on later runs without re-downloading.

## Limits

- The current code only targets inference and inspection.
- No training logic is included.
- `seed-only` capture is the safest supported mode.
- `per-latent-step` is implemented for the official-style CODI latent loop and falls back conservatively for generic upstream models.
- If upstream checkpoint packaging changes, the loader may need a small adapter update.

## Note

The original research-background README content is preserved below. The scaffold documentation above is the operational entry point for the current repository state.

# Interpreting CODI Latent Chain-of-Thought with Step-wise Sparse Autoencoders

## Overview

This repository contains the codebase for a UNSW COMP3902 research project supervised by **Flora Salim**.

The project investigates the internal mechanisms of **continuous-space Chain-of-Thought (CoT)** in **CODI** by applying **Step-wise Sparse Autoencoders (SSAE)** as an interpretability tool. More specifically, the goal is to study whether SSAE can help reveal how latent reasoning states evolve across intermediate reasoning steps, and whether these learned sparse features can provide a more structured view of CODI's internal computation.

This project is positioned at the intersection of:

- mechanistic interpretability,
- latent / implicit reasoning in language models,
- sparse representation learning.

---

## Research Background

Recent work on latent reasoning has explored alternatives to standard text-based Chain-of-Thought, where intermediate reasoning is carried not only by explicit natural language tokens but also by continuous internal representations. CODI is one such approach: instead of relying solely on visible textual reasoning traces, it distils reasoning into continuous latent states embedded within the model's computation.

While this line of work is promising for efficiency and reasoning performance, it also introduces a major interpretability challenge: **what exactly is represented in these latent reasoning states, and how does that representation change over the course of problem solving?**

This project approaches that question through **Step-wise Sparse Autoencoders (SSAE)**. The core idea is to treat latent reasoning as a sequence of internally evolving steps, and to use sparse feature learning to analyse what information is incrementally encoded at different stages.

---

## Research Goal

The main objective of this project is:

> **To use Step-wise Sparse Autoencoders to interpret the internal latent reasoning process of CODI.**

This broad goal can be broken down into four practical research aims:

1. **Reproduce or set up a working CODI pipeline** for latent reasoning on mathematical word problems.
2. **Extract and store latent reasoning steps** generated by CODI.
3. **Adapt and train SSAE** on these step-wise latent representations.
4. **Analyse the learned sparse features** to study how reasoning information is organised, compressed, and transformed across steps.

---

## Why CODI and SSAE

### Why CODI

CODI is a suitable target for this project because it provides a relatively concrete and operational form of latent reasoning: intermediate reasoning is represented in continuous space rather than only in explicit text. This makes it a useful testbed for studying how hidden reasoning states can be interpreted mechanistically.

### Why SSAE

Standard sparse autoencoders are often applied to activations at a single layer or position. In contrast, **Step-wise Sparse Autoencoders** are designed to analyse **knowledge increments across reasoning steps**, making them a particularly relevant candidate for latent Chain-of-Thought settings.

The combination of CODI and SSAE therefore offers a plausible pathway towards a more step-structured analysis of latent reasoning.

---

## Current Project Scope

At the current stage, the project focuses on the following scope:

- **Target model family:** CODI
- **Primary interpretability method:** Step-wise Sparse Autoencoder (SSAE)
- **Primary task domain:** mathematical reasoning
- **Initial dataset focus:** GSM8K-based setup
- **Primary emphasis:** engineering implementation, latent extraction, SSAE adaptation, and interpretability-oriented analysis

This repository is currently in an early setup stage and will be expanded incrementally as the implementation progresses.

---

## Planned Workflow

The current working plan is organised into four phases:

### Phase 1 — CODI Setup
- Set up the CODI codebase and dependencies
- Confirm basic inference and latent reasoning generation
- Establish reproducible local and HPC workflows

### Phase 2 — Latent Data Generation
- Run CODI on the target dataset
- Extract intermediate latent reasoning representations
- Store latent steps in a reusable and analysis-friendly format

### Phase 3 — SSAE Adaptation and Training
- Implement or adapt SSAE for CODI latent steps
- Train sparse models on step-wise latent representations
- Validate reconstruction quality and sparsity behaviour

### Phase 4 — Analysis and Interpretation
- Inspect learned sparse features
- Compare features across reasoning steps
- Study whether features correspond to interpretable reasoning components or transitions

---

## Current Project Decisions

This section records implementation choices that have already been fixed. It is intended to remain **extendable** as the project evolves.

### Confirmed so far

- **Project type:** UNSW COMP3902 research project
- **Supervisor:** Flora Salim
- **Core research direction:** mechanistic interpretability for latent reasoning
- **Target system:** CODI
- **Interpretability method:** Step-wise Sparse Autoencoder (SSAE)
- **Main implementation language:** Python
- **Primary deep learning stack:** PyTorch + HuggingFace Transformers
- **Linting / formatting:** `ruff` and `ruff format`
- **Config style:** typed Python config object with in-code defaults
- **Latent storage preference:** `safetensors` instead of pickle
- **Local development environment:** Windows + PowerShell + VS Code
- **Remote compute environment:** UNSW Katana HPC
- **Cluster scheduler:** PBS

### Expected to be refined later

- exact CODI checkpoint / variant selection
- latent extraction format and schema
- SSAE input representation details
- experiment tracking structure
- evaluation protocol for interpretability claims
- run directory and checkpoint naming conventions

---

## Repository Status

This repository is currently being initialised.

At the moment, the main purpose of the repository is to provide a clean and reproducible foundation for:

- CODI environment setup,
- latent reasoning extraction,
- SSAE training,
- and subsequent interpretability analysis.

As implementation progresses, this repository will be populated with code, configuration files, experiment scripts, and analysis outputs.

---

## References

- **CODI**: *Compressing Chain-of-Thought into Continuous Space via Self-Distillation*  
  arXiv:2502.21074

- **SSAE**: *Step-wise Sparse Autoencoders*  
  arXiv:2603.03031

---

## Disclaimer

This repository is a research codebase under active development.  
Implementation details, experiment design, and analysis methodology may change as the project is refined.
