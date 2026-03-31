# Project: Interpreting CODI Latent CoT with Step-wise SAE

## Context
- Goal: Use Step-wise Sparse Autoencoders (SSAE) to interpret
  continuous-space Chain-of-Thought representations produced by CODI
  (arXiv:2502.21074)
- SSAE reference: arXiv:2603.03031
- CODI variant: GSM8K-Aug (best-performing per paper)
- Dataset: GSM8K (train split for latent generation, test for eval)

## Tech Stack
- Python 3.12.10, PyTorch, HuggingFace Transformers
- Linting: `ruff`; Formatting: `ruff format`
- Config: YAML files + dataclasses or pydantic for typed config
- Latent storage format: safetensors (not pickle)

## Two Environments

### Local (Windows + PowerShell + VS Code)
- For coding, debugging, small-scale testing
- All CLI examples use PowerShell syntax

### Katana (UNSW HPC, Linux)
- For training, inference at scale, evaluation
- Scheduler: **PBS** (`qsub`, `qstat`, `qdel`) — NOT SLURM
- Job scripts: `.pbs` files with `#PBS` directives
- Codebase on Katana: `/home/ssae-codi`
- Scratch (data/cache/runs/eval): `/srv/scratch/$USER/ssae-codi`

## Katana Conventions
- Never write logs, checkpoints, or results back to codebase
- All large outputs go to scratch under `runs/<run-name>/`
- Each run directory contains: `logs/`, `checkpoints/`, `weights/`
- `.pbs` structure: PBS directive block first, then execution block
  (cd to codebase → init env → create run dir → launch program)
- PBS template:
  ```bash
  #PBS -N <job_name>
  #PBS -l select=<nodes>:ncpus=<cpus>:ngpus=<gpus>:mem=<memory>
  #PBS -l walltime=HH:MM:SS
  #PBS -j oe
  #PBS -o <log_path>

  cd /home/ssae-codi
  # load modules, activate venv, export env vars
  RUN_DIR=/srv/scratch/$USER/ssae-codi/runs/<run-name>
  mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints" "$RUN_DIR/weights"
  # launch main program
  ```
- Start with minimal resource requests; scale up after validation
- Complex logic belongs in Python scripts, not `.pbs` files

## Code Conventions
- English-only comments and docstrings (no Chinese in code)
- Type hints, Google-style docstrings
- Scripts use argparse with sensible defaults
- When writing Katana jobs, produce `.pbs` files with resource estimates
- Always check existing code before creating new files
- If unsure about CODI internals, say so — do not hallucinate architecture details

## Project Phases
1. CODI setup: pull corpus, weights, write basic inference code
2. Latent generation: run CODI on GSM8K, extract and store latent thinking steps on Katana
3. Adapted SSAE: implement and train SSAE on latent data
4. Analysis: use trained SSAE to interpret CODI internals