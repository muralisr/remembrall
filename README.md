# Remembrall (Starter Kit)

This repository is a **reference implementation and scaffold** for Remembrall-style on-device continuous learning on SoC GPUs.
It includes well-documented, modular Python code and markdown docs so you can adapt it to your hardware and models.

> **Note:** This is a *lightweight* educational/reference codebase designed to match the paper's design.
It avoids heavyweight training code and external data; the interfaces are ready to integrate with PyTorch / ExecuTorch and your models.

## Features
- **Base + Specialized models** loaded simultaneously (memory-for-compute tradeoff)
- **Embedding-guided sampling** (dedup via online SCPS-style selection; top-k uncertainty via entropy)
- **Inference-aware retraining** with **opportunistic early stopping**
- **Lightweight base model updates** via **Reptile + dynamic EWMA**
- Pluggable schedulers and metrics hooks

## Quick Start
```bash
# (1) Install (editable)
pip install -e .

# (2) Run a dry-run demo (no GPU required; uses stubs and random tensors)
python scripts/run_demo.py --duration 120 --fps 15

# (3) Explore configs
ls configs/
```

## Project Layout
```
remembrall_starter_kit/
├─ src/remembrall/
│  ├─ __init__.py
│  ├─ core.py                 # Orchestrates the pipeline
│  ├─ sampler.py              # Dedup + uncertainty-ranked selection
│  ├─ retrainer.py            # Few-shot retraining loop + early stopping
│  ├─ base_update.py          # Reptile + dynamic EWMA base updates
│  ├─ scheduler.py            # Time-sharing & cadence logic
│  ├─ models/
│  │  ├─ registry.py          # Model loader + wrappers
│  │  ├─ stubs.py             # Lightweight model stubs
│  ├─ utils/
│  │  ├─ metrics.py           # Throughput + accuracy counters
│  │  ├─ embeddings.py        # Cosine similarity, distances, PCA helper
│  │  ├─ logging.py           # Structured logging
├─ scripts/
│  └─ run_demo.py             # End-to-end dry-run with stubs
├─ configs/
│  ├─ default.yaml
│  ├─ nano.yaml
│  └─ adreno.yaml
├─ docs/
│  ├─ design.md               # High-level design and tradeoffs
│  ├─ evaluation.md           # What to measure; suggested tables/figures
│  ├─ memory_runtime.md       # Memory footprint & runtime breakdown notes
│  └─ limitations.md          # Limitations & future directions (as discussed)
├─ LICENSE
├─ pyproject.toml
├─ CITATION.cff
└─ CONTRIBUTING.md
```

## License
MIT © 2025
