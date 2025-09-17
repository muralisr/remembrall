# Design Overview

This starter kit mirrors the Remembrall design:
- Maintain **base** and **specialized** models in memory.
- **Sampling**: online dedup in embedding space + uncertainty prioritization.
- **Retraining**: few-shot fine-tune from base; **opportunistic early stopping** balances retrain benefit vs. drift.
- **Base Update**: **Reptile** with **dynamic EWMA** (scene-similarity gate).
- **Scheduling**: periodic cadence (default 30s), serialized GPU; short pauses for retraining.

See inline docstrings for algorithmic details.
