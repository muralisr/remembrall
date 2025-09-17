# Evaluation Guide

Report:
- Accuracy / mAP vs. time
- Retraining duration distribution (e.g., 2–6 s typical with early stopping)
- Throughput (FPS) during steady-state and retrain windows
- Compute breakdown: inference / retraining / maintenance
- Memory footprint over time (bounded; two models + buffers)

Suggested figures: CDF of retrain time; stacked bars for compute share.
