# Memory and Runtime Notes

- Two models in memory (base + specialized). Footprint bounded by architecture sizes.
- Embedding buffers and selection queues pre-allocated with caps.
- Typical throughput: 15–20 FPS (device-dependent).
- Retraining: 2–6 s with early stopping on few-shot batches.
- Compute split (indicative): >85% inference, remainder retraining + maintenance.
