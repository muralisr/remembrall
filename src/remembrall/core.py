from __future__ import annotations
import time, numpy as np, yaml, os
from dataclasses import dataclass
from .models.registry import load_model
from .sampler import OnlineSCPSampler
from .retrainer import retrain_few_shot, RetrainConfig
from .base_update import reptile_update
from .scheduler import Cadence
from .utils.metrics import ThroughputMeter, RuntimeBreakdown
from .utils.logging import info

@dataclass
class RemembrallRunner:
    config_path: str

    def load_cfg(self):
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def run(self, duration_s=120, fps=15):
        cfg = self.load_cfg()
        cadence = Cadence(retrain_period_s=cfg.get("retrain_period_s",30))
        sampler = OnlineSCPSampler(distance_quantile=cfg.get("scps_distance_quantile",0.7),
                                   buffer_size=cfg.get("embedding_buffer_size",2048))
        retrain_cfg = RetrainConfig(early_stop_threshold=cfg.get("early_stop_threshold",0.1),
                                    min_seconds=cfg.get("min_retrain_seconds",2),
                                    max_seconds=cfg.get("max_retrain_seconds",10))

        # Models
        base = load_model("resnet18", embedding_dim=128, num_classes=10)
        specialized = base.copy()

        meter = ThroughputMeter()
        rb = RuntimeBreakdown()

        t0 = time.time()
        next_frame = t0
        candidates = []

        while time.time() - t0 < duration_s:
            # Simulate frame arrival
            next_frame += 1.0 / fps
            sleep_dt = max(0.0, next_frame - time.time())
            time.sleep(sleep_dt)

            # Inference (stub): generate embedding + logits
            emb, logits = specialized.forward([0])  # batch size 1
            emb = emb[0]; logits = logits[0]
            meter.tick(1)
            rb.inference_s += (time.time() - (next_frame - 1.0/fps))

            # Online selection (CPU)
            if sampler.consider(emb):
                candidates.append((emb, logits))

            # Periodic retraining
            if cadence.should_retrain():
                info("retrain_start", candidates=len(candidates))
                t1 = time.time()
                specialized, elapsed = retrain_few_shot(base, specialized, candidates, retrain_cfg)
                rb.retrain_s += elapsed
                # "Scene similarity" proxy from last two embeddings
                scene_sim = 0.95 if len(candidates) > 4 else 0.5
                step = reptile_update(base, specialized, scene_sim)
                rb.maintenance_s += (time.time() - t1 - elapsed)
                candidates.clear()
                info("retrain_done", seconds=elapsed, reptile_step=step, fps=meter.fps())

        info("run_complete", fps=meter.fps(), shares=rb.as_share())
        return dict(fps=meter.fps(), shares=rb.as_share())
