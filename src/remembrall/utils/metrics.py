from __future__ import annotations
import time
from dataclasses import dataclass, field

@dataclass
class ThroughputMeter:
    window_s: float = 5.0
    _count: int = 0
    _t0: float = field(default_factory=time.time)

    def tick(self, n=1):
        self._count += n

    def fps(self) -> float:
        dt = time.time() - self._t0
        return self._count / dt if dt > 0 else 0.0

@dataclass
class RuntimeBreakdown:
    inference_s: float = 0.0
    retrain_s: float = 0.0
    maintenance_s: float = 0.0

    def as_share(self):
        total = self.inference_s + self.retrain_s + self.maintenance_s
        if total <= 0: return dict(inference=0, retrain=0, maintenance=0)
        return {
            "inference": self.inference_s/total,
            "retrain": self.retrain_s/total,
            "maintenance": self.maintenance_s/total,
        }
