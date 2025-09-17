from __future__ import annotations
import time
from dataclasses import dataclass

@dataclass
class Cadence:
    retrain_period_s: float = 30.0
    _last: float = 0.0

    def should_retrain(self) -> bool:
        now = time.time()
        if (now - self._last) >= self.retrain_period_s:
            self._last = now
            return True
        return False
