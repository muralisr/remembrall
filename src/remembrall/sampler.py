from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

@dataclass
class OnlineSCPSampler:
    distance_quantile: float = 0.7
    buffer_size: int = 2048
    embeds: list[np.ndarray] = field(default_factory=list)

    def consider(self, emb: np.ndarray) -> bool:
        if len(self.embeds) == 0:
            self.embeds.append(emb)
            return True
        X = np.stack(self.embeds + [emb], axis=0)
        dists = np.sqrt(((X[-1] - X[:-1])**2).sum(axis=-1) + 1e-8)
        thr = np.quantile(dists, self.distance_quantile) if len(dists) > 5 else dists.mean()
        if dists.min() > thr:
            self.embeds.append(emb)
            if len(self.embeds) > self.buffer_size:
                self.embeds.pop(0)
            return True
        return False

def entropy_from_logits(logits: np.ndarray) -> float:
    exps = np.exp(logits - logits.max())
    p = exps / (exps.sum() + 1e-8)
    ent = - (p * np.log(p + 1e-8)).sum()
    return float(ent)

def select_topk_by_entropy(candidates: list[tuple[np.ndarray, np.ndarray]], top_p: float=0.05):
    # candidates: list of (emb, logits)
    if not candidates: return []
    ents = np.array([entropy_from_logits(lg) for _, lg in candidates])
    k = max(1, int(len(candidates) * top_p))
    idx = np.argsort(-ents)[:k]
    return [candidates[i] for i in idx]
