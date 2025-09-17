from __future__ import annotations
import numpy as np

def cosine_sim(a, b):
    an = np.linalg.norm(a) + 1e-8
    bn = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (an * bn))

def reptile_update(base_model, specialized_model, scene_sim: float):
    """Dynamic-EWMA Reptile update.
    scene_sim in [0,1]; higher => more similar => larger step.
    """
    eps = 0.3 if scene_sim >= 0.9 else 0.05
    wb = base_model.state_dict()
    ws = specialized_model.state_dict()
    base_model.load_state_dict((1 - eps) * wb + eps * ws)
    return eps
