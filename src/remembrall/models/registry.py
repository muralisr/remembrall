from __future__ import annotations
from .stubs import StubModel

def load_model(kind: str, **kwargs):
    # Hook point: map 'resnet18', 'mobilenetv2-lite', etc. to real models.
    # For now we return stubs with different dims to emulate size.
    presets = {
        "resnet8": dict(embedding_dim=64, num_classes=10),
        "resnet18": dict(embedding_dim=128, num_classes=10),
        "mobilenetv2-lite": dict(embedding_dim=96, num_classes=10),
        "yolov2-tiny": dict(embedding_dim=128, num_classes=20),
        "ssd-mobilenetv2": dict(embedding_dim=128, num_classes=20),
    }
    cfg = presets.get(kind.lower(), dict(embedding_dim=128, num_classes=10))
    cfg.update(kwargs)
    return StubModel(**cfg)
