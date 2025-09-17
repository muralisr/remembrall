from __future__ import annotations
import numpy as np

class StubModel:
    """A tiny stand-in for a NN model.
    Emits random embeddings and logits of configurable size.
    """
    def __init__(self, embedding_dim=128, num_classes=10, name="stub"):
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.name = name
        # "weights"
        self.weights = np.random.randn(embedding_dim, num_classes) * 0.01

    def forward(self, x) -> tuple[np.ndarray, np.ndarray]:
        # x: [B, ...]; we ignore content and emit random features
        B = len(x)
        emb = np.random.randn(B, self.embedding_dim)
        logits = emb @ self.weights
        return emb, logits

    def copy(self):
        m = StubModel(self.embedding_dim, self.num_classes, self.name + "_copy")
        m.weights = self.weights.copy()
        return m

    def load_state_dict(self, other_weights):
        self.weights = other_weights.copy()

    def state_dict(self):
        return self.weights.copy()
