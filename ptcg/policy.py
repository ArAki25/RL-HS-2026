"""Numpy inference for the option-scoring network trained in ``ptcg.model``.

The network embeds the state and every legal option separately and scores each (state, option)
pair; the agent then picks the highest-scoring options. Weights are stored as an ``.npz`` with the
same parameter names as the PyTorch module, so the submission needs numpy only.
"""

from __future__ import annotations

import numpy as np

# Linear layer indices inside each nn.Sequential of ptcg.model.OptionScorer (ReLUs sit in between).
LAYERS = {"state_net": (0, 2), "option_net": (0, 2), "head": (0, 2)}


class NumpyPolicy:
    def __init__(self, weights: dict[str, np.ndarray]):
        self.w = {k: v.astype(np.float32) for k, v in weights.items()}
        self.state_dim = int(self.w["state_net.0.weight"].shape[1])
        self.option_dim = int(self.w["option_net.0.weight"].shape[1])

    @classmethod
    def load(cls, path) -> "NumpyPolicy":
        with np.load(path) as data:
            return cls(dict(data))

    def _mlp(self, name: str, x: np.ndarray, relu_last: bool) -> np.ndarray:
        layers = LAYERS[name]
        for i, layer in enumerate(layers):
            x = x @ self.w[f"{name}.{layer}.weight"].T + self.w[f"{name}.{layer}.bias"]
            if relu_last or i < len(layers) - 1:
                x = np.maximum(x, 0)
        return x

    def scores(self, state: np.ndarray, options: np.ndarray) -> np.ndarray:
        """One score (logit) per option row."""
        h = self._mlp("state_net", state, relu_last=True)
        g = self._mlp("option_net", options, relu_last=True)
        joint = np.concatenate([np.broadcast_to(h, (len(g), len(h))), g], axis=1)
        return self._mlp("head", joint, relu_last=False)[:, 0]
