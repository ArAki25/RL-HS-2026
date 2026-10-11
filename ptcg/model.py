"""PyTorch version of the option-scoring network (training only; inference uses ptcg.policy)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class OptionScorer(nn.Module):
    def __init__(self, state_dim: int, option_dim: int, hidden: int = 256, embed: int = 128):
        super().__init__()
        self.state_net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, embed), nn.ReLU())
        self.option_net = nn.Sequential(nn.Linear(option_dim, embed), nn.ReLU(), nn.Linear(embed, embed), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * embed, embed), nn.ReLU(), nn.Linear(embed, 1))

    def forward(self, states: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        """states (B, S), options (B, K, O) -> scores (B, K)."""
        h = self.state_net(states).unsqueeze(1).expand(-1, options.shape[1], -1)
        g = self.option_net(options)
        return self.head(torch.cat([h, g], dim=-1)).squeeze(-1)

    def export_numpy(self, path: str | Path) -> None:
        """Save weights as .npz for ptcg.policy.NumpyPolicy."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{k: v.detach().cpu().numpy() for k, v in self.state_dict().items()})
