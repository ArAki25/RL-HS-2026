"""Torch version of the policy network (see ptcg/policy.py) and a batched sample buffer."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .features import (
    ATTACK_FEATURES, CARD_FEATURES, GLOBAL_FEATURES, NO_SLOT, OPTION_FEATURES, SLOT_FEATURES, SLOTS, Tables,
)
from .policy import (
    ATTACK_EMBED, CARD_DIM, CARD_EMBED, HEAD_DIM, OPTION_DIM, POOL_SCALE, SLOT_DIM, STATE_DIM, NumpyPolicy,
)

POOLED = ("hand_ids", "disc_ids", "opp_disc_ids", "unseen_ids")
FIXED = ("glob", "slot_ids", "slot_feats", *POOLED, "misc_ids")
OPTION = ("opt_feats", "opt_card", "opt_slot", "opt_attack")
MASKED = -1e9


class MLP(nn.Module):
    def __init__(self, *dims: int):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.relu(layer(x))
        return x


class PolicyNet(nn.Module):
    def __init__(self, tables: Tables):
        super().__init__()
        self.register_buffer("card_static", torch.from_numpy(tables.card))
        self.register_buffer("attack_static", torch.from_numpy(tables.attack))
        self.card_embed = nn.Parameter(torch.randn(len(tables.card), CARD_EMBED) * 0.02)
        self.attack_embed = nn.Parameter(torch.randn(len(tables.attack), ATTACK_EMBED) * 0.02)
        self.card = MLP(CARD_FEATURES + CARD_EMBED, 64, CARD_DIM)
        self.slot = MLP(CARD_DIM + SLOT_FEATURES, SLOT_DIM, SLOT_DIM)
        state_in = GLOBAL_FEATURES + SLOTS * SLOT_DIM + (len(POOLED) + 3) * CARD_DIM
        self.state = MLP(state_in, STATE_DIM, STATE_DIM)
        option_in = OPTION_FEATURES + CARD_DIM + SLOT_DIM + ATTACK_FEATURES + ATTACK_EMBED
        self.option = MLP(option_in, OPTION_DIM, OPTION_DIM)
        self.pi_state = nn.Linear(STATE_DIM, HEAD_DIM)
        self.pi_option = nn.Linear(OPTION_DIM, HEAD_DIM, bias=False)
        self.pi_out = nn.Linear(HEAD_DIM, 1)
        self.v0 = nn.Linear(STATE_DIM, HEAD_DIM)
        self.v1 = nn.Linear(HEAD_DIM, 1)

    def forward(self, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked option logits (B, K) and values (B,)."""
        card = self.card(torch.cat([self.card_static, self.card_embed], 1))
        card = card * (torch.arange(len(card), device=card.device) > 0)[:, None]
        attack = torch.cat([self.attack_static, self.attack_embed], 1)
        attack = attack * (torch.arange(len(attack), device=attack.device) > 0)[:, None]

        # Lookups use F.embedding instead of indexing: its backward pass is much faster on the CPU.
        lookup = F.embedding
        slots = self.slot(torch.cat([lookup(b["slot_ids"], card), b["slot_feats"]], -1))
        pooled = [lookup(b[k], card).sum(1) * POOL_SCALE for k in POOLED]
        misc = lookup(b["misc_ids"], card).flatten(1)
        state = self.state(torch.cat([b["glob"], slots.flatten(1), *pooled, misc], -1))

        n = len(slots)
        slots = torch.cat([slots, slots.new_zeros(n, 1, SLOT_DIM)], 1)
        option_slot = lookup(torch.arange(n)[:, None] * slots.shape[1] + b["opt_slot"], slots.flatten(0, 1))
        option = self.option(torch.cat(
            [b["opt_feats"], lookup(b["opt_card"], card), option_slot, lookup(b["opt_attack"], attack)], -1
        ))
        hidden = F.relu(self.pi_state(state)[:, None, :] + self.pi_option(option))
        logits = self.pi_out(hidden).squeeze(-1).masked_fill(~b["mask"], MASKED)
        value = torch.tanh(self.v1(F.relu(self.v0(state)))).squeeze(-1)
        return logits, value

    def numpy_weights(self) -> dict[str, np.ndarray]:
        """Weights in the layout NumpyPolicy expects (linear weights as (in, out) matrices)."""
        w = {"card_embed": self.card_embed, "attack_embed": self.attack_embed}
        for name in ("card", "slot", "state", "option"):
            for i, layer in enumerate(getattr(self, name).layers):
                w[f"{name}.{i}.w"], w[f"{name}.{i}.b"] = layer.weight.T, layer.bias
        w["pi.state"], w["pi.b"] = self.pi_state.weight.T, self.pi_state.bias
        w["pi.option"] = self.pi_option.weight.T
        w["pi.out.w"], w["pi.out.b"] = self.pi_out.weight.T, self.pi_out.bias
        w["v.0.w"], w["v.0.b"], w["v.1.w"], w["v.1.b"] = self.v0.weight.T, self.v0.bias, self.v1.weight.T, self.v1.bias
        return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in w.items()}

    def export(self, path, tables: Tables, check: list[dict] | None = None) -> None:
        """Save numpy weights; if samples are given, verify NumpyPolicy matches this network on them."""
        weights = self.numpy_weights()
        np.savez_compressed(path, **weights)
        if not check:
            return
        policy = NumpyPolicy(weights, tables)
        buffer = Buffer(check)
        with torch.no_grad():
            logits, values = self(buffer.batch(np.arange(len(check))))
        for i, x in enumerate(check):
            x = {k: v.astype(np.float32) if v.dtype == np.float16 else v for k, v in x.items()}
            np_logits, np_value = policy(x)
            same_logits = np.allclose(np_logits, logits[i, : len(np_logits)].numpy(), atol=1e-3)
            if not same_logits or abs(np_value - values[i].item()) > 1e-3:
                raise RuntimeError("numpy and torch policies disagree; the export is broken")


class Buffer:
    """Stores encoded decisions (with a varying number of options) and builds padded batches."""

    def __init__(self, inputs: list[dict[str, np.ndarray]], **extra: np.ndarray):
        self.n = len(inputs)
        self.fixed = {k: torch.from_numpy(np.stack([x[k] for x in inputs])) for k in FIXED}
        self.counts = np.array([len(x["opt_card"]) for x in inputs])
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)[:-1]])
        pad = {"opt_feats": np.zeros((1, OPTION_FEATURES), inputs[0]["opt_feats"].dtype),
               "opt_card": np.zeros(1, np.int16), "opt_slot": np.full(1, NO_SLOT, np.int16),
               "opt_attack": np.zeros(1, np.int16)}
        self.options = {k: torch.from_numpy(np.concatenate([x[k] for x in inputs] + [pad[k]])) for k in OPTION}
        self.pad_row = int(self.counts.sum())
        self.extra = {k: torch.as_tensor(v) for k, v in extra.items()}

    def __len__(self) -> int:
        return self.n

    def batch(self, idx: np.ndarray) -> dict[str, torch.Tensor]:
        counts = self.counts[idx]
        width = np.arange(counts.max())
        mask = width[None, :] < counts[:, None]
        rows = torch.from_numpy(np.where(mask, self.offsets[idx][:, None] + width[None, :], self.pad_row))
        t = torch.from_numpy(idx)
        b = {k: v[t] for k, v in self.fixed.items()}
        b.update({k: v[rows] for k, v in self.options.items()})
        for k in b:
            b[k] = b[k].float() if b[k].is_floating_point() else b[k].long()
        b["mask"] = torch.from_numpy(mask)
        b.update({k: v[t] for k, v in self.extra.items()})
        return b
