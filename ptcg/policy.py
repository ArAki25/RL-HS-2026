"""Policy network inference in numpy, and the agent that plays with it.

The network scores every candidate option of a decision (a softmax over them is the policy)
and estimates the win chance of the deciding player (value in [-1, 1]):

    card table   = MLP([static card features, card embedding])      one row per card ID
    slot         = MLP([card table[slot card], slot features])      one row per Pokémon slot
    state        = MLP([global, all slots, summed card lists, stadium/effect cards])
    option       = MLP([option features, card table[option card], slot[option slot], attack row])
    logit        = w · relu(Ws·state + Wo·option + b)
    value        = tanh(MLP(state))

ptcg/model.py implements the same network in torch for training and exports its weights to
an .npz file that NumpyPolicy loads. This module is bundled into the Kaggle submission, so it
must only depend on numpy.
"""

from __future__ import annotations

import numpy as np

from .features import Encoder, Tables

CARD_EMBED, CARD_DIM, ATTACK_EMBED = 16, 48, 16
SLOT_DIM, STATE_DIM, OPTION_DIM, HEAD_DIM = 64, 256, 128, 128
POOL_SCALE = 1 / 8  # summed card lists are scaled down to keep activations small


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


class NumpyPolicy:
    def __init__(self, weights: dict[str, np.ndarray], tables: Tables):
        self.w = {k: v.astype(np.float32) for k, v in weights.items()}
        card_in = np.concatenate([tables.card, self.w["card_embed"]], 1)
        self.card = self._mlp(card_in, "card")
        self.card[0] = 0
        self.attack = np.concatenate([tables.attack, self.w["attack_embed"]], 1)
        self.attack[0] = 0

    @classmethod
    def load(cls, path, tables: Tables) -> "NumpyPolicy":
        with np.load(path) as data:
            return cls(dict(data), tables)

    def _mlp(self, x: np.ndarray, name: str) -> np.ndarray:
        i = 0
        while f"{name}.{i}.w" in self.w:
            x = _relu(x @ self.w[f"{name}.{i}.w"] + self.w[f"{name}.{i}.b"])
            i += 1
        return x

    def __call__(self, x: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        """Return the option logits and the value of one encoded decision."""
        card, w = self.card, self.w
        slots = self._mlp(np.concatenate([card[x["slot_ids"]], x["slot_feats"]], 1), "slot")
        pooled = [card[x[k]].sum(0) * POOL_SCALE for k in ("hand_ids", "disc_ids", "opp_disc_ids", "unseen_ids")]
        state_in = np.concatenate([x["glob"], slots.ravel(), *pooled, card[x["misc_ids"]].ravel()])
        state = self._mlp(state_in, "state")

        slots = np.concatenate([slots, np.zeros((1, SLOT_DIM), np.float32)])
        option_in = np.concatenate(
            [x["opt_feats"], card[x["opt_card"]], slots[x["opt_slot"]], self.attack[x["opt_attack"]]], 1
        )
        option = self._mlp(option_in, "option")
        hidden = _relu(state @ w["pi.state"] + option @ w["pi.option"] + w["pi.b"])
        logits = hidden @ w["pi.out.w"] + w["pi.out.b"]
        value = np.tanh(_relu(state @ w["v.0.w"] + w["v.0.b"]) @ w["v.1.w"] + w["v.1.b"])
        return logits[:, 0], float(value[0])


class PolicyAgent:
    """Plays with a NumpyPolicy. Multi-select decisions are made one pick at a time.

    With ``greedy`` the best option is taken, otherwise one is sampled. If ``record`` is set,
    every non-forced pick is appended to ``self.steps`` as (input, choice, log prob, value).
    """

    def __init__(self, policy: NumpyPolicy, tables: Tables, deck: list[int] | None = None,
                 greedy: bool = True, record: bool = False, seed: int | None = None):
        self.policy = policy
        self.encoder = Encoder(tables, deck)
        self.greedy = greedy
        self.record = record
        self.rng = np.random.default_rng(seed)
        self.steps: list[tuple[dict, int, float, float]] = []

    def __call__(self, obs: dict) -> list[int]:
        select = obs["select"]
        n = len(select["option"])
        picked: list[int] = []
        while len(picked) < min(select["maxCount"], n):
            candidates = [i for i in range(n) if i not in picked]
            can_stop = len(picked) >= select["minCount"]
            if len(candidates) == 1 and not can_stop:
                picked.append(candidates[0])
                continue
            x = self.encoder.encode(obs, candidates, len(picked), can_stop)
            logits, value = self.policy(x)
            logp = logits.astype(np.float64) - logits.max()
            logp -= np.log(np.exp(logp).sum())
            choice = int(logp.argmax()) if self.greedy else int(self.rng.choice(len(logp), p=np.exp(logp)))
            if self.record:
                self.steps.append((x, choice, float(logp[choice]), value))
            if choice == len(candidates):  # STOP
                break
            picked.append(candidates[choice])
        return picked
