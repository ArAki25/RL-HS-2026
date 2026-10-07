"""Games played in worker processes for training: heuristic demonstrations and policy rollouts.

Workers only need numpy (the policy runs as NumpyPolicy), and every worker process holds its
own engine, since the engine can run only one game per process.
"""

from __future__ import annotations

import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import engine
from .agents import HeuristicAgent, RandomAgent
from .features import Encoder, Tables, compact
from .policy import NumpyPolicy, PolicyAgent

_context: tuple | None = None


def _setup() -> tuple[Tables, HeuristicAgent]:
    global _context
    if _context is None:
        cards, attacks = engine.card_db(), engine.attack_db()
        _context = (Tables(cards, attacks), HeuristicAgent(cards, attacks))
    return _context


def make_pool(workers: int) -> ProcessPoolExecutor:
    # One BLAS thread per worker: the matrices are tiny and the workers already use every core.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    return ProcessPoolExecutor(workers)


class DemoRecorder:
    """Plays like the heuristic and records its picks in the same single-pick form as PolicyAgent."""

    def __init__(self, heuristic: HeuristicAgent, tables: Tables, deck: list[int]):
        self.heuristic = heuristic
        self.encoder = Encoder(tables, deck)
        self.steps: list[tuple[dict, int]] = []

    def __call__(self, obs: dict) -> list[int]:
        chosen = self.heuristic(obs)
        select = obs["select"]
        n = len(select["option"])
        picked: list[int] = []
        for target in [*chosen, None]:  # None = STOP after the heuristic's picks
            if len(picked) >= min(select["maxCount"], n):
                break
            candidates = [i for i in range(n) if i not in picked]
            can_stop = len(picked) >= select["minCount"]
            if target is None and not can_stop:
                break
            if len(candidates) == 1 and not can_stop:
                picked.append(candidates[0])
                continue
            x = self.encoder.encode(obs, candidates, len(picked), can_stop)
            self.steps.append((compact(x), len(candidates) if target is None else candidates.index(target)))
            if target is None:
                break
            picked.append(target)
        return chosen


def demo_games(games: int, deck: list[int], decks: list[list[int]], random_share: float, seed: int):
    """Heuristic games for behavior cloning. Returns encoded decisions, picks and game outcomes."""
    tables, heuristic = _setup()
    rng = random.Random(seed)
    random.seed(seed)  # the random agent uses the global generator
    inputs, labels, outcomes = [], [], []
    for g in range(games):
        decks_ = (deck if rng.random() < 0.5 else rng.choice(decks), rng.choice(decks))
        players = [DemoRecorder(heuristic, tables, decks_[0]), DemoRecorder(heuristic, tables, decks_[1])]
        agents = list(players)
        if rng.random() < random_share:
            agents[1] = RandomAgent()
        if g % 2:
            agents.reverse()
            decks_ = decks_[::-1]
            players.reverse()
        result = engine.play_game(tuple(agents), decks_)
        for seat, (agent, player) in enumerate(zip(agents, players)):
            if agent is not player:
                continue
            outcome = 0.0 if result == 2 else (1.0 if result == seat else -1.0)
            for x, label in player.steps:
                inputs.append(x)
                labels.append(label)
                outcomes.append(outcome)
    return inputs, labels, outcomes


def policy_games(weights: dict, opponent: str, opponent_weights: dict | None, decks: list[list[int]],
                 opponent_decks: list[list[int]], games: int, seed: int, greedy: bool = False, record: bool = True):
    """Play the policy against an opponent ("self", "policy", "heuristic" or "random").

    Returns the recorded trajectories as (steps, reward) and the learner's rewards (+1/0/-1).
    In "self" games both seats play the current policy and both trajectories are returned.
    """
    tables, heuristic = _setup()
    rng = random.Random(seed)
    random.seed(seed)
    policy = NumpyPolicy(weights, tables)
    other = NumpyPolicy(opponent_weights, tables) if opponent == "policy" else None
    trajectories, rewards = [], []
    for g in range(games):
        deck, opp_deck = rng.choice(decks), rng.choice(opponent_decks)
        learner = PolicyAgent(policy, tables, deck, greedy=greedy, record=record, seed=rng.getrandbits(32))
        if opponent == "self":
            opp = PolicyAgent(policy, tables, opp_deck, greedy=greedy, record=record, seed=rng.getrandbits(32))
        elif opponent == "policy":
            opp = PolicyAgent(other, tables, opp_deck, greedy=greedy, seed=rng.getrandbits(32))
        elif opponent == "heuristic":
            opp = heuristic
        else:
            opp = RandomAgent()
        seat = (seed + g) % 2
        agents, decks_ = ((learner, opp), (deck, opp_deck)) if seat == 0 else ((opp, learner), (opp_deck, deck))
        result = engine.play_game(agents, decks_)
        reward = 0 if result == 2 else (1 if result == seat else -1)
        rewards.append(reward)
        if record:
            trajectories.append(([(compact(x), *rest) for x, *rest in learner.steps], reward))
            if opponent == "self":
                trajectories.append(([(compact(x), *rest) for x, *rest in opp.steps], -reward))
    return trajectories, rewards


def load_weights(path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return dict(data)
