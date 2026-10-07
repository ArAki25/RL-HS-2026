"""Direct access to the cabt battle engine that ships with kaggle-environments.

Playing through the engine directly is much faster than ``kaggle_environments.make("cabt")``
and is what evaluation and training use. The engine keeps one battle in global state,
so only one game can run per process; parallelism has to come from multiple processes.
"""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import sys
from functools import lru_cache
from typing import Callable

_spec = importlib.util.find_spec("kaggle_environments")
if _spec is None or _spec.origin is None:
    raise ImportError("kaggle-environments is not installed (see README)")
CABT_DIR = os.path.join(os.path.dirname(_spec.origin), "envs", "cabt")
# Import the engine bindings without importing kaggle_environments (which loads every env).
sys.path.insert(0, CABT_DIR)

from cg import game  # noqa: E402
from cg.sim import lib  # noqa: E402

lib.AllCard.restype = ctypes.c_char_p
lib.AllAttack.restype = ctypes.c_char_p

Agent = Callable[[dict], list[int]]

# Safety net against games that never end; such games count as a draw.
MAX_DECISIONS = 5000


@lru_cache(maxsize=None)
def card_db() -> dict[int, dict]:
    return {c["cardId"]: c for c in json.loads(lib.AllCard().decode("utf-8"))}


@lru_cache(maxsize=None)
def attack_db() -> dict[int, dict]:
    return {a["attackId"]: a for a in json.loads(lib.AllAttack().decode("utf-8"))}


def load_deck(path: str | os.PathLike) -> list[int]:
    """Read a deck file with one card ID per line."""
    with open(path) as f:
        deck = [int(line) for line in f if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"{path}: a deck needs 60 cards, got {len(deck)}")
    unknown = sorted(set(deck) - card_db().keys())
    if unknown:
        raise ValueError(f"{path}: unknown card IDs {unknown}")
    return deck


def play_game(agents: tuple[Agent, Agent], decks: tuple[list[int], list[int]]) -> int:
    """Play one game and return the winning seat (0 or 1), or 2 for a draw."""
    obs, start = game.battle_start(decks[0], decks[1])
    if obs is None:
        raise ValueError(f"deck of player {start.errorPlayer} rejected (error type {start.errorType})")
    try:
        for _ in range(MAX_DECISIONS):
            current = obs["current"]
            if current["result"] >= 0:
                return current["result"]
            obs = game.battle_select(agents[current["yourIndex"]](obs))
        return 2
    finally:
        game.battle_finish()
