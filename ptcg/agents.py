"""Agents: callables mapping an observation to a list of option indices.

This module is bundled into the Kaggle submission, so it must not import the engine.
"""

from __future__ import annotations

import os
import random

from .enums import AreaType, CardType, OptionType, SelectContext
from .observation import resolve_card

# Contexts where choosing as few options as possible is best (cards leave our control).
FEWER_IS_BETTER = {
    SelectContext.DISCARD,
    SelectContext.TO_DECK,
    SelectContext.TO_DECK_BOTTOM,
    SelectContext.DISCARD_ENERGY_CARD,
    SelectContext.DISCARD_TOOL_CARD,
    SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
    SelectContext.DISCARD_ENERGY,
    SelectContext.TO_DECK_ENERGY,
}

CARD_TYPE_VALUE = {
    CardType.POKEMON: 3.0,
    CardType.SUPPORTER: 2.5,
    CardType.ITEM: 2.0,
    CardType.TOOL: 1.5,
    CardType.STADIUM: 1.0,
    CardType.BASIC_ENERGY: 1.0,
    CardType.SPECIAL_ENERGY: 1.2,
}

MAIN_PLAY_SCORE = {
    CardType.POKEMON: 90,
    CardType.SUPPORTER: 70,
    CardType.ITEM: 60,
    CardType.TOOL: 55,
    CardType.STADIUM: 40,
}


def pick_best(scores, select: dict) -> list[int]:
    """Indices of the highest-scoring options; as few as allowed where cards leave our control."""
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    fewer = select["context"] in FEWER_IS_BETTER
    count = select["minCount"] if fewer else select["maxCount"]
    return order[: max(select["minCount"], min(count, len(scores)))]


class RandomAgent:
    """Uniformly random legal move (same as the built-in Kaggle "random" agent)."""

    def __call__(self, obs: dict) -> list[int]:
        select = obs["select"]
        n = len(select["option"])
        return random.sample(range(n), min(select["maxCount"], n))


class HeuristicAgent:
    """Hand-written baseline: build up the board first, then use the strongest attack."""

    def __init__(self, cards: dict[int, dict], attacks: dict[int, dict]):
        self.cards = cards
        self.attacks = attacks

    def __call__(self, obs: dict) -> list[int]:
        select = obs["select"]
        options = select["option"]
        return pick_best([self._score(option, obs) for option in options], select)

    def _score(self, option: dict, obs: dict) -> float:
        kind = option["type"]
        me = obs["current"]["yourIndex"]
        if kind == OptionType.EVOLVE:
            return 100
        if kind == OptionType.PLAY:
            card = self.cards[obs["current"]["players"][me]["hand"][option["index"]]["id"]]
            return MAIN_PLAY_SCORE.get(card["cardType"], 50)
        if kind == OptionType.ATTACH:
            return 85 if option.get("inPlayArea") == AreaType.ACTIVE else 80
        if kind == OptionType.ABILITY:
            return 65
        if kind == OptionType.ATTACK:
            return 20 + self.attacks[option["attackId"]]["damage"] / 10
        if kind == OptionType.END:
            return 0
        if kind in (OptionType.RETREAT, OptionType.DISCARD):
            return -10
        if kind == OptionType.YES:
            return 1
        if kind == OptionType.NUMBER:
            return option["number"]
        if kind in (OptionType.CARD, OptionType.TOOL_CARD, OptionType.ENERGY_CARD):
            return self._card_score(option, obs)
        return 0

    def _card_score(self, option: dict, obs: dict) -> float:
        current = obs["current"]
        me = current["yourIndex"]
        owner = option.get("playerIndex", me)
        card = resolve_card(option, obs)
        if card is None:
            return 0
        if option.get("area") in (AreaType.ACTIVE, AreaType.BENCH) and "hp" in card:
            # Own Pokémon in play: prefer the healthiest. Opponent's: target the weakest.
            return card["hp"] if owner == me else -card["hp"]
        data = self.cards.get(card["id"])
        if data is None:
            return 0
        value = CARD_TYPE_VALUE.get(data["cardType"], 1.0) + data["hp"] / 1000
        if data["cardType"] == CardType.POKEMON and not data["basic"]:
            value += 0.5
        if obs["select"]["context"] in FEWER_IS_BETTER:
            value = -value
        return value if owner == me else -value


AGENTS = ["random", "heuristic", "rl"]
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "ppo", "model.npz")


def agent_name(name: str) -> str:
    """argparse type for agent names: one of AGENTS, or "rl:<path to .npz>" for a specific model."""
    if name in AGENTS or name.startswith("rl:"):
        return name
    raise ValueError(f"unknown agent {name!r}, choose from {AGENTS} or rl:<path>")


def make_agent(name: str, cards: dict[int, dict], attacks: dict[int, dict], model: str | None = None,
               deck: list[int] | None = None):
    """Create an agent. "rl" plays with the trained weights (``model``, an .npz file, or the path in
    "rl:<path>") and needs its own deck: the network is told which of its cards it has not seen yet."""
    if name == "random":
        return RandomAgent()
    if name == "heuristic":
        return HeuristicAgent(cards, attacks)
    if name == "rl" or name.startswith("rl:"):
        # Imported here so that the baseline agents work without numpy.
        from .features import Tables
        from .policy import NumpyPolicy, PolicyAgent

        path = name.partition(":")[2] or model or DEFAULT_MODEL
        tables = Tables(cards, attacks)
        return PolicyAgent(NumpyPolicy.load(path, tables), tables, deck)
    raise ValueError(f"unknown agent {name!r}, choose from {AGENTS} or rl:<path>")
