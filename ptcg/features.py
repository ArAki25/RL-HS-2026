"""Turn observations into fixed-size vectors for learned agents.

``encode(obs)`` returns one state vector and one vector per legal option. Cards are described by
their properties from the card database (type, HP, attacks, ...) rather than by ID, so a model
trained on one deck still means something for others.

Only depends on numpy, so it is bundled into the submission.
"""

from __future__ import annotations

import numpy as np

from .enums import AreaType, OptionType, SelectContext, SelectType
from .observation import resolve_card, resolve_energy, resolve_target

N_CARD_TYPES = 7
N_POKEMON_TYPES = 5
N_EVOLUTION_TYPES = 4
N_ENERGY_TYPES = 12
N_AREAS = 13
BENCH_SLOTS = 5
STATUSES = ("poisoned", "burned", "asleep", "paralyzed", "confused")
FLAGS = ("basic", "stage1", "stage2", "ex", "megaEx", "tera", "aceSpec")
GAME_FLAGS = ("supporterPlayed", "stadiumPlayed", "energyAttached", "retreated")

HP_SCALE = 400.0
DAMAGE_SCALE = 350.0


def _one_hot(value, size: int) -> list[float]:
    vec = [0.0] * size
    if value is not None and 0 <= value < size:
        vec[value] = 1.0
    return vec


def affordable(cost: list[int], energies: list[int]) -> bool:
    """Whether attached energy types pay an attack cost (0 = colorless)."""
    pool = list(energies)
    for need in sorted(cost, reverse=True):
        if need == 0:
            if not pool:
                return False
            pool.pop()
        elif need in pool:
            pool.remove(need)
        else:
            return False
    return True


class Encoder:
    def __init__(self, cards: dict[int, dict], attacks: dict[int, dict]):
        self.cards = cards
        self.attacks = attacks
        self._card_cache: dict[int, list[float]] = {}
        self._no_card = [0.0] * len(self._card_features({}))
        self._no_pokemon = [0.0] * len(self._pokemon_vec({}))
        dummy_player = {"active": [], "bench": [], "discard": [], "prize": [], "deckCount": 0, "handCount": 0}
        dummy = {
            "current": {"yourIndex": 0, "turn": 0, "players": [dummy_player, dummy_player]},
            "select": {"type": 0, "context": 0, "minCount": 1, "maxCount": 1, "option": []},
        }
        self.state_dim = len(self.state_vec(dummy))
        self.option_dim = len(self.option_vec({"type": OptionType.END}, dummy))

    # --- building blocks -------------------------------------------------------------------

    def card_vec(self, card_id: int | None) -> list[float]:
        if card_id not in self._card_cache:
            c = self.cards.get(card_id) if card_id is not None else None
            self._card_cache[card_id] = self._card_features(c) if c else self._no_card
        return self._card_cache[card_id]

    def _card_features(self, c: dict) -> list[float]:
        attacks = [self.attacks[a] for a in c.get("attacks", []) if a in self.attacks]
        return (
            _one_hot(c.get("cardType"), N_CARD_TYPES)
            + _one_hot(c.get("pokemonType"), N_POKEMON_TYPES)
            + _one_hot(c.get("evolutionType"), N_EVOLUTION_TYPES)
            + _one_hot(c.get("energyType"), N_ENERGY_TYPES)
            + [c.get("hp", 0) / HP_SCALE, c.get("retreatCost", 0) / 4]
            + [float(bool(c.get(f))) for f in FLAGS]
            + [float(c.get("weakness") is not None), float(c.get("resistance") is not None)]
            + [
                len(attacks) / 3,
                max((a["damage"] for a in attacks), default=0) / DAMAGE_SCALE,
                len(c.get("skills", [])) / 2,
            ]
        )

    def pokemon_vec(self, p: dict | None) -> list[float]:
        return self._pokemon_vec(p) if p else self._no_pokemon

    def _pokemon_vec(self, p: dict) -> list[float]:
        present = bool(p)
        energies = p.get("energies", [])
        hp, max_hp = p.get("hp", 0), p.get("maxHp", 0)
        counts = [0.0] * 9
        for e in energies:
            if 0 <= e < 9:
                counts[e] += 0.25
        card = self.cards.get(p.get("id")) or {}
        attack_feats = []
        for attack_id in (card.get("attacks", []) + [None, None])[:2]:
            a = self.attacks.get(attack_id) if attack_id is not None else None
            if a is None:
                attack_feats += [0.0, 0.0]
            else:
                attack_feats += [a["damage"] / DAMAGE_SCALE, float(affordable(a["energies"], energies))]
        return (
            [float(present)]
            + self.card_vec(p.get("id") if present else None)
            + [hp / HP_SCALE, (max_hp - hp) / HP_SCALE, hp / max_hp if max_hp else 0.0]
            + [len(energies) / 6]
            + counts
            + [len(p.get("tools", [])), float(bool(p.get("appearThisTurn"))), len(p.get("preEvolution", [])) / 2]
            + attack_feats
        )

    def player_vec(self, player: dict) -> list[float]:
        vec = list(self.pokemon_vec((player["active"] or [None])[0]))
        bench = list(player["bench"]) + [None] * BENCH_SLOTS
        for p in bench[:BENCH_SLOTS]:
            vec += self.pokemon_vec(p)
        hand = player.get("hand") or []
        hand_sum = np.zeros(len(self._no_card))
        for card in hand:
            hand_sum += self.card_vec(card["id"])
        discard_types = [0.0] * N_CARD_TYPES
        for card in player["discard"]:
            t = (self.cards.get(card["id"]) or {}).get("cardType")
            if t is not None:
                discard_types[t] += 0.1
        prizes_left = sum(1 for _ in player["prize"])
        vec += [
            player.get("benchMax", 5) / 8,
            player["deckCount"] / 60,
            player["handCount"] / 20,
            len(player["discard"]) / 60,
            prizes_left / 6,
        ]
        vec += [float(bool(player.get(s))) for s in STATUSES]
        vec += list(hand_sum / 10)
        vec += discard_types
        return vec

    # --- public API ------------------------------------------------------------------------

    def state_vec(self, obs: dict) -> np.ndarray:
        current, select = obs["current"], obs["select"]
        me = current["yourIndex"]
        stadium = (current.get("stadium") or [None])[0]
        context_card = select.get("contextCard") or select.get("effect")
        vec = (
            [
                current["turn"] / 30,
                current.get("turnActionCount", 0) / 20,
                float(current.get("firstPlayer") == me),
            ]
            + [float(bool(current.get(f))) for f in GAME_FLAGS]
            + self.card_vec(stadium["id"] if stadium else None)
            + _one_hot(select["type"], len(SelectType))
            + _one_hot(select["context"], len(SelectContext))
            + [
                select["minCount"] / 5,
                select["maxCount"] / 5,
                select.get("remainDamageCounter", 0) / 10,
                select.get("remainEnergyCost", 0) / 5,
                len(select["option"]) / 40,
            ]
            + self.card_vec(context_card["id"] if context_card else None)
            + self.player_vec(current["players"][me])
            + self.player_vec(current["players"][1 - me])
        )
        return np.asarray(vec, dtype=np.float32)

    def option_vec(self, option: dict, obs: dict) -> np.ndarray:
        kind = option["type"]
        me = obs["current"]["yourIndex"]
        owner = option.get("playerIndex")
        card_id = None
        pokemon = None
        if kind == OptionType.SKILL:
            card_id = option.get("cardId")
        elif kind == OptionType.ENERGY:
            energy = resolve_energy(option, obs)
            card_id = energy["id"] if energy else None
            pokemon = resolve_card(option, obs)
        else:
            card = resolve_card(option, obs)
            if card is not None:
                card_id = card.get("id")
                if option.get("area") in (AreaType.ACTIVE, AreaType.BENCH):
                    pokemon = card
        if kind in (OptionType.ATTACH, OptionType.EVOLVE):
            pokemon = resolve_target(option, obs)
        attack = self.attacks.get(option.get("attackId")) if kind == OptionType.ATTACK else None
        vec = (
            _one_hot(kind, len(OptionType))
            + _one_hot(option.get("area"), N_AREAS)
            + _one_hot(option.get("inPlayArea"), N_AREAS)
            + [
                option.get("index", 0) / 10,
                option.get("inPlayIndex", 0) / 5,
                float(owner == me),
                float(owner is not None and owner != me),
                option.get("number", 0) / 10,
                option.get("count", 0) / 5,
            ]
            + self.card_vec(card_id)
            + self.pokemon_vec(pokemon)
            + [
                attack["damage"] / DAMAGE_SCALE if attack else 0.0,
                len(attack["energies"]) / 5 if attack else 0.0,
            ]
        )
        return np.asarray(vec, dtype=np.float32)

    def encode(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return (state vector, option matrix with one row per legal option)."""
        state = self.state_vec(obs)
        options = np.stack([self.option_vec(o, obs) for o in obs["select"]["option"]])
        return state, options
