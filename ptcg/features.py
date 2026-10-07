"""Turn an observation into fixed-size numpy arrays for the policy network.

Everything is seen from the deciding player ("me"). A decision is encoded as
  - global features (game state, selection context),
  - one feature row per Pokémon slot (my active + bench, then the opponent's),
  - card ID lists that the network embeds and sums (hand, discard piles, unseen own cards),
  - one feature row per candidate option, plus references from each option to a card ID,
    a Pokémon slot and an attack ID.
Multi-select decisions are split into single picks; a virtual STOP option ends the selection.

This module is bundled into the Kaggle submission, so it must not import the engine.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .enums import AreaType, CardType, OptionType

N_ENERGY = 12  # Colorless, Grass, Fire, Water, Lightning, Psychic, Fighting, Darkness, Metal, Dragon, Rainbow, Rocket
RAINBOW = 10
BENCH = 8  # benchMax is 5, or 8 with some stadiums
SLOTS = 2 * (1 + BENCH)
NO_SLOT = SLOTS  # index of the all-zero slot row
HAND, DISCARD, UNSEEN = 24, 64, 64
STOP = len(OptionType)  # option type of the virtual STOP option
N_OPTION_TYPES = STOP + 1
N_AREAS = 13
N_SELECT_TYPES = 11
N_CONTEXTS = 49
CONDITIONS = ("poisoned", "burned", "asleep", "paralyzed", "confused")

CARD_FEATURES = 7 + N_ENERGY + 1 + 7 + 1 + 2 * N_ENERGY + 2 + 6 + 4 + 5
ATTACK_FEATURES = 4 + N_ENERGY
SLOT_FEATURES = 6 + N_ENERGY + 5 + len(CONDITIONS) + 3
GLOBAL_FEATURES = 6 + 2 * (6 + len(CONDITIONS)) + 4 + N_SELECT_TYPES + N_CONTEXTS + 7
OPTION_FEATURES = N_OPTION_TYPES + N_AREAS + 4 + 2 + N_ENERGY + 3


def _one_hot(n: int, i: int | None) -> list[float]:
    v = [0.0] * n
    if i is not None and 0 <= i < n:
        v[i] = 1.0
    return v


def can_pay(cost: list[int], energies: list[int]) -> bool:
    """Whether attached energy types cover an attack cost (0 = any type, Rainbow = any type)."""
    pool = Counter(energies)
    for t in cost:
        if t == 0:
            continue
        if pool[t] > 0:
            pool[t] -= 1
        elif pool[RAINBOW] > 0:
            pool[RAINBOW] -= 1
        else:
            return False
    return sum(pool.values()) >= sum(1 for t in cost if t == 0)


class Tables:
    """Static per-card and per-attack feature tables, indexed by card ID / attack ID (row 0 = none)."""

    def __init__(self, cards: dict[int, dict], attacks: dict[int, dict]):
        self.cards = cards
        self.attacks = attacks
        self.card = np.zeros((max(cards) + 1, CARD_FEATURES), np.float32)
        for cid, c in cards.items():
            self.card[cid] = self._card_row(c)
        self.attack = np.zeros((max(attacks) + 1, ATTACK_FEATURES), np.float32)
        for aid, a in attacks.items():
            cost = a["energies"]
            self.attack[aid] = [
                a["damage"] / 100, len(cost) / 4, cost.count(0) / 4, float(bool(a["text"])),
                *[cost.count(t) / 4 for t in range(N_ENERGY)],
            ]

    def _card_row(self, c: dict) -> list[float]:
        atk = [self.attacks[a] for a in c["attacks"][:2]]
        atk_feats = []
        for i in range(2):
            if i < len(atk):
                atk_feats += [atk[i]["damage"] / 100, len(atk[i]["energies"]) / 4, float(bool(atk[i]["text"]))]
            else:
                atk_feats += [0.0, 0.0, 0.0]
        return [
            *_one_hot(7, c["cardType"]),
            *_one_hot(N_ENERGY, c["energyType"]),
            c["hp"] / 100,
            *[float(c[k]) for k in ("basic", "stage1", "stage2", "ex", "megaEx", "tera", "aceSpec")],
            c["retreatCost"] / 4,
            *_one_hot(N_ENERGY, c["weakness"]),
            *_one_hot(N_ENERGY, c["resistance"]),
            len(c["attacks"]) / 2,
            len(c["skills"]) / 2,
            *atk_feats,
            *_one_hot(4, c["evolutionType"]),
            *_one_hot(5, c["pokemonType"]),
        ]


def _id(card) -> int:
    if card is None:
        return 0
    return card["id"] if isinstance(card, dict) else int(card)


def _ids(cards, n: int) -> np.ndarray:
    out = np.zeros(n, np.int16)
    ids = [_id(c) for c in cards or []][:n]
    out[: len(ids)] = ids
    return out


class Encoder:
    """Builds the network input for one (sub-)decision."""

    def __init__(self, tables: Tables, deck: list[int] | None = None):
        self.t = tables
        self.deck = Counter(deck or [])

    def _slot_row(self, poke: dict, active: bool, player: dict) -> list[float]:
        card = self.t.cards.get(poke["id"])
        energies = poke.get("energies") or []
        max_hp = poke.get("maxHp") or 1
        costs = [self.t.attacks[a]["energies"] for a in (card["attacks"][:2] if card else [])]
        payable = [float(can_pay(c, energies)) for c in costs] + [0.0] * (2 - len(costs))
        retreat = card["retreatCost"] if card else 0
        return [
            1.0, float(active), poke["hp"] / 100, max_hp / 100, (max_hp - poke["hp"]) / 100, poke["hp"] / max_hp,
            *[energies.count(t) / 3 for t in range(N_ENERGY)],
            len(energies) / 5, len(poke.get("energyCards") or []) / 5, len(poke.get("tools") or []),
            float(bool(poke.get("appearThisTurn"))), len(poke.get("preEvolution") or []) / 2,
            *[float(active and bool(player.get(k))) for k in CONDITIONS],
            *payable, float(len(energies) >= retreat),
        ]

    def _unseen(self, me: dict, my_index: int, stadium: list) -> list[int]:
        """Own cards that are neither visible nor known to be gone: deck + prizes."""
        if not self.deck:
            return []
        seen = Counter(_id(c) for c in (me.get("hand") or []) + (me.get("discard") or []))
        for poke in (me.get("active") or []) + (me.get("bench") or []):
            if poke is None:
                continue
            seen[poke["id"]] += 1
            for key in ("energyCards", "tools", "preEvolution"):
                seen.update(_id(c) for c in poke.get(key) or [])
        seen.update(_id(c) for c in stadium if c.get("playerIndex") == my_index)
        return sorted((self.deck - seen).elements())

    def encode(self, obs: dict, candidates: list[int], picked: int, can_stop: bool) -> dict[str, np.ndarray]:
        cur, select = obs["current"], obs["select"]
        me = cur["yourIndex"]
        players = [cur["players"][me], cur["players"][1 - me]]
        stadium = cur.get("stadium") or []

        slot_ids = np.zeros(SLOTS, np.int16)
        slot_feats = np.zeros((SLOTS, SLOT_FEATURES), np.float32)
        for side, p in enumerate(players):
            in_play = {1 + i: poke for i, poke in enumerate((p.get("bench") or [])[:BENCH])}
            if p.get("active"):
                in_play[0] = p["active"][0]
            for pos, poke in in_play.items():
                if poke is None:
                    continue
                s = side * (1 + BENCH) + pos
                slot_ids[s] = poke["id"]
                slot_feats[s] = self._slot_row(poke, pos == 0, p)

        glob = [
            min(cur.get("turn", 0), 40) / 20, cur.get("turn", 0) % 2, float(cur.get("firstPlayer") == me),
            min(cur.get("turnActionCount", 0), 30) / 10,
            float(bool(stadium) and stadium[0].get("playerIndex") == me),
            float(bool(stadium) and stadium[0].get("playerIndex") != me),
        ]
        for p in players:
            glob += [
                len(p.get("prize") or []) / 6, p.get("deckCount", 0) / 60, p.get("handCount", 0) / 10,
                len(p.get("discard") or []) / 20, len(p.get("bench") or []) / 5, float(bool(p.get("active"))),
                *[float(bool(p.get(k))) for k in CONDITIONS],
            ]
        glob += [float(bool(cur.get(k))) for k in ("supporterPlayed", "stadiumPlayed", "energyAttached", "retreated")]
        glob += _one_hot(N_SELECT_TYPES, select["type"]) + _one_hot(N_CONTEXTS, select["context"])
        glob += [
            select["minCount"] / 3, select["maxCount"] / 3, picked / 3, float(can_stop),
            (select.get("remainDamageCounter") or 0) / 10, (select.get("remainEnergyCost") or 0) / 5,
            len(candidates) / 20,
        ]

        effect, context_card = select.get("effect"), select.get("contextCard")
        misc_ids = np.array([_id(stadium[0]) if stadium else 0, _id(effect), _id(context_card)], np.int16)

        rows, opt_card, opt_slot, opt_attack = [], [], [], []
        for i in candidates:
            row, card, slot, attack = self._option(select["option"][i], obs, players, me)
            rows.append(row)
            opt_card.append(card)
            opt_slot.append(slot)
            opt_attack.append(attack)
        if can_stop:
            rows.append(_one_hot(N_OPTION_TYPES, STOP) + [0.0] * (OPTION_FEATURES - N_OPTION_TYPES))
            opt_card.append(0)
            opt_slot.append(NO_SLOT)
            opt_attack.append(0)

        return {
            "glob": np.array(glob, np.float32),
            "slot_ids": slot_ids,
            "slot_feats": slot_feats,
            "hand_ids": _ids(players[0].get("hand"), HAND),
            "disc_ids": _ids(players[0].get("discard"), DISCARD),
            "opp_disc_ids": _ids(players[1].get("discard"), DISCARD),
            "unseen_ids": _ids(self._unseen(players[0], me, stadium), UNSEEN),
            "misc_ids": misc_ids,
            "opt_feats": np.array(rows, np.float32).reshape(-1, OPTION_FEATURES),
            "opt_card": np.array(opt_card, np.int16),
            "opt_slot": np.array(opt_slot, np.int16),
            "opt_attack": np.array(opt_attack, np.int16),
        }

    def _option(self, option: dict, obs: dict, players: list[dict], me: int) -> tuple[list[float], int, int, int]:
        kind = option["type"]
        area = option.get("area")
        owner = option.get("playerIndex", me)
        side = 0 if owner == me else 1
        card, slot, attack, energy = 0, NO_SLOT, 0, None

        if kind in (OptionType.ATTACH, OptionType.EVOLVE):
            card = _id(_zone_card(players[0], obs, AreaType.HAND, option.get("index")))
            slot = _slot(0, option.get("inPlayArea"), option.get("inPlayIndex"))
        elif kind == OptionType.PLAY:
            card = _id(_zone_card(players[0], obs, AreaType.HAND, option.get("index")))
        elif kind == OptionType.ATTACK:
            attack = option.get("attackId") or 0
        elif kind == OptionType.SKILL:
            card = option.get("cardId") or 0
        elif kind == OptionType.ENERGY:
            slot = _slot(side, area, option.get("index"))
            poke = _zone_card(players[side], obs, area, option.get("index"))
            energies = (poke or {}).get("energies") or []
            idx = option.get("energyIndex")
            energy = energies[idx] if idx is not None and idx < len(energies) else None
        elif area is not None:
            slot = _slot(side, area, option.get("index"))
            card = _id(_zone_card(players[side], obs, area, option.get("index")))

        cdata = self.t.cards.get(card)
        row = [
            *_one_hot(N_OPTION_TYPES, kind),
            *_one_hot(N_AREAS, area),
            float(option.get("inPlayArea") == AreaType.ACTIVE), float(option.get("inPlayArea") == AreaType.BENCH),
            float(side == 0 and area is not None), float(side == 1),
            (option.get("number") or 0) / 10, (option.get("count") or 0) / 4,
            *_one_hot(N_ENERGY, energy),
            float(cdata is not None and cdata["cardType"] == CardType.POKEMON),
            float(cdata is not None and cdata["cardType"] in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)),
            float(attack > 0 and attack in self.t.attacks),
        ]
        return row, card if card in self.t.cards else 0, slot, attack if attack in self.t.attacks else 0


def compact(x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Halve the memory of an encoded decision (float16 features) for storage and transfer."""
    return {k: v.astype(np.float16) if v.dtype == np.float32 else v for k, v in x.items()}


def _slot(side: int, area, index) -> int:
    if index is None:
        return NO_SLOT
    if area == AreaType.ACTIVE:
        return side * (1 + BENCH)
    if area == AreaType.BENCH and index < BENCH:
        return side * (1 + BENCH) + 1 + index
    return NO_SLOT


def _zone_card(player: dict, obs: dict, area, index):
    if area is None or index is None:
        return None
    zones = {
        AreaType.DECK: obs["select"].get("deck"),
        AreaType.HAND: player.get("hand"),
        AreaType.DISCARD: player.get("discard"),
        AreaType.ACTIVE: player.get("active"),
        AreaType.BENCH: player.get("bench"),
        AreaType.LOOKING: obs["current"].get("looking"),
        AreaType.PRIZE: player.get("prize"),
    }
    zone = zones.get(area)
    if not zone or index >= len(zone):
        return None
    return zone[index]
