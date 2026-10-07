"""Generate simple theme decks (one evolution line per energy type) as training opponents.

    python -m ptcg.make_decks --per-type 2

Writes decks/gen_<type>_<n>.csv. The decks are not meant to be strong, only varied:
the RL agent should not overfit to a single opponent deck.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from . import engine
from .agents import HeuristicAgent
from .enums import CardType

ROOT = Path(__file__).resolve().parent.parent
ENERGY_NAMES = {
    1: "grass", 2: "fire", 3: "water", 4: "lightning", 5: "psychic", 6: "fighting", 7: "darkness", 8: "metal",
}

# Generic trainer package (card ID, copies); the line-specific cards are added in build_deck.
TRAINERS = [
    (1121, 4),  # Ultra Ball
    (1097, 2),  # Night Stretcher
    (1123, 2),  # Switch
    (1118, 2),  # Energy Retrieval
    (1124, 1),  # Pokémon Catcher
    (1182, 2),  # Boss's Orders
    (1227, 4),  # Lillie's Determination
    (1224, 2),  # Cheren
    (1219, 2),  # Team Rocket's Petrel
]
POFFIN, RARE_CANDY, DAWN, HILDA = 1086, 1079, 1231, 1225


def _payable(card: dict, energy: int, attacks: dict[int, dict]) -> bool:
    costs = [attacks[a]["energies"] for a in card["attacks"]]
    return bool(costs) and all(set(c) <= {0, energy} and len(c) <= 4 for c in costs)


def _strength(card: dict, attacks: dict[int, dict]) -> float:
    best = max(attacks[a]["damage"] / max(1, len(attacks[a]["energies"])) for a in card["attacks"])
    return best + card["hp"] / 10


def _lines(energy: int, cards: dict[int, dict], attacks: dict[int, dict]) -> list[tuple[float, list[dict]]]:
    """All evolution lines of one energy type whose attacks only need that type, strongest first."""
    pokemon = [c for c in cards.values() if c["cardType"] == CardType.POKEMON]
    by_name: dict[str, list[dict]] = {}
    for c in pokemon:
        by_name.setdefault(c["name"], []).append(c)

    def pre(card: dict, flag: str) -> dict | None:
        options = [c for c in by_name.get(card["evolvesFrom"] or "", []) if c[flag]]
        return max(options, key=lambda c: c["hp"], default=None)

    lines = []
    for top in pokemon:
        if top["energyType"] != energy or not _payable(top, energy, attacks):
            continue
        if top["stage2"]:
            mid = pre(top, "stage1")
            base = pre(mid, "basic") if mid else None
            line = [base, mid, top]
        elif top["stage1"]:
            line = [pre(top, "basic"), top]
        else:
            continue
        if all(line):
            lines.append((_strength(top, attacks), line))
    lines.sort(key=lambda x: -x[0])
    return lines


def build_deck(energy: int, line: list[dict], cards: dict[int, dict], attacks: dict[int, dict]) -> list[int]:
    deck: list[int] = []
    if len(line) == 3:
        counts, extra = [4, 2, 3], [(RARE_CANDY, 3), (DAWN, 2)]
    else:
        counts, extra = [4, 3], [(HILDA, 2)]
    for card, n in zip(line, counts):
        deck += [card["cardId"]] * n
    if line[0]["hp"] <= 70:
        extra.append((POFFIN, 2))
    # A strong Basic attacker of the same type as a second threat.
    basics = [
        c for c in cards.values()
        if c["cardType"] == CardType.POKEMON and c["basic"] and c["energyType"] == energy
        and c["name"] != line[0]["name"] and _payable(c, energy, attacks)
    ]
    if basics:
        deck += [max(basics, key=lambda c: _strength(c, attacks))["cardId"]] * 2
    for card_id, n in TRAINERS + extra:
        deck += [card_id] * n
    deck += [energy] * (60 - len(deck))  # Basic energy card IDs equal their energy type
    return deck


def is_legal(deck: list[int], cards: dict[int, dict]) -> bool:
    names = Counter(cards[c]["name"] for c in deck if cards[c]["cardType"] != CardType.BASIC_ENERGY)
    aces = sum(cards[c]["aceSpec"] for c in deck)
    return len(deck) == 60 and max(names.values()) <= 4 and aces <= 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-type", type=int, default=2, help="decks per energy type")
    parser.add_argument("--out", type=Path, default=ROOT / "decks")
    args = parser.parse_args()

    cards, attacks = engine.card_db(), engine.attack_db()
    heuristic = HeuristicAgent(cards, attacks)
    args.out.mkdir(parents=True, exist_ok=True)
    for energy, name in ENERGY_NAMES.items():
        written = 0
        for _, line in _lines(energy, cards, attacks):
            if written == args.per_type:
                break
            deck = build_deck(energy, line, cards, attacks)
            if not is_legal(deck, cards):
                continue
            try:  # the engine rejects illegal decks; also make sure games finish
                results = [engine.play_game((heuristic, heuristic), (deck, deck)) for _ in range(4)]
            except ValueError:
                continue
            written += 1
            path = args.out / f"gen_{name}_{written}.csv"
            path.write_text("\n".join(map(str, deck)) + "\n")
            print(f"{path.name}: {' > '.join(c['name'] for c in line)} (results {results})")


if __name__ == "__main__":
    main()
