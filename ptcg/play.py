"""Play against a bot in the terminal.

    python -m ptcg.play --opponent rl --model models/ppo/best.npz

Each decision shows the board and the numbered options; type the number(s) of your choice
(several numbers separated by spaces when the game asks for more than one card, empty input
for "nothing" if the game allows it). The bot's moves are printed as they happen.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from . import engine
from .agents import AGENTS, make_agent
from .enums import SelectContext
from .replay import ENERGY, describe

ROOT = Path(__file__).resolve().parent.parent


def board(obs: dict, cards: dict[int, dict]) -> str:
    cur = obs["current"]
    me = cur["yourIndex"]

    def pokemon(p: dict | None) -> str:
        if p is None:
            return "-"
        energies = ",".join(f"{p['energies'].count(t)}x{ENERGY[t]}" for t in sorted(set(p["energies"])))
        tools = f" +{len(p['tools'])} tool" if p["tools"] else ""
        return f"{cards[p['id']]['name']} {p['hp']}/{p['maxHp']}hp [{energies or 'no energy'}]{tools}"

    lines = [f"\n=== Turn {cur['turn']} " + "=" * 50]
    for label, idx in (("Opponent", 1 - me), ("You", me)):
        p = cur["players"][idx]
        status = [k for k in ("poisoned", "burned", "asleep", "paralyzed", "confused") if p.get(k)]
        lines.append(f"{label}: prizes {len(p['prize'])}, deck {p['deckCount']}, hand {p['handCount']}"
                     + (f", {', '.join(status)}" if status else ""))
        lines.append(f"  Active: {pokemon(p['active'][0] if p['active'] else None)}")
        for i, b in enumerate(p["bench"]):
            lines.append(f"  Bench {i + 1}: {pokemon(b)}")
        if idx == me:
            hand = ", ".join(cards[c["id"]]["name"] for c in p["hand"])
            lines.append(f"  Hand: {hand}")
    if cur.get("stadium"):
        lines.append(f"Stadium: {cards[cur['stadium'][0]['id']]['name']}")
    return "\n".join(lines)


class Human:
    def __init__(self, cards: dict, attacks: dict):
        self.cards, self.attacks = cards, attacks

    def __call__(self, obs: dict) -> list[int]:
        select = obs["select"]
        options = select["option"]
        print(board(obs, self.cards))
        low, high = select["minCount"], min(select["maxCount"], len(options))
        print(f"\n{SelectContext(select['context']).name.lower()}: choose "
              + (f"{low}" if low == high else f"{low} to {high}"))
        for i, option in enumerate(options):
            print(f"  [{i}] {describe(option, obs, self.cards, self.attacks)}")
        while True:
            try:
                picks = [int(x) for x in input("> ").split()]
            except ValueError:
                print("numbers only")
                continue
            if low <= len(picks) <= high and len(set(picks)) == len(picks) and all(0 <= i < len(options) for i in picks):
                return picks
            print(f"need {low}-{high} different numbers between 0 and {len(options) - 1}")


class Announced:
    """Wraps the bot and prints what it does."""

    def __init__(self, agent, cards: dict, attacks: dict):
        self.agent, self.cards, self.attacks = agent, cards, attacks

    def __call__(self, obs: dict) -> list[int]:
        chosen = self.agent(obs)
        options = obs["select"]["option"]
        if len(options) > 1 or obs["select"]["minCount"] == 0:
            what = ", ".join(describe(options[i], obs, self.cards, self.attacks) for i in chosen) or "(nothing)"
            print(f"  Bot: {what}")
        return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--opponent", choices=AGENTS, default="rl")
    parser.add_argument("--model", default=str(ROOT / "models" / "ppo" / "best.npz"))
    parser.add_argument("--deck", type=Path, default=ROOT / "decks" / "abomasnow.csv", help="your deck")
    parser.add_argument("--bot-deck", type=Path, default=ROOT / "decks" / "abomasnow.csv")
    parser.add_argument("--seat", type=int, choices=(0, 1), help="0 = you start as first player (default: random)")
    args = parser.parse_args()

    cards, attacks = engine.card_db(), engine.attack_db()
    mine, theirs = engine.load_deck(args.deck), engine.load_deck(args.bot_deck)
    bot = Announced(make_agent(args.opponent, cards, attacks, args.model, theirs), cards, attacks)
    seat = random.randint(0, 1) if args.seat is None else args.seat
    agents = (Human(cards, attacks), bot) if seat == 0 else (bot, Human(cards, attacks))
    decks = (mine, theirs) if seat == 0 else (theirs, mine)
    result = engine.play_game(agents, decks)
    print("\n" + ("Draw." if result == 2 else "You win!" if result == seat else "The bot wins."))


if __name__ == "__main__":
    main()
