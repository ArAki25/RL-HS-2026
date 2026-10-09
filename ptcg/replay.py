"""Watch two agents play: an HTML replay (official Kaggle visualizer) plus a play-by-play log.

    python -m ptcg.replay rl heuristic --model-a models/ppo/model.npz

Writes replays/<agent_a>_vs_<agent_b>.html (open it in a browser) and a .txt log next to it. The log
lists every decision of both agents; for the rl agent also its estimated win chance and the three
options it found most likely, e.g.

    turn  5 | rl        | main | win chance 64%
         71%  Attack Frost Barrier (200)
         18%  Attach Basic {W} Energy -> my bench 1 Snover
          6%  End turn
      -> Attack Frost Barrier (200)
"""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import numpy as np

from .agents import AGENTS, make_agent
from .enums import AreaType, OptionType, SelectContext
from .features import zone_card
from .policy import PolicyAgent

ROOT = Path(__file__).resolve().parent.parent
ENERGY = ["Colorless", "Grass", "Fire", "Water", "Lightning", "Psychic", "Fighting", "Darkness", "Metal",
          "Dragon", "Rainbow", "Rocket"]
AREA = {AreaType.DECK: "deck", AreaType.HAND: "hand", AreaType.DISCARD: "discard", AreaType.ACTIVE: "active",
        AreaType.BENCH: "bench", AreaType.PRIZE: "prize", AreaType.LOOKING: "looking"}


def _databases() -> tuple[dict[int, dict], dict[int, dict]]:
    """Card and attack data from the engine that kaggle_environments loaded.

    ptcg.engine cannot be used here: loading the engine library a second time in the same process crashes.
    """
    from kaggle_environments.envs.cabt.cg.sim import lib

    lib.AllCard.restype = lib.AllAttack.restype = ctypes.c_char_p
    cards = {c["cardId"]: c for c in json.loads(lib.AllCard().decode("utf-8"))}
    attacks = {a["attackId"]: a for a in json.loads(lib.AllAttack().decode("utf-8"))}
    return cards, attacks


def describe(option: dict, obs: dict, cards: dict[int, dict], attacks: dict[int, dict]) -> str:
    """Human-readable text for one option."""
    cur = obs["current"]
    me = cur["yourIndex"]
    kind = option["type"]
    owner = option.get("playerIndex", me)
    player = cur["players"][owner]

    def name(card) -> str:
        if card is None:
            return "hidden card"
        return cards[card["id"]]["name"] if card["id"] in cards else f"card {card['id']}"

    def place(area, index) -> str:
        who = "my" if owner == me else "opponent's"
        where = AREA.get(area, f"area {area}") + (f" {index + 1}" if area == AreaType.BENCH else "")
        return f"{who} {where}"

    def in_play(area, index) -> str:
        return f"{place(area, index)} {name(zone_card(cur['players'][me], obs, area, index))}"

    hand_card = name(zone_card(cur["players"][me], obs, AreaType.HAND, option.get("index")))
    if kind == OptionType.ATTACK:
        attack = attacks.get(option["attackId"], {})
        return f"Attack {attack.get('name', option['attackId'])} ({attack.get('damage', '?')})"
    if kind == OptionType.ATTACH:
        return f"Attach {hand_card} -> {in_play(option['inPlayArea'], option['inPlayIndex'])}"
    if kind == OptionType.EVOLVE:
        return f"Evolve {in_play(option['inPlayArea'], option['inPlayIndex'])} into {hand_card}"
    if kind == OptionType.PLAY:
        return f"Play {hand_card}"
    if kind == OptionType.ABILITY and option.get("area") == AreaType.STADIUM:
        return f"Use stadium {name((cur.get('stadium') or [None])[0])}"
    if kind == OptionType.ABILITY:
        return f"Ability of {in_play(option['area'], option['index'])}"
    if kind == OptionType.ENERGY:
        poke = zone_card(player, obs, option["area"], option["index"])
        energies = (poke or {}).get("energies") or []
        energy = energies[option["energyIndex"]] if option["energyIndex"] < len(energies) else 0
        return f"{ENERGY[energy]} energy on {place(option['area'], option['index'])} {name(poke)}"
    if kind in (OptionType.CARD, OptionType.TOOL_CARD, OptionType.ENERGY_CARD):
        card = zone_card(player, obs, option.get("area"), option.get("index"))
        return f"{name(card)} ({place(option.get('area'), option.get('index'))})"
    if kind == OptionType.SKILL:
        return f"Effect of {cards.get(option['cardId'], {}).get('name', option['cardId'])}"
    if kind == OptionType.NUMBER:
        return f"Number {option['number']}"
    return {OptionType.YES: "Yes", OptionType.NO: "No", OptionType.END: "End turn", OptionType.RETREAT: "Retreat",
            OptionType.DISCARD: "Discard", OptionType.SPECIAL_CONDITION: "Special condition"}.get(kind, str(kind))


class Narrator:
    """Wraps an agent, returns its deck at the start and logs each of its decisions."""

    def __init__(self, agent, label: str, deck: list[int], cards: dict, attacks: dict, lines: list[str]):
        self.agent, self.label, self.deck = agent, label, deck
        self.cards, self.attacks, self.lines = cards, attacks, lines

    def __call__(self, obs, config=None):
        if obs["select"] is None:
            return self.deck
        select = obs["select"]
        options = select["option"]
        context = SelectContext(select["context"]).name.lower()
        header = f"turn {obs['current']['turn']:2d} | {self.label:<9} | {context}"
        top = []
        if isinstance(self.agent, PolicyAgent) and len(options) > 1:
            # The policy's view of the first pick: the same input PolicyAgent builds for it.
            can_stop = select["minCount"] == 0
            x = self.agent.encoder.encode(obs, list(range(len(options))), 0, can_stop)
            logits, value = self.agent.policy(x)
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            header += f" | win chance {(value + 1) / 2:.0%}"
            grouped: dict[str, float] = {}  # identical options (e.g. three copies of a card) are summed
            for i, p in enumerate(probs):
                text = "Stop" if i == len(options) else describe(options[i], obs, self.cards, self.attacks)
                grouped[text] = grouped.get(text, 0.0) + p
            for text, p in sorted(grouped.items(), key=lambda kv: -kv[1])[:3]:
                top.append(f"     {p:4.0%}  {text}")
        chosen = self.agent(obs)
        if len(options) > 1 or select["minCount"] == 0:  # skip forced moves
            picks = ", ".join(describe(options[i], obs, self.cards, self.attacks) for i in chosen) or "(nothing)"
            self.lines += [header, *top, f"  -> {picks}"]
        return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("agent_a", choices=AGENTS)
    parser.add_argument("agent_b", choices=AGENTS)
    parser.add_argument("--model-a", default=str(ROOT / "models" / "ppo" / "model.npz"))
    parser.add_argument("--model-b", default=str(ROOT / "models" / "ppo" / "model.npz"))
    parser.add_argument("--deck-a", type=Path, default=ROOT / "decks" / "abomasnow.csv")
    parser.add_argument("--deck-b", type=Path, default=ROOT / "decks" / "abomasnow.csv")
    parser.add_argument("--bo", type=int, default=1, help="best-of format (games in the match)")
    parser.add_argument("--out", type=Path, help="HTML file (default: replays/<a>_vs_<b>.html)")
    args = parser.parse_args()

    from kaggle_environments import make  # slow import, only needed here

    cards, attacks = _databases()
    lines: list[str] = []
    agents = []
    for name, model, deck_path, label in [(args.agent_a, args.model_a, args.deck_a, f"{args.agent_a} (A)"),
                                          (args.agent_b, args.model_b, args.deck_b, f"{args.agent_b} (B)")]:
        deck = [int(line) for line in deck_path.read_text().split()]
        agents.append(Narrator(make_agent(name, cards, attacks, model, deck), label, deck, cards, attacks, lines))

    env = make("cabt", configuration={"bo": args.bo})
    env.run(agents)
    rewards = [s.reward for s in env.steps[-1]]
    out = args.out or ROOT / "replays" / f"{args.agent_a}_vs_{args.agent_b}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(env.render(mode="html"), encoding="utf-8")
    result = {1: "A wins", -1: "B wins", 0: "draw"}.get(rewards[0], str(rewards))
    lines.append(f"result: {result} (rewards {rewards})")
    out.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{result}; replay: {out}, log: {out.with_suffix('.txt')} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
