"""Pit two agents against each other and report the win rate of the first one.

    python -m ptcg.evaluate heuristic random --games 400 --workers 8
    python -m ptcg.evaluate rl heuristic --model-a models/ppo/model.npz

Seats alternate between games so neither agent always sits in seat 0.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

from . import engine
from .agents import agent_name, make_agent
from .rollout import make_pool

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DECK = ROOT / "decks" / "abomasnow.csv"
DEFAULT_MODEL = ROOT / "models" / "ppo" / "model.npz"


def _play_chunk(name_a: str, name_b: str, deck_a: list[int], deck_b: list[int], games: int, seed: int,
                model_a: str, model_b: str):
    random.seed(seed)
    cards, attacks = engine.card_db(), engine.attack_db()
    agent_a = make_agent(name_a, cards, attacks, model_a, deck_a)
    agent_b = make_agent(name_b, cards, attacks, model_b, deck_b)
    wins = draws = 0
    for g in range(games):
        seat_a = (seed + g) % 2
        agents = (agent_a, agent_b) if seat_a == 0 else (agent_b, agent_a)
        decks = (deck_a, deck_b) if seat_a == 0 else (deck_b, deck_a)
        result = engine.play_game(agents, decks)
        draws += result == 2
        wins += result == seat_a
    return wins, draws


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("agent_a", type=agent_name, help="random, heuristic, rl or rl:<path>")
    parser.add_argument("agent_b", type=agent_name)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--deck-a", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--deck-b", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--model-a", default=str(DEFAULT_MODEL), help="weights if agent_a is rl")
    parser.add_argument("--model-b", default=str(DEFAULT_MODEL), help="weights if agent_b is rl")
    args = parser.parse_args()

    deck_a, deck_b = engine.load_deck(args.deck_a), engine.load_deck(args.deck_b)
    workers = max(1, min(args.workers, args.games))
    chunks = [args.games // workers + (i < args.games % workers) for i in range(workers)]
    start = time.time()
    with make_pool(workers) as pool:
        futures = [
            pool.submit(_play_chunk, args.agent_a, args.agent_b, deck_a, deck_b, n, seed, args.model_a, args.model_b)
            for seed, n in enumerate(chunks)
        ]
        results = [f.result() for f in futures]
    wins = sum(w for w, _ in results)
    draws = sum(d for _, d in results)
    n = args.games
    rate = (wins + draws / 2) / n
    ci = 1.96 * math.sqrt(rate * (1 - rate) / n)
    elapsed = time.time() - start
    print(f"{args.agent_a} vs {args.agent_b}: {wins} wins, {n - wins - draws} losses, {draws} draws")
    print(f"win rate {rate:.1%} +/- {ci:.1%} (95% CI), {n / elapsed:.1f} games/s")


if __name__ == "__main__":
    main()
