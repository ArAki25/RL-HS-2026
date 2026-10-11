"""Behavior cloning: train the option-scoring network to copy recorded decisions.

    python -m ptcg.train_bc --epochs 20
    tensorboard --logdir logs

Training data:
  * your games from ``python -m ptcg.play`` (data/human/*.jsonl), weighted by --human-weight
  * games of a teacher agent playing itself (--teacher heuristic --teacher-games N), so the
    model already plays sensibly before there are many human games

The model is saved as models/bc.npz (use it with agent name "model") and models/bc.pt.
TensorBoard shows loss, how often the model picks the recorded option, and win rates.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from . import engine
from .agents import agent_name, make_agent
from .evaluate import DEFAULT_DECK, _play_chunk
from .features import Encoder
from .model import OptionScorer

ROOT = Path(__file__).resolve().parent.parent

# One training example: (state vector, option matrix, indices that were chosen).
Sample = tuple[np.ndarray, np.ndarray, list[int]]


def _useful(obs: dict, action: list[int]) -> bool:
    """Decisions with a single option or an empty choice teach nothing about ranking."""
    return len(obs["select"]["option"]) > 1 and len(action) > 0


def load_human(data_dir: Path, encoder: Encoder) -> list[list[Sample]]:
    games = []
    for path in sorted(data_dir.glob("*.jsonl")):
        game = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if "obs" in record and _useful(record["obs"], record["action"]):
                    game.append((*encoder.encode(record["obs"]), record["action"]))
        if game:
            games.append(game)
    return games


def _teacher_chunk(teacher: str, deck: list[int], games: int, seed: int) -> list[list[Sample]]:
    random.seed(seed)
    cards, attacks = engine.card_db(), engine.attack_db()
    encoder = Encoder(cards, attacks)
    agent = make_agent(teacher, cards, attacks)
    result = []
    for _ in range(games):
        game: list[Sample] = []

        def recording_agent(obs: dict) -> list[int]:
            action = agent(obs)
            if _useful(obs, action):
                game.append((*encoder.encode(obs), list(action)))
            return action

        engine.play_game((recording_agent, recording_agent), (deck, deck))
        result.append(game)
    return result


def teacher_games(teacher: str, deck: list[int], games: int, workers: int) -> list[list[Sample]]:
    workers = max(1, min(workers, games))
    chunks = [games // workers + (i < games % workers) for i in range(workers)]
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(_teacher_chunk, teacher, deck, n, 1000 + i) for i, n in enumerate(chunks)]
        return [g for f in futures for g in f.result()]


def split(games: list[list[Sample]], val_fraction: float, rng: random.Random):
    """Split by whole games so validation positions come from games the model never saw."""
    games = games[:]
    rng.shuffle(games)
    n_val = int(round(len(games) * val_fraction)) if len(games) > 1 else 0
    flat = lambda gs: [s for g in gs for s in g]  # noqa: E731
    return flat(games[n_val:]), flat(games[:n_val])


def collate(batch: list[tuple[Sample, float]]):
    """Pad option lists to the longest one in the batch."""
    k = max(len(s[1]) for s, _ in batch)
    option_dim = batch[0][0][1].shape[1]
    states = torch.from_numpy(np.stack([s[0] for s, _ in batch]))
    options = torch.zeros(len(batch), k, option_dim)
    mask = torch.zeros(len(batch), k, dtype=torch.bool)
    target = torch.zeros(len(batch), k)
    for i, ((_, opts, action), _) in enumerate(batch):
        options[i, : len(opts)] = torch.from_numpy(opts)
        mask[i, : len(opts)] = True
        target[i, action] = 1.0 / len(action)
    weights = torch.tensor([w for _, w in batch])
    return states, options, mask, target, weights


def batch_loss(model: OptionScorer, batch):
    states, options, mask, target, weights = batch
    logits = model(states, options).masked_fill(~mask, float("-inf"))
    log_probs = torch.log_softmax(logits, dim=-1).masked_fill(~mask, 0.0)
    loss = -(target * log_probs).sum(-1)
    hit = target.gather(1, logits.argmax(-1, keepdim=True)).squeeze(1) > 0
    return (loss * weights).sum() / weights.sum(), loss, hit


@torch.no_grad()
def evaluate_samples(model: OptionScorer, samples: list[Sample], batch_size: int) -> tuple[float, float]:
    """Mean loss and how often the top-scored option is one that was chosen."""
    model.eval()
    losses, hits = [], []
    for i in range(0, len(samples), batch_size):
        _, loss, hit = batch_loss(model, collate([(s, 1.0) for s in samples[i : i + batch_size]]))
        losses.append(loss)
        hits.append(hit)
    model.train()
    return torch.cat(losses).mean().item(), torch.cat(hits).float().mean().item()


def win_rate(model_path: Path, opponent: str, deck: list[int], games: int, workers: int) -> float:
    workers = max(1, min(workers, games))
    chunks = [games // workers + (i < games % workers) for i in range(workers)]
    with ProcessPoolExecutor(workers) as pool:
        futures = [
            pool.submit(_play_chunk, f"model:{model_path}", opponent, deck, deck, n, seed)
            for seed, n in enumerate(chunks)
        ]
        results = [f.result() for f in futures]
    wins = sum(w for w, _ in results)
    draws = sum(d for _, d in results)
    return (wins + draws / 2) / games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--human-data", type=Path, default=ROOT / "data" / "human")
    parser.add_argument("--human-weight", type=float, default=5.0, help="loss weight of your decisions vs teacher ones")
    parser.add_argument("--teacher", type=agent_name, default="heuristic")
    parser.add_argument("--teacher-games", type=int, default=300, help="0 = train on your games only")
    parser.add_argument("--deck", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=5, help="epochs between win-rate games (0 = never)")
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--init", type=Path, help="continue from a .pt checkpoint")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "bc")
    parser.add_argument("--logdir", type=Path, default=ROOT / "logs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    deck = engine.load_deck(args.deck)
    encoder = Encoder(engine.card_db(), engine.attack_db())

    start = time.time()
    human = load_human(args.human_data, encoder) if args.human_data.is_dir() else []
    teacher = teacher_games(args.teacher, deck, args.teacher_games, args.workers) if args.teacher_games else []
    human_train, human_val = split(human, args.val_fraction, rng)
    teacher_train, teacher_val = split(teacher, args.val_fraction, rng)
    train = [(s, args.human_weight) for s in human_train] + [(s, 1.0) for s in teacher_train]
    if not train:
        raise SystemExit(f"no training data: play games with `python -m ptcg.play` or use --teacher-games")
    print(
        f"data: {len(human_train)}/{len(human_val)} human (train/val) from {len(human)} games, "
        f"{len(teacher_train)}/{len(teacher_val)} {args.teacher} from {len(teacher)} games "
        f"({time.time() - start:.0f}s)"
    )

    model = OptionScorer(encoder.state_dim, encoder.option_dim)
    if args.init:
        model.load_state_dict(torch.load(args.init))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    run = args.logdir / f"bc-{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(str(run))
    writer.add_text(
        "setup",
        f"human games {len(human)}, human decisions {len(human_train) + len(human_val)}, "
        f"teacher {args.teacher} games {len(teacher)}, human weight {args.human_weight}",
    )
    npz_path, pt_path = args.out.with_suffix(".npz"), args.out.with_suffix(".pt")
    print(f"TensorBoard: tensorboard --logdir {args.logdir}  (run {run.name})")

    step = 0
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train)
        total, count = 0.0, 0
        for i in range(0, len(train), args.batch_size):
            loss, _, _ = batch_loss(model, collate(train[i : i + args.batch_size]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            total += loss.item()
            count += 1
            writer.add_scalar("train/loss", loss.item(), step)
        line = f"epoch {epoch:3d}  train loss {total / count:.3f}"
        for name, samples in (("human", human_val), ("teacher", teacher_val)):
            if samples:
                val_loss, acc = evaluate_samples(model, samples, args.batch_size)
                writer.add_scalar(f"val_{name}/loss", val_loss, epoch)
                writer.add_scalar(f"val_{name}/accuracy", acc, epoch)
                line += f"  {name} val loss {val_loss:.3f} acc {acc:.1%}"

        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), pt_path)
        model.export_numpy(npz_path)
        if args.eval_every and (epoch % args.eval_every == 0 or epoch == args.epochs):
            for opponent in ("random", "heuristic"):
                rate = win_rate(npz_path, opponent, deck, args.eval_games, args.workers)
                writer.add_scalar(f"winrate/vs_{opponent}", rate, epoch)
                line += f"  vs {opponent} {rate:.0%}"
        print(line, flush=True)
        writer.flush()

    writer.close()
    print(f"saved {npz_path} and {pt_path}")


if __name__ == "__main__":
    main()
