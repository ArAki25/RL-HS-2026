"""Train the policy network: behavior cloning from the heuristic, then PPO with self-play.

    python -m ptcg.train bc --games 2000 --out models/bc
    python -m ptcg.train ppo --init models/bc/model.pt --iterations 300 --out models/ppo

Both commands write model.pt (torch checkpoint) and model.npz (weights for the submission,
see ptcg/policy.py) into --out. PPO additionally keeps snapshots of earlier policies as
opponents, evaluates the greedy policy against the heuristic and saves the best one as best.npz.
Evaluation is a mirror match by default (both play --deck), so 50% means "as good as the heuristic".
"""

from __future__ import annotations

import argparse
import csv
import glob
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import engine, rollout
from .features import Tables
from .model import Buffer, PolicyNet

ROOT = Path(__file__).resolve().parent.parent
OPPONENT_KINDS = ("self", "past", "heuristic", "random")


def _decks(pattern: str) -> list[list[int]]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no deck files match {pattern}")
    return [engine.load_deck(p) for p in paths]


def _save(model: PolicyNet, tables: Tables, out: Path, name: str, check: list[dict], **extra) -> None:
    torch.save({"model": model.state_dict(), **extra}, out / f"{name}.pt")
    model.export(out / f"{name}.npz", tables, check=check)


def evaluate(pool, weights: dict, deck: list[int], eval_decks: list[list[int]], games: int, workers: int,
             opponent: str = "heuristic", seed: int = 10**6) -> float:
    """Win rate (draws count half) of the greedy policy against a fixed opponent."""
    chunks = [games // workers + (i < games % workers) for i in range(workers)]
    futures = [
        pool.submit(rollout.policy_games, weights, opponent, None, [deck], eval_decks, n, seed + i, True, False)
        for i, n in enumerate(chunks) if n
    ]
    rewards = [r for f in futures for r in f.result()[1]]
    return (np.mean(rewards) + 1) / 2


def bc(args) -> None:
    cards, attacks = engine.card_db(), engine.attack_db()
    tables = Tables(cards, attacks)
    deck, decks = engine.load_deck(args.deck), _decks(args.opponent_decks)
    eval_decks = _decks(args.eval_decks) if args.eval_decks else [deck]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    start = time.time()
    with rollout.make_pool(args.workers) as pool:
        per_task = 25
        futures = [
            pool.submit(rollout.demo_games, min(per_task, args.games - i), deck, decks, args.random_share, i)
            for i in range(0, args.games, per_task)
        ]
        inputs, labels, outcomes = [], [], []
        for f in futures:
            x, y, z = f.result()
            inputs += x
            labels += y
            outcomes += z
        print(f"{args.games} demonstration games, {len(inputs)} decisions ({time.time() - start:.0f}s)", flush=True)

        buffer = Buffer(inputs, action=np.array(labels), outcome=np.array(outcomes, np.float32))
        check = inputs[:200]
        del inputs
        order = np.random.default_rng(0).permutation(len(buffer))
        val, train = order[: len(order) // 20], order[len(order) // 20:]

        torch.manual_seed(0)
        model = PolicyNet(tables)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(args.epochs):
            model.train()
            np.random.shuffle(train)
            for i in range(0, len(train), args.batch):
                b = buffer.batch(train[i: i + args.batch])
                logits, value = model(b)
                loss = F.cross_entropy(logits, b["action"]) + args.value_coef * F.mse_loss(value, b["outcome"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                correct = value_loss = 0.0
                for i in range(0, len(val), 2048):
                    b = buffer.batch(val[i: i + 2048])
                    logits, value = model(b)
                    correct += (logits.argmax(1) == b["action"]).sum().item()
                    value_loss += F.mse_loss(value, b["outcome"], reduction="sum").item()
            print(f"epoch {epoch + 1}: validation accuracy {correct / len(val):.1%}, "
                  f"value MSE {value_loss / len(val):.3f} ({time.time() - start:.0f}s)", flush=True)

        _save(model, tables, out, "model", check)
        weights = model.numpy_weights()
        for opponent in ("heuristic", "random"):
            rate = evaluate(pool, weights, deck, eval_decks, args.eval_games, args.workers, opponent)
            print(f"greedy BC policy vs {opponent}: win rate {rate:.1%} ({args.eval_games} games)", flush=True)
    print(f"saved {out / 'model.pt'} and {out / 'model.npz'}")


def _advantages(trajectories, gamma: float, lam: float):
    """Flatten (steps, reward) trajectories into samples with GAE advantages and returns."""
    inputs, actions, logps, advantages, returns = [], [], [], [], []
    for steps, reward in trajectories:
        gae, next_value = 0.0, 0.0
        adv = [0.0] * len(steps)
        for t in reversed(range(len(steps))):
            value = steps[t][3]
            r = reward if t == len(steps) - 1 else 0.0
            delta = r + gamma * next_value - value
            gae = delta + gamma * lam * gae
            adv[t] = gae
            next_value = value
        for (x, action, logp, value), a in zip(steps, adv):
            inputs.append(x)
            actions.append(action)
            logps.append(logp)
            advantages.append(a)
            returns.append(a + value)
    f32 = lambda v: np.array(v, np.float32)  # noqa: E731
    return inputs, np.array(actions), f32(logps), f32(advantages), f32(returns)


def _ppo_update(model, optimizer, buffer: Buffer, args) -> dict[str, float]:
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "kl": [], "clip_frac": []}
    model.train()
    for _ in range(args.epochs):
        order = np.random.permutation(len(buffer))
        for i in range(0, len(order), args.batch):
            b = buffer.batch(order[i: i + args.batch])
            logits, value = model(b)
            logp_all = F.log_softmax(logits, -1)
            logp = logp_all.gather(1, b["action"][:, None]).squeeze(1)
            ratio = torch.exp(logp - b["old_logp"])
            adv = b["advantage"]
            policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv).mean()
            value_loss = F.mse_loss(value, b["return"])
            entropy = -(logp_all.exp() * logp_all).sum(-1).mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.item())
                stats["kl"].append((b["old_logp"] - logp).mean().item())
                stats["clip_frac"].append(((ratio - 1).abs() > args.clip).float().mean().item())
        if np.mean(stats["kl"][-max(1, len(order) // args.batch):]) > args.target_kl:
            break  # the policy moved far enough for this batch of games
    return {k: float(np.mean(v)) for k, v in stats.items()}


def ppo(args) -> None:
    cards, attacks = engine.card_db(), engine.attack_db()
    tables = Tables(cards, attacks)
    deck, opponent_decks = engine.load_deck(args.deck), _decks(args.opponent_decks)
    eval_decks = _decks(args.eval_decks) if args.eval_decks else [deck]
    out = Path(args.out)
    snapshot_dir = out / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    mix = [float(x) for x in args.mix.split(",")]
    if len(mix) != len(OPPONENT_KINDS):
        raise SystemExit(f"--mix needs {len(OPPONENT_KINDS)} weights: {', '.join(OPPONENT_KINDS)}")

    torch.manual_seed(args.seed)
    model = PolicyNet(tables)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_iteration, best = 0, -1.0
    if args.resume and (out / "latest.pt").exists():
        state = torch.load(out / "latest.pt")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_iteration, best = state["iteration"] + 1, state["best"]
        print(f"resuming {out} at iteration {start_iteration}")
    elif args.init:
        model.load_state_dict(torch.load(args.init)["model"])
        print(f"initialized from {args.init}")
    snapshots = sorted(snapshot_dir.glob("*.npz"))

    log_path = out / "log.csv"
    new_log = not log_path.exists() or start_iteration == 0
    rng = random.Random(args.seed + start_iteration)
    with rollout.make_pool(args.workers) as pool, open(log_path, "w" if new_log else "a", newline="") as log_file:
        log = csv.writer(log_file)
        if new_log:
            log.writerow(["iteration", "games", "samples", *[f"reward_{k}" for k in OPPONENT_KINDS],
                          "policy_loss", "value_loss", "entropy", "kl", "clip_frac", "eval_heuristic", "seconds"])
        for it in range(start_iteration, args.iterations):
            start = time.time()
            weights = model.numpy_weights()
            futures = []
            for t in range(args.tasks):
                kind = rng.choices(OPPONENT_KINDS, mix)[0]
                if kind == "past" and not snapshots:
                    kind = "self"
                if kind == "past":
                    opp = rollout.load_weights(rng.choice(snapshots[-args.pool_size:]))
                    task = (weights, "policy", opp)
                else:
                    task = (weights, kind, None)
                seed = it * 1000 + t
                futures.append((kind, pool.submit(rollout.policy_games, *task, [deck], opponent_decks,
                                                  args.games_per_task, seed)))
            trajectories, rewards = [], {k: [] for k in OPPONENT_KINDS}
            for kind, f in futures:
                traj, r = f.result()
                trajectories += [t for t in traj if t[0]]
                rewards[kind] += r
            inputs, actions, logps, advantages, returns = _advantages(trajectories, args.gamma, args.lam)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            buffer = Buffer(inputs, action=actions, old_logp=logps, advantage=advantages, **{"return": returns})
            stats = _ppo_update(model, optimizer, buffer, args)

            weights = model.numpy_weights()
            if (it + 1) % args.snapshot_every == 0:
                np.savez_compressed(snapshot_dir / f"iter_{it + 1:05d}.npz", **weights)
                snapshots = sorted(snapshot_dir.glob("*.npz"))
            eval_rate = ""
            if (it + 1) % args.eval_every == 0:
                eval_rate = evaluate(pool, weights, deck, eval_decks, args.eval_games, args.workers)
                if eval_rate > best:
                    best = eval_rate
                    model.export(out / "best.npz", tables, check=inputs[:100])
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": it,
                        "best": best}, out / "latest.pt")
            model.export(out / "latest.npz", tables)

            games = sum(len(r) for r in rewards.values())
            means = {k: (f"{np.mean(v):+.2f}" if v else "") for k, v in rewards.items()}
            seconds = time.time() - start
            log.writerow([it + 1, games, len(buffer), *means.values(), *(f"{stats[k]:.4f}" for k in stats),
                          f"{eval_rate:.3f}" if eval_rate != "" else "", f"{seconds:.1f}"])
            log_file.flush()
            print(f"it {it + 1:4d} | {len(buffer):6d} samples | reward " +
                  " ".join(f"{k} {v or '  -  '}" for k, v in means.items()) +
                  f" | ent {stats['entropy']:.2f} kl {stats['kl']:.3f} vloss {stats['value_loss']:.3f}" +
                  (f" | eval vs heuristic {eval_rate:.1%} (best {best:.1%})" if eval_rate != "" else "") +
                  f" | {seconds:.0f}s", flush=True)
    if (out / "best.npz").exists():
        shutil.copy(out / "best.npz", out / "model.npz")
        print(f"best policy ({best:.1%} vs heuristic) copied to {out / 'model.npz'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--deck", default=str(ROOT / "decks" / "abomasnow.csv"), help="deck of the trained agent")
    common.add_argument("--opponent-decks", default=str(ROOT / "decks" / "*.csv"), help="glob of opponent decks")
    common.add_argument("--workers", type=int, default=10)
    common.add_argument("--eval-games", type=int, default=200)
    common.add_argument("--eval-decks", help="glob of the heuristic's decks in evaluation (default: --deck)")
    common.add_argument("--value-coef", type=float, default=0.5)

    p = sub.add_parser("bc", parents=[common], help="imitate the heuristic agent")
    p.add_argument("--games", type=int, default=2000)
    p.add_argument("--random-share", type=float, default=0.2, help="share of games against the random agent")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default=str(ROOT / "models" / "bc"))
    p.set_defaults(run=bc)

    p = sub.add_parser("ppo", parents=[common], help="PPO with self-play against an opponent pool")
    p.add_argument("--init", help="start from this checkpoint (e.g. models/bc/model.pt)")
    p.add_argument("--resume", action="store_true", help="continue the run in --out")
    p.add_argument("--out", default=str(ROOT / "models" / "ppo"))
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--tasks", type=int, default=20, help="rollout tasks per iteration (one opponent each)")
    p.add_argument("--games-per-task", type=int, default=8)
    p.add_argument("--mix", default="0.4,0.3,0.25,0.05", help="opponent weights: " + ",".join(OPPONENT_KINDS))
    p.add_argument("--pool-size", type=int, default=20, help="sample past opponents from the last N snapshots")
    p.add_argument("--snapshot-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--target-kl", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(run=ppo)

    args = parser.parse_args()
    torch.set_num_threads(args.workers)
    args.run(args)


if __name__ == "__main__":
    main()
