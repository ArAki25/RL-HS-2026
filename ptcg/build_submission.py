"""Package an agent and a deck for Kaggle as dist/submission.tar.gz.

    python -m ptcg.build_submission --agent heuristic --deck decks/abomasnow.csv
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

from . import engine
from .agents import AGENTS

ROOT = Path(__file__).resolve().parent.parent
# Only engine-free modules go into the submission.
PACKAGE_FILES = ["ptcg/__init__.py", "ptcg/enums.py", "ptcg/agents.py"]


def _add_text(tar: tarfile.TarFile, name: str, text: str) -> None:
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def build(agent: str, deck_path: Path, out: Path) -> Path:
    deck = engine.load_deck(deck_path)
    bundle = {
        "agent": agent,
        "cards": list(engine.card_db().values()),
        "attacks": list(engine.attack_db().values()),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        tar.add(ROOT / "submission" / "main.py", arcname="main.py")
        for name in PACKAGE_FILES:
            tar.add(ROOT / name, arcname=name)
        _add_text(tar, "deck.csv", "\n".join(map(str, deck)) + "\n")
        _add_text(tar, "bundle.json", json.dumps(bundle, ensure_ascii=False))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=AGENTS, default="heuristic")
    parser.add_argument("--deck", type=Path, default=ROOT / "decks" / "abomasnow.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "submission.tar.gz")
    args = parser.parse_args()
    out = build(args.agent, args.deck, args.out)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
