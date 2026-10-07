"""Run a packaged submission through the official kaggle-environments runner.

    python -m ptcg.check_submission dist/submission.tar.gz --opponent random --replay replays/check.html

The archive is unpacked into a temporary directory and played in a separate Python
process from there, so it only sees the files inside the archive (like on Kaggle).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

RUNNER = """
import json, sys, time
from kaggle_environments import make
bo, opponent, replay = int(sys.argv[1]), sys.argv[2], sys.argv[3]
env = make("cabt", configuration={"bo": bo})
start = time.time()
env.run(["main.py", opponent])
final = env.steps[-1]
if replay:
    with open(replay, "w", encoding="utf-8") as f:
        f.write(env.render(mode="html"))
print(json.dumps({
    "status": [s.status for s in final],
    "reward": [s.reward for s in final],
    "games": env.result if hasattr(env, "result") else None,
    "seconds": round(time.time() - start, 2),
    "time_left": final[0].observation.remainingOverageTime,
    "error": env.steps[0][0].get("error"),
}))
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--opponent", default="random", help='built-in "random"/"first" or path to a main.py')
    parser.add_argument("--matches", type=int, default=3)
    parser.add_argument("--bo", type=int, default=3, help="best-of format of one match")
    parser.add_argument("--replay", type=Path, help="write an HTML replay of the last match here")
    args = parser.parse_args()

    opponent = args.opponent
    if Path(opponent).is_file():
        opponent = str(Path(opponent).resolve())
    replay = str(args.replay.resolve()) if args.replay else ""
    if args.replay:
        args.replay.parent.mkdir(parents=True, exist_ok=True)

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(args.archive) as tar:
            tar.extractall(tmp, filter="data")
        for i in range(args.matches):
            proc = subprocess.run(
                [sys.executable, "-c", RUNNER, str(args.bo), opponent, replay if i == args.matches - 1 else ""],
                cwd=tmp,
                capture_output=True,
                text=True,
            )
            lines = proc.stdout.strip().splitlines()
            if proc.returncode != 0 or not lines:
                print(proc.stdout, proc.stderr, sep="\n")
                sys.exit(f"match {i + 1}: runner crashed")
            result = json.loads(lines[-1])
            print(f"match {i + 1}: {result}")
            ok &= result["status"] == ["DONE", "DONE"] and not result["error"]
    if not ok:
        sys.exit("submission did not finish cleanly (see status/error above)")
    print("submission OK")


if __name__ == "__main__":
    main()
