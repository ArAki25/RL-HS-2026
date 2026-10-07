"""Kaggle entry point. Packaged into dist/submission.tar.gz by `python -m ptcg.build_submission`."""

import json
import os
import sys

# Kaggle exec()s this file without __file__, and its directory is on sys.path only while loading.
_DIR = next(
    p
    for p in [*reversed(sys.path), "/kaggle_simulations/agent", os.getcwd()]
    if os.path.isfile(os.path.join(p, "deck.csv"))
)
sys.path.insert(0, _DIR)

from ptcg.agents import make_agent  # noqa: E402

with open(os.path.join(_DIR, "deck.csv")) as f:
    DECK = [int(line) for line in f if line.strip()]
with open(os.path.join(_DIR, "bundle.json"), encoding="utf-8") as f:
    _bundle = json.load(f)
_agent = make_agent(
    _bundle["agent"],
    {c["cardId"]: c for c in _bundle["cards"]},
    {a["attackId"]: a for a in _bundle["attacks"]},
)


# Kaggle uses the last callable defined in this file as the agent.
def agent(obs, config=None):
    if obs["select"] is None:
        return DECK
    return _agent(obs)
