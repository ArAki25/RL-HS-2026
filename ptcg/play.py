"""Play against a bot in the browser and record your decisions for training.

    python -m ptcg.play --opponent heuristic

Opens http://localhost:8000. Every decision you make is appended to data/human/<time>.jsonl as
{"obs": ..., "action": [...]}, followed by a {"result": ...} line when the game ends.
``python -m ptcg.train_bc`` learns from these files.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from . import engine
from .agents import agent_name, make_agent
from .describe import Describer

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).resolve().parent / "web" / "play.html"
DEFAULT_DECK = ROOT / "decks" / "abomasnow.csv"
DEFAULT_DATA = ROOT / "data" / "human"
LOG_LIMIT = 300


def count_recorded(data_dir: Path) -> int:
    total = 0
    for path in data_dir.glob("*.jsonl"):
        with open(path, encoding="utf-8") as f:
            total += sum(1 for line in f if line.startswith('{"obs"'))
    return total


class Session:
    """One game at a time between the browser player and a bot (the engine is single-game)."""

    def __init__(self, opponent: str, deck: list[int], bot_deck: list[int], data_dir: Path | None):
        self.cards, self.attacks = engine.card_db(), engine.attack_db()
        self.describer = Describer(self.cards, self.attacks)
        self.opponent = opponent
        self.deck, self.bot_deck = deck, bot_deck
        self.data_dir = data_dir
        self.recorded = count_recorded(data_dir) if data_dir else 0
        self.score = {"wins": 0, "losses": 0, "draws": 0}
        self.obs = None
        self.seat = 0
        self.bot = None
        self.log: list[str] = []
        self.record_file = None
        self.running = False
        self.in_engine = False  # a battle is open in the engine and must be finished

    # --- game flow -------------------------------------------------------------------------

    def new_game(self, opponent: str | None = None) -> None:
        self._finish()
        if opponent:
            self.opponent = opponent
        self.bot = make_agent(self.opponent, self.cards, self.attacks)
        self.seat = random.randrange(2)
        decks = (self.deck, self.bot_deck) if self.seat == 0 else (self.bot_deck, self.deck)
        obs, start = engine.game.battle_start(decks[0], decks[1])
        if obs is None:
            raise ValueError(f"deck of player {start.errorPlayer} rejected (error type {start.errorType})")
        self.obs = obs
        self.in_engine = True
        self.running = True
        self.log = [f"Neue Partie gegen {self.opponent}. Du bist Spieler {self.seat + 1}."]
        try:
            if self.data_dir:
                self.data_dir.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 10**6:06d}"
                opponent = "".join(c if c.isalnum() else "-" for c in self.opponent)
                self.record_file = open(self.data_dir / f"{stamp}_{opponent}.jsonl", "a", encoding="utf-8")
            self._advance()
        except Exception:
            self._finish()
            raise

    def select(self, indices: list[int]) -> None:
        if not self.running or self._to_move() != self.seat:
            raise ValueError("du bist nicht am Zug")
        select = self.obs["select"]
        n = len(select["option"])
        if len(set(indices)) != len(indices) or not all(0 <= i < n for i in indices):
            raise ValueError("ungültige Auswahl")
        if not select["minCount"] <= len(indices) <= select["maxCount"]:
            raise ValueError(f"wähle {select['minCount']} bis {select['maxCount']} Optionen")
        if self.record_file:
            self.record_file.write(json.dumps({"obs": self.obs, "action": indices}, ensure_ascii=False) + "\n")
            self.record_file.flush()
            self.recorded += 1
        self._log_choice("Du", indices)
        self.obs = engine.game.battle_select(indices)
        self._advance()

    def _to_move(self) -> int:
        return self.obs["current"]["yourIndex"]

    def _advance(self) -> None:
        """Let the bot play until it is the human's turn or the game is over."""
        for _ in range(engine.MAX_DECISIONS):
            result = self.obs["current"]["result"]
            if result >= 0:
                self._game_over(result)
                return
            if self._to_move() == self.seat:
                return
            action = self.bot(self.obs)
            self._log_choice("Bot", action)
            self.obs = engine.game.battle_select(action)
        self._game_over(2)

    def _log_choice(self, who: str, indices: list[int]) -> None:
        options = self.obs["select"]["option"]
        if len(options) <= 1 and who == "Bot":
            return
        labels = ", ".join(self.describer.option(options[i], self.obs) for i in indices) or "nichts"
        self.log.append(f"{who}: {labels}")
        del self.log[:-LOG_LIMIT]

    def _game_over(self, result: int) -> None:
        if result == self.seat:
            self.score["wins"] += 1
            self.log.append("Du hast gewonnen!")
        elif result == 2:
            self.score["draws"] += 1
            self.log.append("Unentschieden.")
        else:
            self.score["losses"] += 1
            self.log.append("Der Bot hat gewonnen.")
        if self.record_file:
            self.record_file.write(json.dumps({"result": result, "seat": self.seat, "opponent": self.opponent}) + "\n")
        self._finish()

    def _finish(self) -> None:
        if self.record_file:
            self.record_file.close()
            self.record_file = None
        self.running = False
        if self.in_engine:
            engine.game.battle_finish()
            self.in_engine = False

    # --- view for the browser --------------------------------------------------------------

    def view(self) -> dict:
        view = {
            "opponent": self.opponent,
            "running": self.running,
            "log": self.log,
            "score": self.score,
            "recorded": self.recorded,
            "recording": self.data_dir is not None,
            "board": None,
            "prompt": None,
        }
        if self.obs is None:
            return view
        view["board"] = self.describer.board(self.obs, self.seat)
        if self.running and self._to_move() == self.seat:
            select = self.obs["select"]
            view["prompt"] = {
                "text": self.describer.prompt(self.obs),
                "min": select["minCount"],
                "max": select["maxCount"],
                "options": [
                    {"label": self.describer.option(o, self.obs), "text": self.describer.option_text(o, self.obs)}
                    for o in select["option"]
                ],
            }
        return view


def make_handler(session: Session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep the console quiet
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: dict, status: int = 200) -> None:
            self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json")

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._json(session.view())
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            try:
                if self.path == "/api/new":
                    session.new_game(agent_name(body["opponent"]) if body.get("opponent") else None)
                elif self.path == "/api/select":
                    session.select([int(i) for i in body["indices"]])
                else:
                    self._send(404, b"not found", "text/plain")
                    return
                self._json(session.view())
            except (ValueError, OSError) as e:
                self._json({"error": str(e), **session.view()}, status=400)
            except Exception as e:  # show engine/agent bugs in the page instead of dropping the request
                traceback.print_exc()
                self._json({"error": f"Serverfehler: {e!r}"}, status=500)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--opponent", type=agent_name, default="heuristic", help="random, heuristic, model or model:<path>")
    parser.add_argument("--deck", type=Path, default=DEFAULT_DECK, help="your deck")
    parser.add_argument("--bot-deck", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="where recordings are written")
    parser.add_argument("--no-record", action="store_true", help="play without recording")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    session = Session(
        args.opponent,
        engine.load_deck(args.deck),
        engine.load_deck(args.bot_deck),
        None if args.no_record else args.data,
    )
    server = HTTPServer(("127.0.0.1", args.port), make_handler(session))
    url = f"http://localhost:{args.port}"
    print(f"Spiel läuft auf {url} (Strg+C zum Beenden)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        session._finish()


if __name__ == "__main__":
    main()
