# RL-HS-2026

Reinforcement-Learning-Projekt für den RL-Kurs an der FHNW: ein Bot für das Pokémon-Sammelkartenspiel
im Kaggle-Wettbewerb [PTCG AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview).

Der Hauptwettbewerb (Simulation) war bis 17.08.2026 offen. Abgaben laufen jetzt über den
Fortsetzungswettbewerb [PTCG AI Battle Challenge Playground](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground)
(Best-of-3-Matches gegen andere Bots).

## Setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install kaggle-environments==1.33.0 --no-deps
```

Die Spiel-Engine (`cg.dll` bzw. `libcg.so`) ist in `kaggle-environments` enthalten, ein Kaggle-Login
ist zum lokalen Spielen nicht nötig.

## Nutzung

```powershell
# Zwei Agents gegeneinander spielen lassen (Seiten wechseln jede Partie)
.venv\Scripts\python -m ptcg.evaluate heuristic random --games 400 --workers 10

# Abgabe bauen -> dist/submission.tar.gz
.venv\Scripts\python -m ptcg.build_submission --agent heuristic --deck decks/abomasnow.csv

# Abgabe mit dem offiziellen Kaggle-Runner testen und ein HTML-Replay speichern
.venv\Scripts\python -m ptcg.check_submission dist/submission.tar.gz --replay replays/check.html
```

## Aufbau

| Pfad | Inhalt |
| --- | --- |
| `ptcg/engine.py` | Direkter Zugriff auf die Engine: Partien spielen, Karten-/Attacken-Datenbank, Decks laden |
| `ptcg/agents.py` | Agents (`random`, `heuristic`); wird mit in die Abgabe gepackt |
| `ptcg/enums.py` | Konstanten der Engine-API ([Doku](https://matsuoinstitute.github.io/cabt/)) |
| `ptcg/evaluate.py` | Gewinnrate zweier Agents mit Konfidenzintervall |
| `ptcg/build_submission.py`, `ptcg/check_submission.py` | Abgabe bauen und testen |
| `submission/main.py` | Einstiegspunkt der Kaggle-Abgabe |
| `decks/` | Decklisten (60 Karten-IDs, eine pro Zeile) |

## Wie der Bot spielt

Die Engine fragt den Bot bei jeder Entscheidung nach einer Auswahl: `obs["select"]["option"]` enthält die
legalen Optionen, der Bot gibt eine Liste von Indizes zurück (zwischen `minCount` und `maxCount` viele).
Ganz am Anfang (`obs["select"] is None`) gibt er sein Deck zurück.

## Stand

| Agent | Gegner | Gewinnrate |
| --- | --- | --- |
| `heuristic` | `random` | 89 % ± 3 % (400 Partien, gleiches Deck) |
