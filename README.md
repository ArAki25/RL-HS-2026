# RL-HS-2026

Reinforcement-Learning-Projekt für den RL-Kurs an der FHNW: ein Bot für das Pokémon-Sammelkartenspiel
im Kaggle-Wettbewerb [PTCG AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview).

Der Hauptwettbewerb (Simulation) war bis 17.08.2026 offen. Abgaben laufen jetzt über den
Fortsetzungswettbewerb [PTCG AI Battle Challenge Playground](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground)
(Best-of-3-Matches gegen andere Bots).

## Setup (Windows)

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install kaggle-environments==1.33.0 --no-deps
```

`torch` und `tensorboard` braucht nur das Training; die Abgabe rechnet das Modell mit numpy.

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

## Selbst spielen und den Bot daraus lernen lassen

```powershell
# 1. Im Browser gegen einen Bot spielen (http://localhost:8000); jede deiner Entscheidungen
#    wird in data/human/*.jsonl gespeichert
.venv\Scripts\python -m ptcg.play --opponent heuristic

# 2. Modell trainieren: deine Partien (5-fach gewichtet) + Partien der Heuristik als Startwissen
#    -> models/bc.npz und models/bc.pt
.venv\Scripts\python -m ptcg.train_bc --epochs 20

# 3. Training live verfolgen (http://localhost:6006)
.venv\Scripts\tensorboard --logdir logs

# 4. Gegen das eigene Modell spielen bzw. es messen und abgeben
.venv\Scripts\python -m ptcg.play --opponent model
.venv\Scripts\python -m ptcg.evaluate model heuristic --games 400 --workers 10
.venv\Scripts\python -m ptcg.build_submission --agent model
```

Im Spiel wählt man Optionen per Klick oder mit den Tasten 1–9; bei Mehrfachauswahl mit Enter bestätigen.
Ein Tooltip auf Karten und Optionen zeigt den Kartentext. `--teacher-games 0` trainiert nur auf den
eigenen Partien, `--init models/bc.pt` trainiert ein vorhandenes Modell weiter.

In TensorBoard:

| Kurve | Bedeutung |
| --- | --- |
| `train/loss` | Fehler auf den Trainingsdaten |
| `val_human/accuracy`, `val_teacher/accuracy` | Anteil der Entscheidungen aus nicht gesehenen Partien, bei denen das Modell denselben Zug wählt (deine bzw. die der Heuristik) |
| `winrate/vs_random`, `winrate/vs_heuristic` | Gewinnrate des aktuellen Modells (alle `--eval-every` Epochen) |

## Aufbau

| Pfad | Inhalt |
| --- | --- |
| `ptcg/engine.py` | Direkter Zugriff auf die Engine: Partien spielen, Karten-/Attacken-Datenbank, Decks laden |
| `ptcg/agents.py` | Agents (`random`, `heuristic`, `model`); wird mit in die Abgabe gepackt |
| `ptcg/observation.py` | Karten zu einer Option auflösen |
| `ptcg/features.py` | Zustand und Optionen als Zahlenvektoren für das Modell |
| `ptcg/model.py`, `ptcg/policy.py` | Netz, das jede Option bewertet (PyTorch fürs Training, numpy für die Abgabe) |
| `ptcg/train_bc.py` | Training durch Nachahmen (Behavior Cloning) mit TensorBoard-Logs |
| `ptcg/play.py`, `ptcg/describe.py`, `ptcg/web/` | Browser-Spiel gegen einen Bot mit Aufzeichnung |
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
| `model` (Probelauf: 100 Heuristik-Partien, 3 Epochen) | `random` / `heuristic` | 90 % / 45 % (je 40 Partien) |
