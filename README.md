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
ist zum lokalen Spielen nicht nötig. PyTorch (CPU reicht) wird nur fürs Training gebraucht.

## Nutzung

```powershell
# 1. Gegner-Decks erzeugen (einmalig) -> decks/gen_*.csv
.venv\Scripts\python -m ptcg.make_decks

# 2. Behavior Cloning: die Heuristik imitieren -> models/bc/model.pt
.venv\Scripts\python -m ptcg.train bc --games 2000

# 3. PPO mit Self-Play, ausgehend vom BC-Modell -> models/ppo/model.npz (bestes Modell)
.venv\Scripts\python -m ptcg.train ppo --init models/bc/model.pt --iterations 300
#    unterbrochenen Lauf fortsetzen:
.venv\Scripts\python -m ptcg.train ppo --resume --iterations 300

# Zwei Agents gegeneinander spielen lassen (Seiten wechseln jede Partie)
.venv\Scripts\python -m ptcg.evaluate rl heuristic --model-a models/ppo/model.npz --games 1000 --workers 10

# Abgabe bauen -> dist/submission.tar.gz
.venv\Scripts\python -m ptcg.build_submission --agent rl --model models/ppo/model.npz

# Abgabe mit dem offiziellen Kaggle-Runner testen und ein HTML-Replay speichern
.venv\Scripts\python -m ptcg.check_submission dist/submission.tar.gz --replay replays/check.html
```

Der PPO-Lauf schreibt pro Iteration eine Zeile nach `models/ppo/log.csv` (Belohnung je Gegnertyp,
Verluste, Entropie, KL, Gewinnrate gegen die Heuristik) und legt alle 10 Iterationen einen Snapshot
in `models/ppo/snapshots/` ab.

## Aufbau

| Pfad | Inhalt |
| --- | --- |
| `ptcg/engine.py` | Direkter Zugriff auf die Engine: Partien spielen, Karten-/Attacken-Datenbank, Decks laden |
| `ptcg/agents.py` | Agents (`random`, `heuristic`, `rl`); wird mit in die Abgabe gepackt |
| `ptcg/features.py` | Kodierung einer Entscheidung (Zustand + Optionen) als numpy-Arrays; in der Abgabe |
| `ptcg/policy.py` | Policy-Netz in numpy und der `PolicyAgent`, der damit spielt; in der Abgabe |
| `ptcg/model.py` | Dasselbe Netz in PyTorch (Training), Export der Gewichte nach `.npz`, Batch-Puffer |
| `ptcg/rollout.py` | Partien in Worker-Prozessen: Demonstrationen der Heuristik und Self-Play |
| `ptcg/train.py` | Training: Behavior Cloning (`bc`) und PPO mit Gegner-Pool (`ppo`) |
| `ptcg/make_decks.py` | Erzeugt einfache Themen-Decks pro Energietyp als Trainingsgegner |
| `ptcg/enums.py` | Konstanten der Engine-API ([Doku](https://matsuoinstitute.github.io/cabt/)) |
| `ptcg/evaluate.py` | Gewinnrate zweier Agents mit Konfidenzintervall |
| `ptcg/build_submission.py`, `ptcg/check_submission.py` | Abgabe bauen und testen |
| `submission/main.py` | Einstiegspunkt der Kaggle-Abgabe |
| `decks/` | Decklisten (60 Karten-IDs, eine pro Zeile); `abomasnow.csv` ist unser Deck |

## Wie der Bot spielt

Die Engine fragt den Bot bei jeder Entscheidung nach einer Auswahl: `obs["select"]["option"]` enthält die
legalen Optionen, der Bot gibt eine Liste von Indizes zurück (zwischen `minCount` und `maxCount` viele).
Ganz am Anfang (`obs["select"] is None`) gibt er sein Deck zurück.

### Das RL-Problem

- **Variabler Aktionsraum:** pro Entscheidung 1 bis ~50 legale Optionen (Karte spielen, Energie anlegen,
  angreifen, Ziel wählen, …). Das Netz bewertet deshalb jede Option einzeln, statt einen festen
  Aktionsvektor auszugeben; illegale Aktionen gibt es so gar nicht.
- **Mehrfachauswahl** (z.B. 2 Karten abwerfen) wird in Einzelschritte zerlegt: Das Netz wählt eine Option
  nach der anderen, eine virtuelle STOP-Option beendet die Auswahl, sobald `minCount` erreicht ist.
- **Späte Belohnung:** +1 für einen Sieg, −1 für eine Niederlage, 0 bei Unentschieden, erst am Spielende.
  Pro Partie trifft ein Spieler etwa 50 Entscheidungen.
- **Versteckte Information:** Hand, Deck und Preiskarten des Gegners sind unbekannt.

### Zustand und Netz

`features.py` kodiert eine Entscheidung immer aus Sicht des entscheidenden Spielers:

- globale Merkmale (Zug, verbleibende Preiskarten, Deck-/Handgrößen, Statuszustände, Auswahl-Kontext),
- je ein Merkmalsvektor pro Pokémon-Platz (eigenes/gegnerisches Aktiv-Pokémon und Bank: KP, Schaden,
  angelegte Energie je Typ, ob die Attacken bezahlbar sind, …),
- Kartenlisten (eigene Hand, beide Ablagestapel, eigene noch nicht gesehene Karten = Deck + Preiskarten),
- pro Option: Typ, Bereich, betroffene Karte, betroffener Pokémon-Platz, Attacke.

Jede Karte wird durch statische Merkmale aus der Kartendatenbank (Typ, KP, Stufe, Attacken, Schwäche, …)
plus ein gelerntes Karten-Embedding beschrieben. So kann das Netz auch Karten einordnen, die es im
Training selten gesehen hat. Aufbau (`policy.py`): Karten-MLP → Platz-MLP → Zustands-MLP (256) und
Options-MLP (128); der Policy-Kopf berechnet aus Zustand und Option einen Score pro Option (Softmax
darüber = Policy), der Value-Kopf schätzt die Gewinnchance in [−1, 1]. ~0,67 Mio. Parameter.

### Training

1. **Behavior Cloning:** Das Netz imitiert zuerst die Heuristik (Kreuzentropie auf deren Züge, der
   Value-Kopf lernt den Spielausgang). Das gibt PPO einen sinnvollen Startpunkt, denn mit zufälligen
   Zügen und nur einer Belohnung am Spielende lernt man sehr langsam.
2. **PPO mit Self-Play:** In jeder Iteration spielen 10 Worker-Prozesse ~160 Partien mit der aktuellen
   (stochastischen) Policy. Gegner sind zu 40 % die aktuelle Policy selbst, zu 30 % frühere Snapshots,
   zu 25 % die Heuristik und zu 5 % der Zufallsagent (`--mix`). Der Gegner-Pool verhindert, dass die
   Policy nur gegen sich selbst gut wird. Unser Agent spielt `abomasnow.csv`, die Gegner zufällige Decks
   aus `decks/`. Vorteile mit GAE (γ = 1, λ = 0,95), geclippter PPO-Verlust, Entropie-Bonus.
3. **Evaluation:** Alle 10 Iterationen spielt die greedy Policy 200 Spiegelmatches (beide `abomasnow.csv`)
   gegen die Heuristik. 50 % bedeutet also „so gut wie die Heuristik“. Das beste Modell wird als
   `best.npz` gespeichert und am Ende nach `model.npz` kopiert.

### Abgabe

Die Abgabe enthält nur numpy-Code: `model.py` exportiert die Gewichte als `.npz`, `policy.py` rechnet
denselben Forward-Pass in numpy (beim Export wird geprüft, dass beide übereinstimmen). Eine Entscheidung
dauert ~1 ms, die 600 s Zeitreserve pro Match sind also kein Problem. Gespielt wird greedy (beste Option).

### Nicht umgesetzt: Suche

Die Engine-Bibliothek exportiert Funktionen für eine Vorwärtssuche (`SearchBegin`, `SearchStep`, …), und
die Beobachtung enthält `search_begin_input`. Der dokumentierte Python-Wrapper dafür (`api.py`) ist aber
nicht in `kaggle-environments` 1.33.0 enthalten, und die C-Signaturen sind nicht dokumentiert. Eine
Suche (z.B. MCTS mit dem Value-Netz) wäre der nächste Schritt, sobald der Wrapper verfügbar ist.

## Stand

| Agent | Gegner | Gewinnrate |
| --- | --- | --- |
| `heuristic` | `random` | 89 % ± 3 % (400 Partien, gleiches Deck) |
