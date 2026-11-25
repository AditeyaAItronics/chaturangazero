# ChaturangaZero Web UI: Guide for the Team

A browser app to play against the trained AlphaZero agent, watch models play each other, and see what the agent is "thinking". It is built for the demo video and for exploring results for the report.

- **Backend:** `web/app.py` (Flask)
- **Frontend:** `web/static/` (`index.html`, `app.js`, `pieces.js`, `style.css`)
- **Tests:** `tests/test_web_features.py`, plus web tests in `tests/test_second_pass.py` and `tests/test_low_and_speed.py`

---

## 1. Run it (5 minutes)

### Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
pip install -r requirements.txt                                      # includes flask
```

### Put the models where the app looks

The app reads checkpoints from **`runs/chaturanga/`** in the repo. `runs/` is git-ignored, so **the models are not in GitHub**: every collaborator downloads them once from Hugging Face. One command does it, and checks every file's checksum:

```bash
python tools/fetch_models.py
```

It puts these in `runs/chaturanga/` (run it again any time; it only replaces missing or outdated files):

| File | What it becomes in the app | Source on Hugging Face |
|---|---|---|
| `best.pt` | **AlphaZero · best model**, iteration 1,350 (the default opponent) | `run-1500/best.pt` |
| `metrics.json` | Fills the **model card**: 1,500 iterations, games, promotions, hours | `run-1500/metrics.json` |
| `checkpoint_1499.pt` | **AlphaZero · latest checkpoint**, iteration 1,500 | `run-1500/checkpoint_1499.pt` |
| `checkpoint_0869.pt` | **AlphaZero · iteration 870**, the previous best, for old-vs-new games | `run-1000/best.pt` |

Any other `checkpoint_NNNN.pt` you add (for example `checkpoint_0499.pt` at the repo root of Hugging Face, or one from Kaggle `final/` or `run/`) becomes one more opponent, **AlphaZero · iteration N+1**.

> Don't use `run-1000/metrics.json`: it only covers the first 1,000 iterations, so the model card would say "1,000 iterations".

Checkpoint filenames are zero-based, the app is one-based: `checkpoint_0999.pt` is shown as **iteration 1000**.

With no `best.pt`, the AlphaZero players are labelled "no checkpoint: plays random moves" instead of silently playing randomly.

### Start the server

```bash
python web/app.py              # http://127.0.0.1:5000
python web/app.py --port 5001  # another port
python web/app.py --device cuda
```

Open **http://127.0.0.1:5000** in a browser. Search runs on the CPU by default and is fast: about 0.1 s per 100 simulations.

> **One shared game:** the server holds a single game. Two browser tabs on the same server show and change the same game. To test without disturbing someone's game, start a second server on another port.

---

## 2. What is on the screen

```
┌ Header: title · Flip View · New Game ────────────────────────────┐
│ Black player bar                          │ Player Matchup         │
│ ┌──────────── board ─────────────┐        │  White / Black pickers │
│ │                                │        │  Pause / Resume AI     │
│ └────────────────────────────────┘        │  Undo                  │
│ White player bar                          │  Pause before AI moves │
│ ┌ Momentum (win-chance chart) ──────────┐ │  Search strength       │
│ ┌ History ───────┐ ┌ What the AI is ───┐ │ The AlphaZero model    │
│ │ moves, review, │ │ thinking          │ │ How this variant works │
│ │ analysis       │ │                   │ │                        │
│ └────────────────┘ └───────────────────┘ │                        │
└─ footer: Chaturanga · Pipeline check: Connect-4 ──────────────────┘
```

On a narrow window the sidebar moves below the board.

---

## 3. Features and how to use them

### Play a game
- **Move:** click one of your pieces (its legal moves are marked), then click the destination.
- **Who plays what:** in **Player Matchup**, pick **White** and **Black**: Human, an AlphaZero model, or a baseline (alpha-beta 3-ply, greedy material, random). Changing either starts a new game.
- **Play as Black:** set White to an AI and Black to Human. The board flips so Black is at the bottom, and the AI opens. **Flip View** still works by hand.
- **Undo:** in human-vs-AI it takes back your move and the AI's reply.
- **Highlights:** gold squares show the last move. A **pulsing red frame** marks a Raja that can be captured right now. There is no check in this variant, so this is the only warning a player gets.
- **Player bars** show each side's material (Ratha 5, Ashva 3, Gaja 1.5, Mantri 1.5, Padati 1; Raja not counted) and captured pieces.

### Watch AI against AI
- Set both players to AIs. The game plays itself, and **⏸ Pause AI vs AI** / **▶ Resume** controls it.
- **Pause before AI moves** (0.1–2.5 s) sets the pace. Use about 1–1.5 s when recording, so viewers can follow.
- **Old vs new:** put an early checkpoint (for example iteration 500) against the best model to show what training achieved. Results vary game to game, so don't present one game as proof.

### Search strength
**AlphaZero search strength** sets the Monte Carlo tree search simulations per AlphaZero move, from **10** to **800** (default 200; training used 40). Lower it to make the AI beatable, raise it to make it stronger and slower. The server enforces the 10–800 range.

### What the AI is thinking (after each AlphaZero move)
- **Evaluation bar:** the search's **expected score**, from White's point of view (win = 100%, draw = 50%).
- **Instinct vs search** (the core AlphaZero idea, live): the panel says whether the 200 simulations **confirmed** the network's first instinct (its raw move probability) or **overruled** it. The overruled case is highlighted.
- **Top moves:** the three moves the search spent most time on, each with two thin bars: the **network's instinct** (grey) and the share **after search** (gold). Hover a row for the exact numbers.

### Momentum chart
- **What it shows:** White's expected score across the whole game, gold above the 50% line when White is better and rose below it when Black is.
- **Source of the values:** during a game, they come from each AlphaZero move's search; human and baseline moves have none. After **Analyse**, every position gets a value from one model.
- **Hover:** a point shows the move and both sides' percentages. **After the game, click** a point to jump to that position.
- **In AI vs AI games** a zig-zag is expected: each model's search judges the position differently.

### Review a finished game
- **How to start:** on the game-over card, click **Review game**, or click any move in **History**.
- **Stepping:** use **⏮ ◀ ▶ ⏭** or the **← →** keys. The reviewed move is highlighted in History, and a cursor appears on the momentum chart.
- **Game-over card:** says who won, how (Raja captured, bare Raja, stalemate, 300-ply cap), and after how many moves.

### Analyse: blunders and brilliant moves
- **How to start:** after a game, click **🔍 Analyse** in History, or **Analyse moves** on the game-over card.
- **What it does:** the **best model** searches every position (200 simulations each, about 0.7 s per move) and labels each move:

| Label | Meaning |
|---|---|
| **??** Blunder | Lost **20+ points** of expected score, or one of two exact checks: **missed a Raja capture** that won on the spot, or **left the Raja capturable** |
| **?** Mistake | Lost 10–20 points |
| **?!** Inaccuracy | Lost 5–10 points |
| **!** Good | The only strong move; every serious alternative was 10+ points worse |
| **!!** Brilliant | The search's best move, which the network gave **under 5%** on instinct, while its own favourite was 10+ points worse |

- **Where results show:** hover a label for the reason and a better move. Each side gets a tally, and the momentum chart marks blunders (red) and brilliant moves (teal).
- **Safeguards:**
  - the search's own best move is never labelled a mistake (protects against search noise);
  - nothing gets praise once the game is already decided.
- **Be honest about it:** the judge is our own model, so a label means "a blunder **according to the best model**", not objective truth.

### Model card and rules
- **The AlphaZero model:** iteration, network size (ResNet, 5 blocks × 64 channels, 514,449 parameters), gradient steps, and from `metrics.json` the training iterations, self-play games, promotions and hours.
- **How this variant works:** no check, Raja capture wins, stalemate wins, bare Raja loses, the Gaja jump, Mantri/Padati moves, the 300-ply cap.

### Connect-4
The footer link **Pipeline check: Connect-4** switches to the Phase 0 pipeline test. It needs `runs/connect4/final.pt` or `runs/connect4/phase0_passed.pt`.

---

## 4. Demo-video recipes

| Shot | Setup |
|---|---|
| **Play against the AI** | White = Human, Black = best model; strength 200; pause about 1 s. |
| **Old vs new** | White = iteration 500, Black = best model; pause 1–1.5 s; record until the game ends, then **Analyse moves** and click the turning point on the momentum chart. |
| **Search beats instinct** | Human vs best model; play a few moves and watch the panel for **"Search overruled the network"**. |
| **Strength matters** | Same opening against the AI at strength 10, then at 400–800; the top-move bars sharpen. |
| **Baselines** | Best model vs **Baseline · alpha-beta (3 ply)**, AI vs AI. |

**Tips:** make the browser window wide (at least 1366 × 768) so the sidebar sits next to the board. Zoom to 90% (Ctrl + −) to fit everything without scrolling. Start each take with **New Game**.

---

## 5. For developers

### HTTP API (all JSON)

| Method & path | Does |
|---|---|
| `GET /api/models` | Player list for both games (AlphaZero entries include the checkpoint iteration) |
| `GET /api/model_info?id=az_best` | Model card data |
| `GET /api/state` | Current game: board, legal moves, `last_move`, `move_history`, `evals` (White's expected score per move) |
| `POST /api/new_game` | `{game, p1_type, p2_type}` |
| `POST /api/make_move` | `{action}`: human move |
| `POST /api/ai_move` | `{sims}` (optional, 10–800): the AI to move plays; returns `ai_stats` (value, top moves with `prior` and `pct`, `instinct`, `search_agrees`) |
| `POST /api/undo` | Undo |
| `GET /api/positions` | Board after every ply (for review) |
| `POST /api/analyze_move` | `{ply, sims?, model?}`: label move `ply` (0-based) of the current game |

**Player ids:** `human`, `az_best`, `az_latest`, `ckpt_NNNN` (a specific checkpoint; digits only, so it can't point outside `runs/chaturanga/`), `baseline_alphabeta`, `baseline_greedy`, `random`.

### Rules that apply here too
- `RULES.md` is frozen. The capturable-Raja highlight and the analysis use `legal_moves()` only; there is no `is_square_attacked()` in the engine and none should be added.
- Material adjudication is training-only: a game in the app that reaches 300 plies is a **draw**.
- **Process (from `AGENTS.md`):** run `python -m pytest -q` after each change, review it, and work on a feature branch. Push only when the owner says so.

### Tuning the analysis
The thresholds are constants near the top of `web/app.py` (`BLUNDER`, `MISTAKE`, `INACCURACY`, `BRILLIANT_MAX_PRIOR`, `BRILLIANT_MIN_GAIN`, `ONLY_MOVE_GAP`, `DECIDED`, `ANALYSIS_SIMS`). `tests/test_web_features.py` checks each rule on hand-built positions; update it with any change.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| "no checkpoint: plays random moves" | Run `python tools/fetch_models.py`, then reload the page. |
| Model card says "no model" or shows few stats | `best.pt` missing, or `metrics.json` not beside it: run `python tools/fetch_models.py`. |
| Model card says **1,000 iterations** or best model **iteration 870** | Old files from `run-1000/`: run `python tools/fetch_models.py`, then restart the app. |
| An old checkpoint isn't in the list | The name must be `checkpoint_NNNN.pt` in `runs/chaturanga/`; reload the page. |
| The page doesn't show a new feature | Restart `web/app.py` (backend changes need a restart), then hard-reload (Ctrl + Shift + R). |
| My game changed by itself | Someone else uses the same server; run your own on another `--port`. |
| `ModuleNotFoundError: flask` | `pip install -r requirements.txt` |
