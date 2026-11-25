# ChaturangaZero

**An AlphaZero agent that taught itself Chaturanga, the 1,500-year-old Indian ancestor of chess, purely by playing against itself.** No human games, no opening book, no hand-written evaluation.

This is the historical ruleset with **no check**: any move is legal, even leaving your Raja (king) attacked, and the game ends only when a Raja is actually captured. A stalemated player *wins*.

*TeamRandom: Aditeya, Ansh, Adit, Sarthak*

| | |
|---|---|
| 🤖 **Trained models** | [huggingface.co/ansh-k03/chaturangazero-500](https://huggingface.co/ansh-k03/chaturangazero-500) (`run-1000/`, `run-1500/`) |
| 📊 **Results** | [`results/`](results/README.md): plots, evaluation matches, how to reproduce them |
| 🎮 **Play it in your browser** | [`WEB_UI_GUIDE.md`](WEB_UI_GUIDE.md) |
| 📜 **Rules (frozen)** | [`RULES.md`](RULES.md) |
| 📝 **Report** | [`REPORT.md`](REPORT.md) · plan for the final PDF: [`reportplan.md`](reportplan.md) |
| 🏗️ **How it works** | [`ARCHITECTURE_AND_TRAINING.md`](ARCHITECTURE_AND_TRAINING.md) · [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) |

---

## At a glance

- **1,500 training iterations:** 48,000 self-play games, 212,309 gradient steps, about 60 hours (Azure CPU, then a Kaggle GPU).
- **Network:** a 5-block, 64-channel ResNet with policy and value heads (514,449 parameters), guided by Monte Carlo tree search.
- **Engine verified:** matches an independently written reference engine exactly, to 6 moves deep on 15 test positions (19,932,596 positions from the start alone).
- **Strength:** beats random and greedy players **40–0**; against a 3-ply alpha-beta search it scores only **7–16%**, even with 20× more search (40 → 800 simulations per move).

## Interesting findings

1. **Stalemate never happened.** In 48,000 self-play games, not one ended in stalemate, even though a stalemated player *wins*. To stalemate yourself, your own pieces have to block your Raja in, and the agent never found that worthwhile. Games ended by Raja capture (49%), the 300-move cap (41%) or a bare Raja (10%).
2. **A one-line bug froze learning for 840 iterations.** The learning-rate schedule was written for a 200-iteration run, so from iteration 160 the network learned at 1/100th of its starting rate, and every restart restored that setting whatever the config said. After the fix, the value network's accuracy at predicting the winner of decisive games jumped from **72% to 85%**. Part of that jump comes from a scoring change made at the same moment, which we had to untangle.
3. **The agent's own quality check fooled itself.** A new model replaces the old one only if it wins a 40-game match with statistical significance (p < 0.05). That happened twice in 500 iterations. But a 200-game rematch showed the first "winner", iteration 1,270, was actually **weaker** than the model it replaced (46.5%, p = 0.005 the other way). Run 50 tests at a 5% false-alarm rate and you should *expect* about 2.5 false winners, and we got 2. The lesson is general: **always confirm an improvement with a larger, independent test.**
4. **Most games between strong versions are draws.** 169 of 200 games between iterations 1,350 and 870 were drawn, so a 40-game match can barely tell two good models apart.
5. **No check makes the game brutally tactical.** On its non-random moves, the agent skipped an available winning Raja capture 12–14% of the time; its search ranked some other move higher. Our hypothesis: the search treats unexplored moves as draws, so when it's already winning, an untried capture looks *worse* than its other options, and 40 simulations may never try it. A 3-ply alpha-beta search never misses such a capture. **But more search didn't save the agent:** from 40 to 800 simulations, its score against alpha-beta only went from 7.5% to about 11–16% (some losses became draws). So missed captures aren't the whole story: an engine that checks *every* line three moves deep beats a selective search whose network judges sharp tactics imperfectly.
6. **Search beats instinct, live.** In the web app you can watch the network's first instinct (its raw move probabilities) next to where 200 tree-search simulations actually went, and see the search confirm or overrule the network. That is AlphaZero's core idea, made visible.

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
pip install -r requirements.txt
```

**Play against the agent:** the trained models live on Hugging Face, not in this repository. Download them once (into `runs/chaturanga/`, with checksums verified), then start the app:

```bash
python tools/fetch_models.py   # best model (it. 1,350), 1,500-iteration metrics, it. 870 and 1,500
python web/app.py              # open http://127.0.0.1:5000
```

The web app lets you play either side, watch AI vs AI or old vs new models, set the search strength, see the momentum chart and the AI's reasoning, and analyse a finished game for blunders and brilliant moves. See [`WEB_UI_GUIDE.md`](WEB_UI_GUIDE.md).

**Train:**

```bash
python train_chaturanga.py --config configs/local.yaml --output runs/chaturanga
```

Kaggle GPU continuation, resumable across sessions: [`notebooks/chaturangazero_kaggle.ipynb`](notebooks/chaturangazero_kaggle.ipynb) with [`configs/kaggle.yaml`](configs/kaggle.yaml).

**Evaluate:**

```bash
python eval/evaluate.py match runs/chaturanga/best.pt runs/chaturanga/checkpoint_0869.pt --games 200 --sims 40
python eval/evaluate.py baselines runs/chaturanga/best.pt --games 40 --sims 40
python eval/evaluate.py plots runs/chaturanga
```

**Tests:** `python -m pytest -q` (196 test functions, 266 cases).

## Repository layout

| Folder | Contents |
|---|---|
| `chaturanga/` | Rules engine, independent reference engine, perft, encoding |
| `net/` | ResNet, training loop, compact replay buffer |
| `search/` | MCTS, arena with paired openings and the significance gate, baselines |
| `pipeline/` | Self-play (batched, multi-process), training loop, resume, cloud sync |
| `eval/` | Evaluation: matches, baselines, Elo ladder, plots, sample games |
| `web/` | Flask web app |
| `connect4/` | Connect-4, used to prove the pipeline before Chaturanga |
| `results/` | Plots and evaluation results of the 1,500-iteration run |
