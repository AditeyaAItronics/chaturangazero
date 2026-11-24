# Results: the 1,500-iteration run

All numbers come from `runs/chaturanga/metrics.json` (iterations 1–1,500) and the evaluation files in this folder. Models and full metrics are on Hugging Face (`ansh-k03/chaturangazero-500`, folders `run-1000/` and `run-1500/`).

## The run

| Iterations | Where | Learning rate | Adjudication margin | Best model at the end |
|---|---|---|---|---|
| 1–500 | Azure CPU | 2e-3 → 2e-4 (it. 80) → 2e-5 (it. 160) | 4.0 | 500 |
| 501–1,000 | Azure CPU | 2e-5 (stuck: the saved schedule was restored on resume) | 4.0 | 870 |
| 1,001–1,500 | Kaggle GPU | **2e-4** (`reset_lr_on_resume`) | **2.0** | 1,350 |

All other settings are unchanged throughout: 5 × 64 ResNet (514,449 parameters), 40 MCTS simulations per move, 32 self-play games per iteration, move cap 300, root noise 0.25, and an arena gate of 40 games needing score ≥ 55% and paired p < 0.05.

## Playing strength

| Match (40 sims, paired openings, no adjudication) | Result | Score | 95% CI | Files |
|---|---|---|---|---|
| **Iteration 1,350 vs 870**, 200 games | 15W-169D-16L | **49.8%** | 42.9%–56.6% | `match_1350_vs_870.json` |
| Iteration 1,350 vs random, 40 games | 40W-0D-0L | 100% | | `baselines_1350.json` |
| Iteration 1,350 vs greedy material, 40 games | 40W-0D-0L | 100% | | `baselines_1350.json` |
| **Iteration 1,350 vs 3-ply alpha-beta**, 40 games | 2W-2D-36L | **7.5%** | | `baselines_1350.json` |

**Findings:**
1. **The continuation did not measurably improve playing strength.** Iteration 1,350 passed the arena gate twice (vs 870: 7W-32D-1L, p = 0.031; vs 1,270: 8W-30D-2L, p = 0.035). But 200 games against 870 give 49.8% (p = 0.63). With 50 arena tests at α = 0.05, about 2.5 false promotions are expected by chance alone, so a per-comparison gate needs a confirmation match or a multiple-testing correction.
2. **The agent beats the weak baselines completely but loses heavily to 3-ply alpha-beta** at its training search budget (40 simulations).
3. **Most games between the two networks are drawn** (169 of 200), so small strength differences need many games to detect.

## Training dynamics (plots in `plots/`)

| Plot | What it shows |
|---|---|
| `loss.png` | Policy loss plateaus at about 2.4 from iteration 200; after the LR fix it eases from about 2.42 to about 2.38. Value loss is about 0.3–0.4. |
| `value_head.png` | Sign accuracy steps up from about 0.72 to about 0.85 at iteration 1,001, and mean \|v\| from about 0.2 to about 0.43. **Caution:** that step coincides with the margin change 4.0 → 2.0, which redefines the decisive targets, and with an empty replay buffer, so it is not only learning. |
| `cap_and_draws.png` | Before 1,000, draws track the move-cap rate. After it, the 2.0 margin decides most capped games on material: draws fall to about 25% while the cap rate drifts from 46% to 40%. |
| `game_length.png`, `terminal_reasons.png` | Games average about 200 plies; most end by Raja capture or the cap, some by bare Raja; **stalemate never occurred** in 1,500 iterations. |
| `missed_raja.png` | How often an available Raja capture was not played. |
| `throughput.png` | Self-play speed: about 3.3 min per iteration on Azure CPU, 1.26–1.28 on the Kaggle GPU. |

## Reproduce

```bash
python eval/evaluate.py match runs/chaturanga/best.pt runs/chaturanga/checkpoint_0869.pt --games 200 --sims 40
python eval/evaluate.py baselines runs/chaturanga/best.pt --games 40 --sims 40
python eval/evaluate.py plots runs/chaturanga
```

(`checkpoint_0869.pt` is the earlier `best.pt` of `run-1000/`, iteration 870.)
