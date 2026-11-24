# Report plan: ChaturangaZero (TeamRandom)

Everything the final report needs, in the order it will appear, with every number sourced. A later step converts this to the course's LaTeX template.

## Submission rules (from the course)

- [ ] **Team name:** TeamRandom
- [ ] **PDF** built from the **course LaTeX template, unmodified**: no changes to font size, margins, spacing or columns
- [ ] **At most 4 double-column pages, references included** (the update raised the limit from 2)
- [ ] Sections on **individual member contributions** and **generative-AI disclosure**
- [ ] **Link to the GitHub repository:** https://github.com/AditeyaAItronics/chaturangazero
- [x] **Repository is public** (checked on 24 Nov 2026)

## Title and authors

**ChaturangaZero: AlphaZero Self-Play for No-Check Chaturanga**
TeamRandom: Aditeya, Ansh, Adit, Sarthak
Code: https://github.com/AditeyaAItronics/chaturangazero · Models: https://huggingface.co/ansh-k03/chaturangazero-500

## Page budget (4 pages, double column, references included)

| Section | Space |
|---|---|
| Abstract + 1 Introduction | 0.4 page |
| 2 Game and rules | 0.35 |
| 3 Method (engine, network, search, training loop) | 0.9 |
| 4 Experiments and results (2 figures, 2 tables) | 1.3 |
| 5 Discussion and future work | 0.45 |
| 6 Web demo | 0.15 |
| 7 Contributions + 8 Generative-AI disclosure | 0.3 |
| References | 0.15 |

If over the limit, cut in this order: the web demo paragraph, then one figure, then the perft table (keep one sentence).

---

## Abstract (about 120 words)

We train an AlphaZero-style agent, from self-play alone, for a historical Chaturanga ruleset with **no check**: the game ends only when a Raja is captured, and a stalemated player wins. The engine is verified against an independent reference engine by perft to depth 6. A 5-block ResNet (514k parameters) with MCTS (40 simulations per move) was trained for **1,500 iterations (48,000 self-play games, 60 GPU/CPU hours)**. A learning-rate bug froze learning for 840 iterations; fixing it revived the value head, but 200-game confirmation matches show the extra 500 iterations brought **no measurable strength gain**, and that one arena promotion was a **false positive** caused by repeated testing. The agent beats random and greedy players 100% of the time but loses to a 3-ply alpha-beta search.

## 1 Introduction (0.4 page)

- **Motivation:** AlphaZero learns games from rules alone. Chaturanga, the Indian ancestor of chess, is a small, well-defined but under-studied test bed.
- **Why the no-check rule matters:** every pseudo-legal move is legal, so there is no king-safety filtering, and the game is decided by tactics (Raja captures) rather than mating patterns.
- **Contributions:**
  1. a rules engine verified to perft depth 6 against an independent engine;
  2. a complete AlphaZero pipeline (batched self-play, gated training, resumable cloud runs);
  3. a 1,500-iteration training study with an honest evaluation: a learning-rate bug, false promotions from repeated gating, and a tactical weakness against shallow search;
  4. an interactive web demo that shows the agent's search, momentum and move-quality analysis.

## 2 Game and rules (0.35 page)

Source: `RULES.md` (frozen, signed off by all four members).

- 8×8 board, White moves first. **Ratha** (rook-like), **Ashva** (knight), **Gaja** (jumps exactly two squares diagonally, over anything), **Mantri** (one square diagonally), **Raja** (king, one square any direction), **Padati** (pawn: one step, no double step, no en passant, always promotes to Mantri).
- **No check, no checkmate.** A Raja may move onto an attacked square; the game ends only when a Raja is **actually captured**.
- **Terminal conditions, checked in this order:** Raja captured → loss; King vs King → draw; bare Raja → loss; **stalemate → win for the stalemated player**; move cap.
- **No repetition or 50-move rule:** a **300-ply cap** is the only termination guarantee. Capped **training** games are adjudicated on material (Ratha 5, Ashva 3, Gaja 1.5, Mantri 1.5, Padati 1) with a margin of 4.0, later 2.0. **Evaluation** games at the cap are draws.
- **Two consequences:** stalemate needs material, so it can't be an escape hatch; King vs King can't arise from play (a bare Raja loses first).

## 3 Method (0.9 page)

**3.1 Engine and verification.**
- The production engine is checked against an **independent reference engine** (its own board representation and move generation): exact perft agreement **to depth 6 on 15 positions**, for example 19,932,596 nodes from the start position.
- Property tests: action encoding round-trips, legal-mask counts, mirror symmetry, 10,000 random playouts terminate, the terminal ladder and its order.
- The test suite has 196 test functions (266 cases with parameters).

**3.2 Representation.**
- **Input:** 14 planes of 8×8: 6 own piece types, 6 opponent piece types, ply/300, all-ones. Always from the side to move's view (the board is flipped for Black), with left-right mirror augmentation.
- **Actions:** 4,096 = from-square × to-square; illegal moves are masked to −∞ before the softmax.

**3.3 Network.**
- Dual-head ResNet: **5 residual blocks × 64 channels, 514,449 parameters**.
- Policy head: a 1×1 convolution to 64 planes, read as 4,096 logits, avoiding a 34M-parameter linear layer.
- Value head: 1×1 convolution → fully connected → tanh.
- Loss (z − v)² − πᵀ log p; AdamW (weight decay 0.01, decoupled), batch 128, gradient clipping 1.0.

**3.4 Search.**
- PUCT MCTS with c = 1.5; **40 simulations per move** in self-play and the arena.
- Dirichlet root noise α = 0.3, ε = 0.25; temperature 1 for 20 plies, then the most-visited move; subtree reuse; evaluation cache; virtual loss for batched leaves.
- Unvisited moves score as a draw (no first-play-urgency reduction); see §5.

**3.5 Training loop.**
- 32 self-play games per iteration, batched across parallel games on the GPU.
- Replay buffer of 100,000 positions, sampled uniformly, mirrored at sample time; replay ratio 3, so about 3 samples per new position.
- **Arena gate** every 10 iterations: 40 games from paired random 4-ply openings. A challenger replaces the incumbent only if its score is ≥ 55% **and** a one-sided paired sign-flip test gives p < 0.05. Self-play always uses the incumbent.
- Atomic checkpoints and an iteration commit marker make runs resumable after preemption.

## 4 Experiments and results (1.3 pages)

**Table 1: training phases** (source: `results/README.md`, `metrics.json`)

| Iterations | Hardware | LR | Margin | Hours | Promotions (of 50 arenas) |
|---|---|---|---|---|---|
| 1–500 | Azure CPU VM | 2e-3 → 2e-4 (it. 80) → 2e-5 (it. 160) | 4.0 | 23.9 | 11 |
| 501–1,000 | Azure CPU VM | 2e-5 (stuck) | 4.0 | 25.8 | 2 |
| 1,001–1,500 | Kaggle GPU | 2e-4 (reset) | 2.0 | 10.6 | 2 |
| **Total** | | | | **60.3** | **15 of 150** |

Total: 48,000 self-play games, 212,309 gradient steps.

**4.1 The learning-rate bug.**
- `lr_milestones: [80, 160]` were written for a 200-iteration run, so the LR was **2e-5 from iteration 160**. Each resume restored the saved scheduler, overriding the config, which froze learning for **840 iterations** (promotions fell from 11 to 2 per 500 iterations).
- **Fix:** `reset_lr_on_resume`, with tests through the real training loop.

**4.2 Effect on training** (Figure 1: `results/plots/value_head.png`; optional Figure 2: `results/plots/cap_and_draws.png`)
- Value-head sign accuracy on decisive targets: **about 0.72 → about 0.85** at iteration 1,001; mean |v| about 0.2 → about 0.43; policy loss about 2.42 → about 2.38.
- **Caveat to state:** the jump coincides with the margin change 4.0 → 2.0 (which redefines the decisive targets) and an empty replay buffer, so it is not by itself evidence of stronger play.
- After iteration 1,000 the draw rate falls (about 25%) while the cap rate drifts from 46% to 40%: the 2.0 margin decides most capped games on material.

**Table 2: playing strength** (40 simulations, paired random openings, no adjudication; sources `results/*.json`)

| Match | Result | Score (95% CI) | Paired p |
|---|---|---|---|
| it. 1,350 vs 870, 200 games | 15W-169D-16L | 49.8% (42.9–56.6%) | 0.63 |
| it. 1,270 vs 870, 200 games | 7W-172D-21L | 46.5% (39.7–53.4%) | **0.005 that 870 is better** |
| it. 1,350 vs random | 40W-0D-0L | 100% | |
| it. 1,350 vs greedy material | 40W-0D-0L | 100% | |
| it. 1,350 vs **3-ply alpha-beta** | 2W-2D-36L | **7.5%** | |

**4.3 False promotions.**
- The gate promoted 1,270 over 870 (7W-32D-1L, p = 0.031), then 1,350 over 1,270 (8W-30D-2L, p = 0.035).
- 200-game matches show **870 ≈ 1,350 > 1,270**: 1,270's promotion was a false positive, and 1,350 only recovered 870's level.
- With **50 arena tests at α = 0.05, about 2.5 false promotions are expected** even with no real improvement.
- About 85% of games between the networks are drawn (169–172 of 200), so 40-game matches have very little power.

**4.4 Other observations.**
- **Stalemate never occurred** in 48,000 self-play games; endings were Raja capture 49%, move cap 41%, bare Raja 10% (`terminal_reasons.png`).
- **Missed Raja captures:** on non-sampled moves, an available winning Raja capture was not played **12–14%** of the time (1–1,000: 1,414 of 10,025; 1,001–1,500: 704 of 5,841), and in almost all of these the search ranked another move higher.

## 5 Discussion and future work (0.45 page)

- **Why the agent loses to 3-ply alpha-beta (hypothesis):**
  - alpha-beta never misses a short tactic, and this variant is decided by tactics;
  - our MCTS scores unvisited moves as a **draw**, so when the side to move is winning, its other moves look better than an untried capture, and with 40 simulations the capture may never be tried;
  - this fits the 12–14% missed captures.
  - **Test:** first-play urgency (`fpu_reduction`), more simulations (the strength-vs-simulations experiment), or always checking Raja captures before searching.
- **Gating under repeated testing:** confirm promotions with a long match, or use sequential testing (SPRT) with a fixed overall error rate.
- **Shuffling to the move cap:** about 40% of training games hit the cap. Options: a shorter cap (200), more root noise, or a small **draw penalty** (about −0.1 for both sides, training-only), which breaks zero-sum scoring and needs ablations.
- **Capacity:** 514k parameters and 40 simulations; the planned 8 × 128 network was never trained.
- **Lessons:** confirm every claimed improvement with a large, independent test; read the training schedule and the restored optimizer state when resuming; separate metric changes caused by **redefined targets** from real learning.

## 6 Web demo (0.15 page)

Flask app (`web/app.py`; guide in `WEB_UI_GUIDE.md`):
- play either side, or watch AI vs AI or old vs new checkpoints, with an adjustable search strength (10–800 simulations);
- a live **instinct vs search** view (the network's prior against MCTS visit shares);
- a **momentum** chart of White's expected score;
- **post-game analysis** labelling blunders and brilliant moves by the model's own search;
- step-by-step game review.

Used for the demo video.

## 7 Individual contributions

| Member | Contributions |
|---|---|
| **Aditeya** | Web UI fixes and features; code bug fixes (including the learning-rate reset and porting the promotion gate); training iterations 1,000–1,500 on Kaggle. |
| **Ansh** | Trained the model for iterations 0–1,000 (Azure); helped write `RULES.md` and the implementation plan. |
| **Sarthak** | Presenter; wrote the initial code and the UI skeleton. |
| **Adit** | Wrote the report; helped with the implementation plan. |

## 8 Generative-AI disclosure (draft: **each member must confirm or extend it**)

> We used Anthropic's Claude (via Claude Code) as a coding and writing assistant. **Aditeya** directed a review of the training pipeline, in which the AI identified the frozen learning-rate schedule. Aditeya confirmed the diagnosis, decided on the fix and had the AI implement it with tests. At Aditeya's direction, the AI also ported the promotion gate into the repository, wrote the Kaggle training notebook (which Aditeya configured and ran for iterations 1,000–1,500), and implemented the web-UI features (piece set, layout, model card, instinct-vs-search view, momentum chart, game review and move analysis) with tests. It also ran the evaluation matches, produced the plots and helped draft documentation and this report. All AI-generated code was reviewed, tested (196 test functions) and run by team members, and all reported numbers come from our own training runs and evaluation files. *[Add any other AI use by any member, for example in earlier code, the slides or the rules.]*

## References (keep short)

1. D. Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play," *Science* 362, 2018.
2. D. Silver et al., "Mastering the game of Go without human knowledge," *Nature* 550, 2017.
3. C. D. Rosin, "Multi-armed bandits with episode context," *Annals of Mathematics and AI* 61, 2011 (PUCT).
4. K. He et al., "Deep residual learning for image recognition," CVPR 2016.
5. I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," ICLR 2019 (AdamW).
6. Optional, for Chaturanga history: H. J. R. Murray, *A History of Chess*, Oxford, 1913.

*Check each reference against the original before submission.*

---

## Still to do before writing the LaTeX

- [ ] Team members confirm the contribution lines and the AI disclosure
- [ ] Optional experiment: AlphaZero vs 3-ply alpha-beta at 200 and 800 simulations (tests the §5 hypothesis; about 30 minutes)
- [ ] Optional experiment: the same with `fpu_reduction` enabled
- [ ] Pick 1–2 figures (recommended: `value_head.png`; then `cap_and_draws.png`)
- [ ] Get the course LaTeX template from the Google Drive link
