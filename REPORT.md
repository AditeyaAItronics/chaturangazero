# ChaturangaZero: Final Report

*Self-play AlphaZero agent for Chaturanga. Team of 4: Aditeya, Ansh, Adit, Sarthak.*

> **Status of this document.** Sections marked ✅ are written from work that is done and measured. Sections marked ⏳ need the main training run. Each ⏳ item names the command that produces its numbers and figures. Per the plan (Part 4, "Shared work"), each student writes their own section and Student 4 assembles the report.

---

## 1. Rules, and the no-check design ✅ *(Student 1)*

- **Ruleset:** frozen in `RULES.md`, with sign-off table. It is the historically authentic rules, not chess-aligned:
  - no castling;
  - no pawn double-step or en passant;
  - forced promotion to Mantri;
  - **no check**: the game ends when a Raja is actually captured, and a Raja may step onto a guarded square;
  - bare king loses, except King-vs-King, which is a draw;
  - a stalemated player **wins** (z = +1);
  - no repetition or 50-move rule.
- **Why no-check simplifies the engine:** every pseudo-legal move is legal, so move generation needs no king-safety filter, pin tracking or check evasion. There is deliberately no `is_square_attacked()` anywhere. The state is fully Markov (board + side + ply), so the input is 14 planes with no history.
- **Termination:** with no repetition or 50-move rule, the 300-ply cap is the only termination guarantee. It is enforced as a hard assertion. Training games at the cap are adjudicated on material (margin 4); evaluation games at the cap are draws.
- **Two consequences worth reporting:**
  - **Stalemate needs material.** It requires blocking in your own Raja with your own pieces, so a player losing material can't use it as an escape hatch.
  - **King-vs-King is unreachable from play.** A move never reduces the mover's own pieces, and a bare Raja loses at the start of its turn. This is tested in `tests/test_low_and_speed.py`.

## 2. Verification methodology ✅ *(Students 1 & 3)*

**Differential testing.** The production engine (`chaturanga/`) is checked against an independent reference engine (`chaturanga/reference.py`). The reference has its own 8×8 rank/file representation and its own move generation, move application, terminal ladder and adjudication, and no module-level imports of production code. Perft counts from the two engines were compared at every depth.

**Exit criterion (§2.3): met.** The engines agree exactly to **depth 6 on all 15 positions**. The numbers are frozen in `tests/perft_reference.json`, and every position has its claimed property checked by `tools/perft_freeze.py`.

| Position | d1 | d2 | d3 | d4 | d5 | d6 |
|---|---:|---:|---:|---:|---:|---:|
| `initial_position` | 16 | 256 | 4,176 | 68,133 | 1,164,969 | 19,932,596 |
| `pawn_promotion` | 15 | 269 | 4,851 | 86,296 | 1,611,972 | 29,097,088 |
| `bare_king` | 14 | 104 | 1,592 | 15,294 | 241,815 | 2,418,681 |
| `king_vs_king` | 0 | 0 | 0 | 0 | 0 | 0 |
| `stalemate` | 0 | 0 | 0 | 0 | 0 | 0 |
| `raja_on_guarded_square` | 3 | 73 | 390 | 10,448 | 56,675 | 1,561,645 |
| `gaja_jump` | 6 | 34 | 260 | 1,903 | 15,705 | 129,249 |
| `elephant_orbits` | 9 | 72 | 808 | 7,973 | 96,290 | 1,015,394 |
| `lone_ratha_endgame` | 15 | 90 | 1,619 | 12,063 | 225,914 | 1,715,011 |
| `complex_midgame_1` | 19 | 362 | 7,132 | 140,647 | 2,876,798 | 59,287,534 |
| `complex_midgame_2` | 29 | 641 | 17,847 | 406,906 | 11,078,965 | 258,776,099 |
| `raja_capture_imminent` | 16 | 135 | 2,253 | 17,643 | 298,305 | 2,404,685 |
| `all_pieces_present` | 22 | 418 | 9,102 | 179,847 | 3,959,061 | 80,508,888 |
| `padati_captures` | 14 | 182 | 2,149 | 25,604 | 285,551 | 3,293,722 |
| `ratha_open_file` | 19 | 331 | 6,019 | 106,780 | 1,971,021 | 36,111,098 |

**Property tests** (`tests/`, 163 tests, run in CI at the plan's full sizes):
- `decode(encode(m)) == m` for every legal move in 10,000 random positions;
- the legal mask has exactly `len(legal_moves())` bits set;
- `apply()` never mutates its input;
- `perft(mirror(s), d) == perft(s, d)` for all positions;
- 10,000 random playouts all terminate;
- the stalemate +1 sign, each rung of the terminal ladder, and ladder order;
- canonical (Black-to-move) orientation of both board and policy.

**Pipeline proof (Phase 0):** ⏳ run the Connect-4 gate (random > 95%, greedy > 80%) through the real training loop, then score it with `train_connect4.evaluate_gates`. The MCTS perfect-evaluator test already passes: optimal play from 20 positions at 200 sims (`tests/test_medium_fixes.py`).

## 3. Search ✅ *(Student 2)*

- **Core search:** PUCT MCTS with c = 1.5, Dirichlet root noise (α 0.3, ε 0.25), temperature 1 for 20 plies then argmax, and subtree reuse.
- **Evaluation cache:** a transposition table of network outputs keyed on position + ply.
- **Virtual loss:** for multi-leaf batching.
- **Sign convention:** values are stored from each node's side to move and negated on backup. The stalemate terminal returns +1 to the stalemated player, the opposite of most reference implementations, and is guarded by a unit test.
- **Baselines:**
  - random;
  - greedy-material (one-ply tactics for Connect-4);
  - 3-ply alpha-beta with material + mobility *difference*, depth-aware win scores and captures-first ordering.

  All baselines break ties randomly.

## 4. Network and training ✅ design / ⏳ curves *(Student 3)*

- **Network:** dual-head ResNet with 8×128 blocks (cloud) or 5×64 (local). The policy head is a 1×1 conv to 4,096 logits (channel = from-square, spatial = to-square), with illegal logits masked to −∞ **before** the softmax in training. The value head is a 1×1 conv → FC → tanh.
- **Loss and optimiser:** (z − v)² − πᵀ log p, with AdamW and decoupled weight decay on conv/linear weights only. The plan's explicit c‖θ‖² term is available with `optimizer: adam`.
- **Replay buffer:** compact (bit-packed planes, sparse π), uniform position-level sampling, mirror augmentation applied at sample time.
- **Gating:** the incumbent generates self-play data; a challenger is promoted at ≥ 55% over 60 games with shared random openings.
- ⏳ **Loss curves, policy/value split, value-head calibration:** `python eval/evaluate.py plots RUN_DIR`.
- ⏳ **Mirror-augmentation ablation:** a second run with `train_chaturanga.py --no-mirror`, compared with `python eval/evaluate.py match RUN_A/best.pt RUN_B/best.pt`.

## 5. Systems, performance and cloud ✅ code / ⏳ GPU numbers *(Student 4)*

- **Self-play:** lockstep parallel-games batching (FP16 AMP on CUDA), plus:
  - virtual loss;
  - evaluation cache;
  - multiprocessing workers;
  - batched arena.

  See `ARCHITECTURE_AND_TRAINING.md` §5.
- **Core speed, before → after the optimisation pass** (pure Python, no network):

  | Metric | Before | After |
  |---|---|---|
  | MCTS simulations/s | 3,050 | 9,755 |
  | Memory per tree node | 50.7 KB | 2.3 KB |
  | Move generation | 16.7k positions/s | 83.2k positions/s |
  | perft(start, 4) | 2.04 s | 0.12 s |

  Search decisions are unchanged: identical visit counts on 40 test positions.
- **Cloud (Phase 4b):**
  - `infra/create_vm.sh`: a spot VM in a 1-instance managed instance group, so it is re-created after preemption.
  - `infra/startup.sh`: on boot, pulls the code and the run directory and resumes unattended.
  - `pipeline/gcs.py`: syncs the run directory after every iteration.
  - SIGTERM handler.
  - `--resume auto`, with atomic checkpoint, metrics and buffer writes.
- **Soak test:** `tools/soak_test.py` hard-kills the run mid-iteration 3 times and deletes its working directory each time. **Passed locally:** iterations 0–9, each exactly once; all checkpoints and the buffer load.
- ⏳ **Profiling table and games-per-dollar by machine shape:** run `python -m pipeline.profile --config configs/cloud.yaml --games 256 --sweep workers=1,4,8 leaves_per_step=1,4 --price-per-hour <price>` on each VM shape.

## 6. Compute budget: estimated vs actual ⏳

| Phase | Estimated (plan §2.4) | Actual hours | Actual $ |
|---|---|---|---|
| 4: performance tuning and machine-shape sweep | 25 h | | |
| 4b: cloud soak | 15 h | | |
| 5: main training | 450 h | | |
| 6: arena, baselines, ablations | 90 h | | |
| **Total** | **600 h** | | |

## 7. Results ⏳ *(Student 2 lead)*

| Deliverable (Part 8) | Command |
|---|---|
| Elo curve across ≥ 30 promoted checkpoints | `python eval/evaluate.py ladder RUN_DIR --games 20 --sims 50` |
| Win rates vs random, greedy, 3-ply alpha-beta | `python eval/evaluate.py baselines RUN_DIR/best.pt --games 40` |
| Sims-vs-strength (10/30/100/400 at fixed weights) | `python eval/evaluate.py sims RUN_DIR/best.pt` |
| Terminal-reason distribution across training | `python eval/evaluate.py plots RUN_DIR` → `terminal_reasons.png` |
| One annotated sample game | `python eval/evaluate.py game RUN_DIR/best.pt` → `sample_game.md` |

## 8. Rule-specific findings ⏳ *(all)*

Questions from plan Phase 6, with where to look:
- **Gaja orbits:** does the agent exploit orbit confinement? Inspect sample games; compare Gaja trade rates between early and late checkpoints.
- **Raja activity:** does the Raja stay active, and does the agent bait the opponent's Raja onto guarded squares?
- **How games end:** bare king vs Raja capture, and how that ratio shifts with training (`terminal_reasons.png`).
- **Missed captures:** how fast the missed-Raja-capture rate falls (`missed_raja.png`).
- **Stalemate:** did it ever fire? It is counted every iteration in `metrics.json` → `terminal_stats`.

## 9. What did not work, and what we fixed ✅ (so far)

From the code review (`needtofix.md`). These are the bugs that would have silently ruined training, with how each was caught:

1. **Black's policy wasn't oriented like its board.** The board was flipped for Black, but move targets weren't. Fixed with `orient_policy`; test: colour-flipped twins.
2. **The first PUCT visit ignored the prior.** `sqrt(0)` = 0 meant every first visit went to `legal_actions[0]`.
3. **Policy logits were clamped to ≥ 0** by a stray ReLU.
4. **Gating had no effect.** Self-play used the newest net, and deterministic arena players replayed 2 games 30 times.
5. **Resume overwrote `best.pt`** and erased the metrics history.
6. **The "independent" reference engine shared production code**, so differential testing didn't cover the terminal ladder.
7. **A tests-that-can't-fail pattern.** The stalemate test's position wasn't actually stalemate, and it printed PASS anyway.
8. **The replay buffer would have needed ~20 GB** at 1M positions.

⏳ Add anything that goes wrong during the main run, for example cap-rate spikes or plateaus, and what you changed in response.
