# ChaturangaZero: Known Issues

From a full code review against `IMPLEMENTATION_PLAN.md` (v2.1), at commit `cfa2e49`.
File:line references point to that commit.

**Status:** ✅ fixed · 🟡 built, waiting on training / cloud account / team · ⬜ open. Critical fixes are on `main`; everything else is on branch `fix/high-priority`.

**Summary:** all 7 critical, 9 high, 9 medium, 8 low and 7 speed/memory items are fixed. Of the 7 plan deliverables, I1 is done and I2–I7 are built and tested; what remains is running the training, using your GCP account, and the team's own sign-offs and report sections.

---

## 🔴 Critical: silently ruins training

| # | Status | Where | Problem | Fix |
|---|---|---|---|---|
| C1 | ✅ | `chaturanga/encoding.py:54` | **Black's board was flipped but its moves weren't.** For Black to move, the board was encoded rank-flipped, but π targets and network policy stayed in absolute squares. The network had to learn a second, scrambled move mapping for Black. | Added `CANONICAL_PERM` (`sq ^ 56` on both squares) and `State.orient_policy()`. It's applied to π when recording training data, and to network output in every evaluator (`create_evaluator`, `BatchedEvaluator`, web app). |
| C2 | ✅ | `search/mcts.py:159` | **The first PUCT visit ignored the prior.** With 0 visits, `sqrt(0)=0` zeroed the exploration term, so every move tied and the first visit always went to `legal_actions[0]`. | `sqrt(max(total_n, 1))` |
| C3 | ✅ | `net/model.py:126` | **Policy logits were clamped to ≥ 0.** BN → ReLU fed the logits directly, so bad moves couldn't be pushed below the rest. | ReLU removed from the logits. It's kept only before the Connect-4 FC layer, where it's a hidden activation. |
| C4 | ✅ | `pipeline/loop.py:141,256` | **Arena gating did nothing.** Self-play always used the newest network, so rejecting a challenger changed nothing. Both arena players were deterministic, so the 60 games were 2 games repeated 30×. | Self-play now uses the incumbent (`best_net`); a challenger replaces it only after winning the gate. Each pair of arena games starts from the same random opening (`arena.opening_plies`, default 4), once per colour. |
| C5 | ✅ | `pipeline/loop.py:130,141` | **Resuming wiped history.** `best.pt` was overwritten with the resumed network, and `metrics.json` restarted from empty. | On resume, the existing `best.pt` is loaded, not overwritten. `metrics.json` keeps its entries before the resume point. |
| C6 | ✅ | `chaturanga/reference.py` | **The "independent" reference engine reused production code:** `apply_move` (promotion), `is_terminal` (the whole terminal ladder) and `encode_action`. So differential perft checked only piece movement. | The reference now has its own `ref_apply_move`, `ref_is_terminal` and adjudication, written from `RULES.md`. Only the `State` container is shared, and a test enforces that. |
| C7 | ✅ | `net/replay.py` | **Replay buffer ≈ 20 GB at 1M positions.** Each entry held a dense 4096-float π and float32 planes. Mirrored copies used half the capacity. Random access on a `deque` is O(n). | numpy ring buffer: bit-packed planes, sparse π, mirroring at sample time (p = 0.5). ≈ 40× smaller, O(1) access. The buffer-age histogram is now real too. It still loads the old `.npz` format. |

Regression tests for all seven are in `tests/test_critical_fixes.py`. The C2 and C3 tests were confirmed to fail on the old code.

**Side effects of the fixes:**
- Checkpoints trained before C1/C3 load without errors, but their policy is now read differently, so retrain rather than reuse them.
- `buffer_size` now counts unique positions. A 1M buffer used to hold only 500k unique positions plus their mirrored copies.
- `run_batched_self_play(..., mirror_augmentation=...)` and `ReplayBuffer.add_game(..., mirror_augment, mirror_perm)` were removed. Pass `mirror_perm` to `ReplayBuffer(...)` instead.

---

## 🟠 High: broken features or plan violations

| # | Status | Where | Problem | Fix |
|---|---|---|---|---|
| H1 | ✅ | `eval/play_cli.py:57` | **The human-play CLI crashes on startup.** It imports `RandomPlayer`, `GreedyMaterialPlayer` and `AlphaBetaPlayer`, which don't exist; `baselines.py` only defines functions. | Call `random_player` / `greedy_material_player` / `alpha_beta_player` directly. |
| H2 | ✅ | `eval/play_cli.py:76`, `web/app.py:68` | **The 5×64 architecture is hard-coded**, so cloud (8×128) checkpoints fail to load. | Save the architecture config in the checkpoint and rebuild the network from it when loading. |
| H3 | ✅ | `net/train.py:95` | **No masked policy loss.** The softmax runs over all 4096 actions (plan §3.1 says to mask). | Store legal masks in the buffer and pass `legal_mask=` to the net. Don't multiply 0 by −inf in the cross-entropy (it gives NaN). |
| H4 | ✅ | `search/arena.py`, `search/baselines.py` | **Arena and baseline games apply material adjudication at the cap**, which §2.2 forbids. | Add `is_terminal(adjudicate=False)` → a draw at the cap, for arena and evaluation games. |
| H5 | ✅ | `chaturanga/*.py`, configs | **Config values are ignored.** `move_cap` (hard-coded `MOVE_CAP = 300`), `adjudicate_margin` (4.0), `virtual_loss`, `workers` and `reuse_subtree` are unused, so the §2.2 "tighten cap to 200" response is impossible. The `assert ply_count <= MOVE_CAP` invariant is also missing. | Make the cap and margin configurable at runtime and add the assertion in `apply_move`. Wire up or delete the other keys. |
| H6 | ✅ | `tests/test_rules.py:123,301` | **Two tests can never fail.** The stalemate test's position isn't a stalemate (a7/b7 can promote), so a fallback branch prints PASS. The termination test sets a flag but never asserts. | Use a real stalemate position (see below) and add `assert all_terminated`. |
| H7 | ✅ | `search/baselines.py:104,147` | **The alpha-beta baseline is skewed.** Mobility counts only the side to move (biased by ply parity). Win scores ignore depth, so it can postpone a Raja capture. There's no move ordering. | Use mobility difference, `±(1000 + depth)` for wins, and captures first (MVV). |
| H8 | ✅ | `search/baselines.py` | **Greedy and alpha-beta break ties by taking the first move**, so they're deterministic and repeat identical games. | Pick randomly among tied moves. |
| H9 | ✅ | `eval/elo.py:40` | **Elo is updated one game at a time (K=32)**, so the result depends on game order. Wrong for a checkpoint ladder. | Fit all results at once (Bradley–Terry / maximum-likelihood). |

Regression tests for H1–H5 and H7–H9 are in `tests/test_high_fixes.py`. The H6 fix is in `tests/test_rules.py` itself. 9 of the new tests were confirmed to fail on the old code.

**How each was fixed:**
- **H1:** the CLI uses the baseline functions through `make_baseline()`.
- **H2:** checkpoints store `net_config`. `net.model.load_net_from_checkpoint()` rebuilds the right architecture, and infers it from the weight shapes for older checkpoints. The CLI, Connect-4 CLI and web app all use it.
- **H3:** the buffer stores each position's legal moves (mirrored together with π). `Trainer` masks illegal logits to −inf and avoids 0 × −inf = NaN.
- **H4:** `is_terminal(adjudicate=False)` scores a capped game as a draw. The arena and both baselines use it.
- **H5:** `configure_rules()`, `get_move_cap()` and `get_adjudicate_margin()`. The loop applies `play.move_cap` and `play.adjudicate_margin`, `apply_move` asserts the cap invariant, and `mcts.reuse_subtree` is wired into the batched server. `virtual_loss` and `workers` are marked "not implemented (M3)" in the configs.
- **H6:** a real stalemate position, and the termination test now asserts.
- **H7:** mobility is us-minus-them, wins score `±(1000 + remaining depth)`, captures are ordered most valuable victim first.
- **H8:** moves are shuffled before search, so ties break randomly.
- **H9:** Bradley–Terry maximum-likelihood fit (MM algorithm) with a weak prior. `round_robin_elo` uses shared random openings.

**Side effects:**
- Buffer files saved before H3 still load; their masks fall back to π's support.
- The move cap is process-global state. Tests reset it after each test.
- The reference engine keeps its own fixed `REF_MOVE_CAP = 300`, so differential perft is only meaningful at the default cap.

A real stalemate position for H6 (White to move has no legal move, so White wins):
White Raja a1, Gajas b1 b2, Padatis a2 c2 d3 d4; Black Padatis a3 c3 d5, Raja h8.

---

## 🟡 Medium

| # | Status | Where | Problem |
|---|---|---|---|
| M1 | ✅ | `pipeline/selfplay.py` | Subtree reuse is fake in sequential mode: `_get_root_from_search` returns `None`. It does work in batched mode. |
| M2 | ✅ | `search/` | No transposition table. `State.key()` exists but nothing uses it. |
| M3 | ✅ | `pipeline/inference_server.py` | No virtual loss and no multiprocessing worker pool. It evaluates one leaf per game per simulation, and batches shrink as games finish (the "tail" problem). |
| M4 | ✅ | `pipeline/selfplay.py:142` | `_classify_terminal` checks the move cap before stalemate, the reverse of the terminal ladder, so a stalemate at ply 300 is counted as a cap ending. |
| M5 | ✅ | `pipeline/loop.py` | Reproducibility: Python's `random` isn't seeded, and no git SHA is stored with a run. |
| M6 | ✅ | `net/train.py`, `pipeline/` | Missing diagnostics: value-head calibration, draw rate, missed-Raja-capture counter, stalemate counter. (Buffer age is fixed by C7.) |
| M7 | ✅ | `train_connect4.py`, `tests/` | Phase-0/2 gates not checked. Only "vs random" is tested. "Greedy > 80%" is meaningless for Connect-4, since greedy always plays the first column. There's no perfect-evaluator MCTS test. |
| M8 | ✅ | `pipeline/loop.py` | The buffer snapshot is taken every 10 iterations, so a resume can lose up to 9 iterations of data. C7 makes snapshots cheap enough to take every iteration. |
| M9 | ✅ | `tests/perft_reference.json` | Perft numbers exist only for the start position at depths 1–4. The other 14 positions are empty, and nothing reaches depth 6. |

How each was fixed. Tests are in `tests/test_medium_fixes.py` and `tests/test_perft_reference.py`:

- **M1:** `search(..., return_root=True)`. Sequential self-play now passes the chosen child's subtree into the next search.
- **M2:** `search/cache.py` `EvalCache` is an LRU evaluation cache keyed on `State.key()` + ply. The ply is included because it's in the network input. It stores legal moves + priors + value, or the terminal value, and is shared by all games in one self-play run. Only evaluations are shared; each tree keeps its own visit counts, so backup is unchanged. Hit rate is logged per iteration. Setting: `mcts.eval_cache_size`.
- **M3:**
  - Virtual loss in the batched server: `mcts.leaves_per_step` leaves per game per network batch, with `mcts.virtual_loss` applied along pending paths and fully reverted afterwards. The default is 1, so behaviour is unchanged until you benchmark.
  - Worker processes: `pipeline/workers.py`, set with `loop.workers`. They use 'spawn', so they work on Windows and with CUDA. Each worker runs its own batched server and returns a compact buffer, merged with `ReplayBuffer.merge`.
- **M4:** `chaturanga.terminal.evaluate_terminal()` returns `(done, z, reason)`. `is_terminal`, `terminal_reason` and the self-play classifier all use it, so they can't disagree.
- **M5:** Python's `random` is seeded. The git SHA and a dirty flag go into `config.yaml` (`run_info`) and every checkpoint.
- **M6:**
  - In `metrics.json` per iteration: draw rate, missed-Raja-capture count and rate, games/hour, eval-cache hit rate, buffer-age histogram.
  - Value-head calibration error, sign accuracy and mean |v|.
  - All terminal reasons are always reported, even at 0, so a zero stalemate count is visible.
- **M7:** `connect4_tactical_player` (win now, else don't hand over a win, else random) is the Connect-4 "greedy". `train_connect4.py` gates on random > 95% and greedy > 80%. `connect4/solver.py` provides exact values, and the test plays MCTS (200 sims, perfect evaluator) from 20 late positions: all optimal.
- **M8:** buffer snapshot every iteration by default, written to a temp file and renamed, so a preemption can't corrupt it.
- **M9:** `tools/perft_freeze.py` defines the 15 positions as FENs (new `to_fen` / `from_fen`), checks each position has its claimed property, runs both engines (in parallel for depths ≥ 5), stops on any disagreement with a per-move breakdown, and writes `tests/perft_reference.json`. Both perfts count the last ply without applying moves (same counts, much faster). `pytest` checks every position up to ~200k leaves with both engines; set `CHATURANGA_FULL_PERFT=1` for every frozen depth.

---

## 🔵 Low

| # | Status | Where | Problem |
|---|---|---|---|
| L1 | ✅ | `requirements.txt` | `flask` is missing, and so is `pytest` for development. |
| L2 | ✅ | `ARCHITECTURE_AND_TRAINING.md` §5 | Describes a queue/thread inference server and a measured 35× speedup. Neither exists; the server works in lockstep and nothing has been measured. |
| L3 | ✅ | `net/train.py` | L2 is applied via Adam's `weight_decay`, which interacts badly with Adam's adaptive step sizes. Use AdamW or an explicit L2 term. |
| L4 | ✅ | `web/app.py` | One global game session with no lock, so concurrent requests race. |
| L5 | ✅ | `net/model.py` | The policy head's flat index is channel×64 + square, so channel = from-square and spatial position = to-square. Plan §3.3 states the opposite. It works, but the docs should match the code. |
| L6 | ✅ | `chaturanga/perft.py` | `perft_divide` doesn't check for a terminal root, unlike `perft`. |
| L7 | ✅ | `chaturanga/reference.py` | The reference engine still uses the production `State` container (board array + side + ply). That's acceptable, but a fully separate 8×8 list representation would make it truly independent. |
| L8 | ✅ | report | **Note, not a bug:** King-vs-King can never arise from normal play. A player can't lose a piece on their own move, and a bare king loses as soon as it's its turn. So rule 2 never fires, like stalemate. Add a unit test for this and mention it in the report. |

---

How each was fixed. Tests are in `tests/test_low_and_speed.py`:

- **L1:** `flask` added to `requirements.txt`, with install notes for the CPU vs GPU PyTorch builds. New `requirements-dev.txt` adds `pytest`.
- **L2:** `ARCHITECTURE_AND_TRAINING.md` §5 now describes the real lockstep server, virtual loss, cache, workers and batched arena. The unmeasured "35× / 650 games/h" figures were removed, and measured core numbers were added.
- **L3:** default optimizer is **AdamW** with decoupled `train.weight_decay` (1e-2, a new hyperparameter worth tuning), applied to conv/linear weights only, never BatchNorm or biases. `optimizer: adam` gives the plan's literal `+ c‖θ‖²` loss term, logged as `l2_loss`.
- **L4:** every web route that touches the shared game session holds a lock.
- **L5:** `net/model.py` and `ARCHITECTURE_AND_TRAINING.md` now state the real layout: channel = from-square, spatial = to-square. A test pins it. `IMPLEMENTATION_PLAN.md` is the team's plan and was left unchanged.
- **L7:** the reference engine has its own `RefPosition` (8×8 rank/file grid) and its own constants, with no module-level production imports. It reads a `State` only at the boundary.
- **L8:** tested: a move never reduces the mover's own piece count, and random games never end King-vs-King. Still worth a sentence in the report.

## ⚡ Speed and memory optimizations, biggest first

| # | Status | Where | Change |
|---|---|---|---|
| S1 | ✅ | `search/node.py:35` | **MCTS nodes are huge:** 3 × 4096 float32 = 48 KB per node, for about 30 legal moves. That's ≈ 600 MB of allocate-and-discard per move step at 128 games × 100 simulations. Size the per-edge arrays to the legal moves instead, and do PUCT with numpy. |
| S2 | ✅ | `net/replay.py` | Compact ring buffer (C7). |
| S3 | ✅ | `chaturanga/terminal.py`, MCTS | Move generation runs 2–3× per expanded node (`is_terminal` for the stalemate check, then expansion, then move selection). Cache the legal moves on `State`. |
| S4 | ✅ | `chaturanga/board.py`, `encoding.py` | `encode_state`, `has_raja`, `find_pieces` and `material_score` loop over 64 squares in Python. Vectorize with numpy masks. |
| S5 | ✅ | `pipeline/inference_server.py` | Recycled-slot and replacement-root expansions each do a batch-of-1 forward pass. Group them into one batch. |
| S6 | ✅ | `search/arena.py`, `eval/elo.py` | Arena and Elo games run sequential batch-1 MCTS. Reuse the batched server. |
| S7 | ✅ | `search/baselines.py` | Alpha-beta generates moves twice per node (`is_terminal` + the move loop) plus once more for mobility. With move ordering (H7) it would be much faster. |

---

How each was fixed. Search decisions are unchanged: visit counts are identical on 40 test positions, move lists on 3,000, encodings byte-for-byte, and the batched arena gives the same results as the sequential one.

- **S1:** `Node` stores `N/W/P` only for its legal moves, plus `actions` (edge → action). PUCT is one numpy expression. `visit_distribution()` still returns the full 4096 vector.
- **S3:** each `State` caches its legal moves and its terminal result. The terminal cache is keyed on adjudicate/cap/margin, so `configure_rules()` invalidates it. A `State` must not be modified in place after use.
- **S4:** move generation uses precomputed per-square target tables over a plain Python list, visiting only the side's own pieces, with the same move order as before. `encode_state`, `mirror_state`, `has_raja`, `material_score` and `find_pieces` are numpy-vectorised.
- **S5:** recycled-slot and replacement roots are expanded in one batched forward pass, plus the cache, instead of batch-of-1 calls.
- **S6:** `run_batched_arena()` plays all gating games concurrently with one forward pass per network per simulation step. The training loop uses it.
- **S7:** alpha-beta benefits from S3/S4: terminal check and move loop now share one cached move generation per node.

## 🛠 Improvements and missing plan deliverables

✅ done · 🟡 everything built and tested; the remaining step needs the training run, your GCP account, or the team

| # | Status | Area | What was done | What's left |
|---|---|---|---|---|
| I1 | ✅ | Verification | All 15 perft positions frozen to **depth 6** (both engines agree: the §2.3 exit criterion). `tests/test_terminal.py`, `test_encoding.py` and `test_mirror.py` cover: `decode(encode(m))` for every legal move in 10k positions, the legal-mask count, `apply()` never mutating its input, `perft(mirror(s)) == perft(s)` on all positions, 10k terminating playouts with a sane end-reason mix, and every terminal rung and the ladder order. | — |
| I2 | 🟡 | CI | `.github/workflows/tests.yml`: Ubuntu, Python 3.12, CPU PyTorch, full suite with `CHATURANGA_FULL_TESTS=1` (the plan's 10k sizes) on every push and pull request. | It hasn't run on GitHub yet; it will on the first push. |
| I3 | 🟡 | Cloud (Phase 4b) | `infra/create_vm.sh` (spot VM in a 1-instance managed instance group, re-created after preemption), `infra/startup.sh` (pull code and run directory, resume unattended), `infra/Dockerfile`, `pipeline/gcs.py` (sync after every iteration; never syncs temp files), `pipeline/resume.py` + `--resume auto` (newest checkpoint that loads), SIGTERM handler, atomic checkpoint/metrics/buffer writes, fresh RNG streams on resume. **Soak test** `tools/soak_test.py`: 3 hard kills mid-iteration, each wiping the working directory → resumed every time, iterations 0–9 each exactly once. **Passed locally.** | Needs your GCP account: GPU quota, a bucket, then `infra/create_vm.sh` and `tools/soak_test.py --bucket gs://… --signal term` on the VM. A preemption repeats the iteration in progress. |
| I4 | 🟡 | Performance (Phase 4) | `pipeline/profile.py`: end-to-end self-play games/hour, sims/s, network time share, mean batch, cache hit rate, sweeps (`workers`, `parallel_games`, `leaves_per_step`, `sims`), games per dollar. `tools/bench_core.py`: measured core before/after table (2.4–5.9× MCTS, 22× less memory per node), now in `ARCHITECTURE_AND_TRAINING.md` §5.2. | Run `pipeline.profile` on the 3050 machine and on each VM shape to get the GPU numbers and pick the Phase-4 tier. |
| I5 | 🟡 | Results (Phases 5–6) | `eval/evaluate.py`: `ladder` (Elo curve over checkpoints), `baselines` (vs random / greedy / 3-ply alpha-beta), `sims` (10/30/100/400 at fixed weights), `match` (mirror ablation head-to-head), `plots` (loss split, value head, game length, cap and draw rate, terminal reasons, missed Raja captures, throughput), `game` (annotated sample game). `train_chaturanga.py --no-mirror` for the ablation run. All smoke-tested on a real tiny run. | Needs the main training run (and a short `--no-mirror` run), then the commands listed in `REPORT.md` §7. |
| I6 | 🟡 | Report (Phase 7) | `REPORT.md`: the plan's structure with section owners. Pre-filled: rules and no-check design, verification (the full frozen perft table), search, network, systems and cloud with measured numbers, and "what didn't work" from this review. Results sections list the exact command for each figure. | Each student writes their section; fill in results, compute actuals and rule-specific findings after training. |
| I7 | 🟡 | Process | §2.2 move-cap and adjudication rule (with the monitoring response), the King-vs-King-unreachable note, and a sign-off table added to `RULES.md`. | The four of you initial and date the sign-off table. |

---|---|---|
| I1 | Verification | Fill all 15 perft positions to depth 6 and freeze them in `perft_reference.json`. Add property tests: `decode(encode(m))` on 10k positions, `perft(mirror(s)) == perft(s)`, 10k terminating playouts. Add the missing `test_terminal.py`, `test_encoding.py` and `test_mirror.py`. |
| I2 | CI | No CI config exists. Add a GitHub Actions workflow that runs `pytest` on every push. |
| I3 | Cloud (Phase 4b) | Not started: `infra/` (Dockerfile, `create_vm.sh`, `startup.sh`), `pipeline/gcs.py` (checkpoint sync), `pipeline/resume.py`, SIGTERM preemption handler, soak test. |
| I4 | Performance (Phase 4) | `pipeline/profile.py` and a measured before/after games-per-hour table. Machine-shape sweep. |
| I5 | Results (Phases 5–6) | Main training run, Elo curve over ≥ 30 promoted checkpoints, win rates against the 3 baselines, sims-vs-strength and mirror ablations, terminal-reason distribution over training, annotated sample game. |
| I6 | Report (Phase 7) | Compute budget (estimated vs actual), rule-specific findings, what did not work. |
| I7 | Process | Record the four sign-offs on `RULES.md`, and add the §2.2 adjudication rule to it. |

---

## ✅ Other changes on this branch (not bugs)

- `utils/device.py`: `get_device()` picks CUDA, then MPS, then CPU automatically. Override with `--device`, `CHATURANGA_DEVICE`, or `device:` in the config. AMP is turned on only on CUDA.
