# ChaturangaZero: Known Issues

From a full code review against `IMPLEMENTATION_PLAN.md` (v2.1), at commit `cfa2e49`.
File:line references point to that commit.

**Status:** ✅ fixed on branch `feature/device-autodetect` (not yet committed) · ⬜ open

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
| H1 | ⬜ | `eval/play_cli.py:57` | **The human-play CLI crashes on startup.** It imports `RandomPlayer`, `GreedyMaterialPlayer` and `AlphaBetaPlayer`, which don't exist; `baselines.py` only defines functions. | Call `random_player` / `greedy_material_player` / `alpha_beta_player` directly. |
| H2 | ⬜ | `eval/play_cli.py:76`, `web/app.py:68` | **The 5×64 architecture is hard-coded**, so cloud (8×128) checkpoints fail to load. | Save the architecture config in the checkpoint and rebuild the network from it when loading. |
| H3 | ⬜ | `net/train.py:95` | **No masked policy loss.** The softmax runs over all 4096 actions (plan §3.1 says to mask). | Store legal masks in the buffer and pass `legal_mask=` to the net. Don't multiply 0 by −inf in the cross-entropy (it gives NaN). |
| H4 | ⬜ | `search/arena.py`, `search/baselines.py` | **Arena and baseline games apply material adjudication at the cap**, which §2.2 forbids. | Add `is_terminal(adjudicate=False)` → a draw at the cap, for arena and evaluation games. |
| H5 | ⬜ | `chaturanga/*.py`, configs | **Config values are ignored.** `move_cap` (hard-coded `MOVE_CAP = 300`), `adjudicate_margin` (4.0), `virtual_loss`, `workers` and `reuse_subtree` are unused, so the §2.2 "tighten cap to 200" response is impossible. The `assert ply_count <= MOVE_CAP` invariant is also missing. | Make the cap and margin configurable at runtime and add the assertion in `apply_move`. Wire up or delete the other keys. |
| H6 | ⬜ | `tests/test_rules.py:123,301` | **Two tests can never fail.** The stalemate test's position isn't a stalemate (a7/b7 can promote), so a fallback branch prints PASS. The termination test sets a flag but never asserts. | Use a real stalemate position (see below) and add `assert all_terminated`. |
| H7 | ⬜ | `search/baselines.py:104,147` | **The alpha-beta baseline is skewed.** Mobility counts only the side to move (biased by ply parity). Win scores ignore depth, so it can postpone a Raja capture. There's no move ordering. | Use mobility difference, `±(1000 + depth)` for wins, and captures first (MVV). |
| H8 | ⬜ | `search/baselines.py` | **Greedy and alpha-beta break ties by taking the first move**, so they're deterministic and repeat identical games. | Pick randomly among tied moves. |
| H9 | ⬜ | `eval/elo.py:40` | **Elo is updated one game at a time (K=32)**, so the result depends on game order. Wrong for a checkpoint ladder. | Fit all results at once (Bradley–Terry / maximum-likelihood). |

A real stalemate position for H6 (White to move has no legal move, so White wins):
White Raja a1, Gajas b1 b2, Padatis a2 c2 d3 d4; Black Padatis a3 c3 d5, Raja h8.

---

## 🟡 Medium

| # | Status | Where | Problem |
|---|---|---|---|
| M1 | ⬜ | `pipeline/selfplay.py` | Subtree reuse is fake in sequential mode: `_get_root_from_search` returns `None`. It does work in batched mode. |
| M2 | ⬜ | `search/` | No transposition table. `State.key()` exists but nothing uses it. |
| M3 | ⬜ | `pipeline/inference_server.py` | No virtual loss and no multiprocessing worker pool. It evaluates one leaf per game per simulation, and batches shrink as games finish (the "tail" problem). |
| M4 | ⬜ | `pipeline/selfplay.py:142` | `_classify_terminal` checks the move cap before stalemate, the reverse of the terminal ladder, so a stalemate at ply 300 is counted as a cap ending. |
| M5 | ⬜ | `pipeline/loop.py` | Reproducibility: Python's `random` isn't seeded, and no git SHA is stored with a run. |
| M6 | ⬜ | `net/train.py`, `pipeline/` | Missing diagnostics: value-head calibration, draw rate, missed-Raja-capture counter, stalemate counter. (Buffer age is fixed by C7.) |
| M7 | ⬜ | `train_connect4.py`, `tests/` | Phase-0/2 gates not checked. Only "vs random" is tested. "Greedy > 80%" is meaningless for Connect-4, since greedy always plays the first column. There's no perfect-evaluator MCTS test. |
| M8 | ⬜ | `pipeline/loop.py` | The buffer snapshot is taken every 10 iterations, so a resume can lose up to 9 iterations of data. C7 makes snapshots cheap enough to take every iteration. |
| M9 | ⬜ | `tests/perft_reference.json` | Perft numbers exist only for the start position at depths 1–4. The other 14 positions are empty, and nothing reaches depth 6. |

---

## 🔵 Low

| # | Status | Where | Problem |
|---|---|---|---|
| L1 | ⬜ | `requirements.txt` | `flask` is missing, and so is `pytest` for development. |
| L2 | ⬜ | `ARCHITECTURE_AND_TRAINING.md` §5 | Describes a queue/thread inference server and a measured 35× speedup. Neither exists; the server works in lockstep and nothing has been measured. |
| L3 | ⬜ | `net/train.py` | L2 is applied via Adam's `weight_decay`, which interacts badly with Adam's adaptive step sizes. Use AdamW or an explicit L2 term. |
| L4 | ⬜ | `web/app.py` | One global game session with no lock, so concurrent requests race. |
| L5 | ⬜ | `net/model.py` | The policy head's flat index is channel×64 + square, so channel = from-square and spatial position = to-square. Plan §3.3 states the opposite. It works, but the docs should match the code. |
| L6 | ⬜ | `chaturanga/perft.py` | `perft_divide` doesn't check for a terminal root, unlike `perft`. |
| L7 | ⬜ | `chaturanga/reference.py` | The reference engine still uses the production `State` container (board array + side + ply). That's acceptable, but a fully separate 8×8 list representation would make it truly independent. |
| L8 | ⬜ | report | **Note, not a bug:** King-vs-King can never arise from normal play. A player can't lose a piece on their own move, and a bare king loses as soon as it's its turn. So rule 2 never fires, like stalemate. Add a unit test for this and mention it in the report. |

---

## ⚡ Speed and memory optimizations, biggest first

| # | Status | Where | Change |
|---|---|---|---|
| S1 | ⬜ | `search/node.py:35` | **MCTS nodes are huge:** 3 × 4096 float32 = 48 KB per node, for about 30 legal moves. That's ≈ 600 MB of allocate-and-discard per move step at 128 games × 100 simulations. Size the per-edge arrays to the legal moves instead, and do PUCT with numpy. |
| S2 | ✅ | `net/replay.py` | Compact ring buffer (C7). |
| S3 | ⬜ | `chaturanga/terminal.py`, MCTS | Move generation runs 2–3× per expanded node (`is_terminal` for the stalemate check, then expansion, then move selection). Cache the legal moves on `State`. |
| S4 | ⬜ | `chaturanga/board.py`, `encoding.py` | `encode_state`, `has_raja`, `find_pieces` and `material_score` loop over 64 squares in Python. Vectorize with numpy masks. |
| S5 | ⬜ | `pipeline/inference_server.py` | Recycled-slot and replacement-root expansions each do a batch-of-1 forward pass. Group them into one batch. |
| S6 | ⬜ | `search/arena.py`, `eval/elo.py` | Arena and Elo games run sequential batch-1 MCTS. Reuse the batched server. |
| S7 | ⬜ | `search/baselines.py` | Alpha-beta generates moves twice per node (`is_terminal` + the move loop) plus once more for mobility. With move ordering (H7) it would be much faster. |

---

## 🛠 Improvements and missing plan deliverables

| # | Area | Item |
|---|---|---|
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
