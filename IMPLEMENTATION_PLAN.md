# ChaturangaZero — Implementation Plan (v2)

**Self-play AlphaZero agent for Chaturanga**
Team of 4 · 16 weeks · local RTX 3060 for development + GCP for training

*Planning document only. No code has been written. Every performance number is an estimate to be replaced with a measurement in Week 1 (§4.4).*

**Changes from v1:** ruleset is now historically authentic rather than chess-aligned, per the team's decision. This is a strictly better outcome for the project — see §2.0. Compute assumptions revised upward for GCP; network and search scaled up accordingly. Schedule extended to 16 weeks.

**Changes in v2.1:** clarified that a Raja may step onto a guarded square and the opponent may then capture it **on their turn** — the capture is *played*, not automatic (§2.1). Stalemate-wins is **retained** as originally specified: a player with no legal move wins.

**Changes in v2.2** (from the second code review, `needtofixagain.md` P1–P10): corrected the stalemate geometry (§2.1) and the Gaja orbit count (§2.3); replaced the `undo` property test, which the interface contract doesn't support (§2.3); noted that King-vs-King is unreachable from legal play (§2.1, §2.3); fixed the ply-plane normaliser so tightening the cap doesn't change what the network sees (§2.2, §3.2); added a target replay ratio and a warm-up buffer (§2.4, Part 6); set arena search sims and a statistically meaningful gate (§5 Phase 5, Part 6); made the Elo-curve gate reachable (§5 Phase 5); documented AdamW as the optimiser (§1.2, Part 6); and added an optional relative move-type policy head as an ablation (§3.3). No game rule changed.

---

## 0. Decisions locked

| Question | Decision |
| --- | --- |
| Castling | **No** |
| Pawn double-step / en passant | **No** |
| Promotion | **Mantri only**, forced |
| Check / checkmate | **Does not exist.** Game ends on Raja capture |
| Raja on a guarded square | **Legal.** The opponent may capture it on their turn — no automatic loss |
| Stalemate | **Win for the stalemated player** — the side with no legal move |
| Repetition | **No repetition rule.** Positions may repeat freely |
| Bare king | **Automatic loss**, except King-vs-King which is a draw |
| Compute | GCP, spot instances, ~600 effective GPU-hours assumed (§4.4) |
| Inference | **Parallel-games batched mode** — committed, not optional |
| Timeline | 16 weeks |
| Chess compatibility | **Dropped entirely.** No chess-configured build |

---

# PART 1 — Understanding the Proposal

## 1.1 The game

Chaturanga is the 6th-century Indian ancestor of chess, played on the 8×8 uncheckered **Ashtapada**. Under the frozen ruleset (§2.1):

| Piece | Sanskrit | Movement |
| --- | --- | --- |
| King | **Raja** | One square in any of eight directions. No castling. **Capturable, and may legally step onto a guarded square.** |
| Minister | **Mantri** | Exactly one square diagonally |
| Chariot | **Ratha** | Any distance orthogonally — the only sliding piece in the game |
| Elephant | **Gaja** | Exactly two squares diagonally, **jumping** the intervening square |
| Horse | **Ashva** | Standard knight leap |
| Foot-soldier | **Padati** | One step forward; captures one step diagonally forward; no double-step; promotes to Mantri on the last rank |

Four structural facts that drive most of the engineering below. None appear in the original proposal.

**The Gaja reaches only 8 squares, ever.** A two-square diagonal leaper partitions the board into eight disjoint 8-square orbits. Each elephant is permanently confined to one orbit and can never attack a piece on any other. Two elephants on different orbits are mutually invisible for the whole game. Not a bug — a property the network must discover, and the reason a Gaja is worth about a pawn rather than about a knight.

**Only one piece slides.** The Ratha is the sole long-range attacker. Everything else is a one- or two-square leaper. Material converts slowly and positions are close-quarters.

**The Raja is a fighting piece.** With no check rule, the king can walk anywhere, including onto attacked squares. King activity from move one is legal and probably strong. This is the largest tactical departure from chess intuition and the network has no rule scaffolding to help it — it must learn king safety purely from losing games.

**Piece values are not chess values.** A working first estimate:

```
Ratha 5.0   Ashva 3.0   Gaja 1.5   Mantri 1.5   Padati 1.0
```

The Mantri, despite being the "queen slot," is the second-weakest piece on the board.

## 1.2 The AI approach

AlphaZero. No opening book, no hand-written evaluation, no human games. The agent gets the rules and nothing else, and improves by playing itself.

**A. Game environment** — a state machine encoding the board as a multi-channel binary tensor, generating legal moves, detecting terminal states. Final specification: **14 planes** (§3.2) and a **4096-action** space (§3.1). The original proposal's estimate of 13 channels and 4096 actions was essentially correct — the authentic ruleset is what makes it correct.

**B. Guided Monte Carlo Tree Search** — the agent does not move on raw network output and does not use random rollouts:
- the **policy head supplies the prior** `P(s,a)`, steering which branches get explored;
- the **value head evaluates leaf nodes**, replacing random playouts entirely;
- **PUCT selection** balances prior against accumulated visit statistics.

The root visit-count distribution — not the raw policy — becomes both the move played and the training target. This is the core of AlphaZero: search improves on the policy, and the improved policy is distilled back into the network.

**C. Dual-head residual network** —
- **Policy head** → distribution over 4096 actions
- **Value head** → scalar in [−1, +1], expected outcome for the side to move
- **Loss** `ℓ = (z − v)² − πᵀ log p + c‖θ‖²`. In practice the regulariser is applied as **decoupled weight decay (AdamW)** on conv/linear weights: an L2 term inside plain Adam is rescaled by Adam's per-parameter step sizes and barely regularises the most-updated weights. `optimizer: adam` keeps the literal loss for comparison.

**The loop.** Self-play → `(position, π, z)` triples → replay buffer → gradient steps → stronger network → stronger self-play. No external data at any point.

## 1.3 The proposal's scaling decisions, revisited for GCP

The proposal scaled for a single consumer GPU. With GCP credits the constraint relaxes, and the right response is to spend the extra compute on **simulations per move** first and **network depth** second.

| | AlphaZero | Proposal (3060) | **This plan (GCP)** |
| --- | --- | --- | --- |
| Residual blocks | 20–40 | 4–6 | **8** |
| Channels | 256 | 64–128 | **128** |
| Parameters | ~46M | ~500k | **~2.5M** |
| Simulations/move | 800 | 30–50 | **100** |
| Action space | 4672 | 4096 | **4096** ✓ |

Sims are the higher-value purchase. Each simulation improves the *training target* π, so more sims means every game teaches the network more. Doubling sims is generally worth more than doubling games. We keep a `local.yaml` at the proposal's original scale for development on the 3060.

---

# PART 2 — Gaps Closed

## 2.0 Why this ruleset is the better engineering choice

Worth stating explicitly, because it changes the shape of the whole project versus v1.

**Removing check removes most of the bug surface.** In a chess engine, the top sources of move-generation bugs are, in order: castling legality, en passant, pin detection, and check-evasion filtering. This ruleset deletes all four. Move generation becomes **purely pseudo-legal** — generate every piece's moves and you are done. There is no "does this leave my king attacked" filter, no pin tracking, no discovered-check handling. Realistically this removes 50–60% of the difficulty from Phase 1.

**The state becomes fully Markov.** No castling rights, no en passant square, no repetition counter, no 50-move counter. The board plus side-to-move *is* the state. This is why the encoding drops to 14 planes and why no history planes are needed — a genuinely clean setup that AlphaZero-for-chess does not enjoy.

**Bare-king-loses largely solves the draw problem.** v1's single biggest risk was draw-rate collapse: with only one sliding piece, constructing a mate is hard, so most games would end 0–0 and the value head would learn to output constant zero. Bare-king-loses replaces "construct a mating net" with "win the material war," which is a goal a young network can actually achieve and get gradient signal from. Combined with king-capture termination, **early games will be short and decisive**. This is a large, real improvement.

**The cost:** the `python-chess` perft validation trick from v1 is gone. With no check, no castling, and no en passant there is no shared surface left to validate against. Verification now rests entirely on differential testing (§2.3), which becomes correspondingly more important.

**Net assessment: substantially easier to build, substantially more likely to train successfully, harder to prove correct.**

## 2.1 Gap 1: the rules were never pinned down → **frozen ruleset**

Copy this verbatim into `RULES.md`. Changes require all four students to sign off.

### Board and setup

8×8, files a–h, ranks 1–8. White moves first.

```
8  r  n  b  m  k  b  n  r     Black
7  p  p  p  p  p  p  p  p
6  .  .  .  .  .  .  .  .
5  .  .  .  .  .  .  .  .
4  .  .  .  .  .  .  .  .
3  .  .  .  .  .  .  .  .
2  P  P  P  P  P  P  P  P
1  R  N  B  M  K  B  N  R     White
   a  b  c  d  e  f  g  h

R/r Ratha   N/n Ashva   B/b Gaja
M/m Mantri  K/k Raja    P/p Padati
```

Raja opposes Raja on the e-file; Mantri opposes Mantri on the d-file. Mirror-symmetric.

### Movement

- **Raja** — one square in any of eight directions. **No castling.** May legally move onto a square attacked by an enemy piece.
- **Mantri** — exactly one square diagonally.
- **Ratha** — any number of squares orthogonally, blocked by the first occupied square; captures on it if enemy.
- **Gaja** — exactly two squares diagonally, **jumping**: the intervening square's contents are irrelevant. Cannot move one square. Destination must be empty or enemy-occupied.
- **Ashva** — standard knight leap.
- **Padati** — one square straight forward to an empty square. **No two-square opening move. No en passant.** Captures one square diagonally forward. On reaching the last rank, **promotes to Mantri**, always, with no choice.

All captures are by displacement. There are no restrictions on capturing any piece, including the Raja.

### There is no check

This is the defining rule of this variant and it must be implemented as an *absence*, not as a special case:

- A player **may** move a piece to a square where it can be captured, including the Raja.
- A player **may** leave their Raja attacked, and is under no obligation to respond.
- A player **may** move their Raja onto a currently guarded square. **The opponent may then capture it on their turn, and that ends the game.** The capture is not automatic — the opponent must actually play the capturing move. If they overlook it, play continues normally.
- There is **no checkmate**. The game ends when a Raja is **actually captured**.
- Consequently, **every pseudo-legal move is legal**. `legal_moves()` performs no king-safety filtering whatsoever.

That "not automatic" clause is a real design commitment, not a technicality. The alternative — ending the game the instant a Raja lands on a guarded square — would require exactly the `is_square_attacked()` machinery this ruleset otherwise eliminates, and would change search: king-into-danger would prune as an immediate terminal instead of costing the opponent a move to find. We are **not** doing that.

One training consequence follows, and it is expected rather than a problem: an untrained network will routinely fail to see a free Raja capture, so very early self-play is extremely noisy. Capturing an undefended king is about the most learnable pattern in the game and this self-corrects within the first few iterations. Log a "missed Raja capture" counter during Phase 3 so you can watch it fall — it is a good cheap sanity check that learning is happening at all.

Implementation note for Student 1: do not write an `is_square_attacked()` function and then decline to call it. Do not write it at all. Its existence in the codebase is an invitation for someone to wire it into legality filtering in week 9 and silently change the game.

### Terminal conditions — evaluated in this exact order

Checked **after** each move is applied, from the perspective of the player now to move (call them S):

```
1. RAJA CAPTURED
   S has no Raja on the board          →  S loses        (z = −1)

2. KING vs KING
   Both sides have only their Raja     →  draw           (z =  0)

3. BARE KING
   S has only their Raja, opponent
   has at least one other piece        →  S loses        (z = −1)

4. STALEMATE
   S has no legal move at all          →  S WINS         (z = +1)

5. MOVE CAP  (training device, not a rule)
   ply_count >= 300                    →  adjudicate, §2.2
```

Order matters. Rule 2 must precede rule 3, or King-vs-King resolves as a loss for whoever happens to be to move.

**Rule 2 is unreachable from the start position.** A move never removes the mover's own pieces, and a side reduced to a bare Raja loses under rule 3 the moment it is to move — before the opponent could ever lose its last piece. So King-vs-King, like stalemate, only fires in constructed positions. Keep the ordering test (it is cheap and guards the ladder), but don't expect a non-zero King-vs-King counter.

**On rule 4 — stalemate is a win for the stalemated player.** Under this ruleset "stalemate" and "no legal move" are the same condition, since there is no check and therefore no king-safety filtering: a player is stalemated exactly when every piece they own is completely blocked. Because the Raja may step onto guarded squares and may capture, that requires **every** neighbouring square — 3 in a corner, 5 on an edge, 8 elsewhere — to be occupied by their **own** pieces, each itself blocked. The cheapest construction is a Raja in a corner (e.g. Raja a1 boxed in by its own pieces on a2, b1, b2).

Two things follow.

**It will almost never fire.** The condition is constructible but vanishingly rare in real play. Implement it correctly, put a hand-built position in the test suite, and log a counter. If it never occurs across 400k games, that is the expected result and a reportable finding — not evidence of a bug.

**It is not an escape hatch for a losing player, and the ruleset is self-consistent about this.** Seeking stalemate requires *owning material* — you need enough of your own pieces to box in your own Raja. A player who has been stripped down toward a bare king cannot reach stalemate at all; they hit rule 3 and lose. So the one player who would most want the stalemate win is structurally unable to claim it. That is a pleasant property, and worth a sentence in the report.

**This is the project's one sign trap, and it is a real one.** Rule 4 returns **+1** to the player who cannot move. Every AlphaZero reference implementation you will find online treats a no-legal-move terminal as a loss or a draw, so copying reference code will introduce this bug, and nothing will catch it except an agent that mysteriously fails to improve. Student 2 must write the terminal-sign test before writing the backup, and it must include this case explicitly.

**On rules 1 and 3 together.** These make the game highly decisive. Reducing the opponent to a lone Raja wins outright — no mating technique required. Expect a low draw rate, which is exactly what the training pipeline needs.

**There is no rule 6.** No repetition draw, no 50-move rule, no insufficient-material draw beyond King-vs-King. See §2.2 — this is the one place the chosen ruleset creates a real hazard.

### Deviation summary

| Rule | This project |
| --- | --- |
| Castling | **None** |
| Pawn double-step / en passant | **None** |
| Promotion | **Mantri only, forced** |
| Check / checkmate | **Does not exist** — Raja capture ends the game |
| Raja onto a guarded square | **Legal**; opponent may capture on their turn, not automatically |
| Stalemate | **Win for the side to move** (the stalemated player) |
| Repetition | **No rule** |
| 50-move | **No rule** |
| Bare king | **Loss**, except K-v-K → draw |

## 2.2 Gap 2: termination — **now the sharpest hazard in the project**

In v1 this was a housekeeping item. Under this ruleset it is a genuine risk, and it deserves the emphasis.

**There is no rule that guarantees a game ends.** No repetition draw. No 50-move rule. Two kings and two elephants on separate orbits can shuffle for eternity, and nothing in the ruleset stops them. Worse, a self-play agent that has learned material is bad to lose but has not yet learned how to win will *actively discover* infinite shuffling as a locally optimal policy — it avoids the −1.

**The move cap is therefore not a convenience. It is the sole termination guarantee of the entire system.** Treat it as a hard invariant:

```python
assert ply_count <= MOVE_CAP, "game exceeded cap — termination logic is broken"
```

Set `move_cap = 300` plies. On reaching it, adjudicate on material rather than scoring zero:

```
material = 5·Ratha + 3·Ashva + 1.5·Gaja + 1.5·Mantri + 1·Padati
if |diff| >= 4:  z = sign(diff)   # decisive
else:            z = 0            # draw
```

Adjudication is a **training-only** device. Never use it in arena games or against baselines — those run to a real terminal state or are scored as draws with a note.

**Instrument this from iteration one.** Log the fraction of games ending by cap. Healthy is under 15%. If it climbs above 25% the agent has found the shuffling attractor, and the responses in priority order are: (1) tighten the cap to 200, (2) lower the adjudication margin to 2.0 so more capped games score decisively, (3) raise Dirichlet ε at the root to force exploration out of the loop.

**Ply count is part of the state.** Because adjudication depends on it, the network must be able to see it — hence plane 12 in §3.2. Omitting it makes the value target genuinely unlearnable near the cap.

**Tightening the cap must not change what the network sees.** Plane 12 is normalised by a **fixed** 300 (`ply_count / 300`), not by the live cap. If plane 12 were `ply / move_cap`, response (1) above would silently re-scale the input for every existing checkpoint and every position already in the buffer (an old ply 150 would jump from 0.5 to 0.75). The cap in force is a run-level setting recorded in the config; the network learns where the cap is from the data.

## 2.3 Gap 3: "100% accuracy via unit tests" is not testable as written

Chess engines prove move generation with **perft** — enumerate all leaf nodes at depth N from a fixed position, compare against a published reference. No such reference exists for Chaturanga, and v1's fallback of validating against `python-chess` is gone with the chess rules. So build the ground truth:

**1. Differential testing — now the primary method.**
Student 1 writes the production engine. Student 3 independently writes a deliberately slow, obviously-correct reference engine: naive 8×8 arrays, no optimisation, one function per piece, written from `RULES.md` **without reading S1's code**. Cross-check perft node counts at depths 1–6 from 15 fixed positions.

Depth 6 is affordable here precisely because there is no legality filtering — pseudo-legal generation is fast, and the branching factor is modest with only one sliding piece.

**2. Freeze the agreed numbers** into `tests/perft_reference.json`. That file becomes ground truth and the regression suite. Any later optimisation of the engine must reproduce it exactly.

**3. Position set — cover the pathologies deliberately.** The 15 positions must include: the start position; a position with a Padati one step from promotion; both a bare-king and a King-vs-King position (the latter constructed — it is unreachable from play, §2.1); a hand-built stalemate position, contrived but mandatory — it is the only test that will ever exercise the `+1` terminal; a position where a Raja's only moves are onto guarded squares, confirming they generate as legal; a position where a Gaja's only moves jump over occupied squares; a position with elephants on all eight orbits (§1.1); and an endgame with a lone Ratha.

**4. Property tests.**
- `decode(encode(m)) == m` for every legal move in 10,000 random positions
- `apply` never mutates its input state (the §3.4 contract returns a new state; there is no `undo`)
- `legal_mask` has exactly `len(legal_moves())` bits set
- **Mirror invariance:** `perft(mirror(s), d) == perft(s, d)` for all test positions and depths — this simultaneously validates the movegen symmetry and the augmentation index permutation of §3.5
- **Termination:** every one of 10,000 random playouts terminates, and the terminal reason distribution is sane

**Exit criterion for Phase 1:** two independent implementations agree exactly to depth 6 on all 15 positions.

## 2.4 Gap 4: compute budget

### The account situation, stated plainly

Google Cloud's free-trial terms restrict a customer to one trial; running several in parallel is a terms-of-service matter, and the practical risk lands on the project rather than on the principle. **The failure mode that would actually hurt you is an account suspended mid-run**, taking the VM and any un-synced state with it. Two consequences, both of which are good engineering regardless:

- **Every checkpoint, buffer snapshot, and metrics file goes to a Cloud Storage bucket immediately, never only to VM local disk.** Design for the VM disappearing without warning — which you need anyway, because spot instances get preempted routinely.
- **Keep the local RTX 3060 as the always-available fallback.** If cloud access ends in week 12, tier B in the table below still produces a complete, reportable result.

### The blocker to clear in Week 1, not Week 9

**New GCP accounts have a GPU quota of zero.** You must request `GPUS_ALL_REGIONS` quota before you can create a GPU VM at all, and **free-trial accounts are frequently denied**. The standard path is to upgrade to a paid account — which *retains* the $300 credit — and then request quota. Approval takes anywhere from minutes to several business days.

**Put this on day one of Week 1.** Discovering it in Week 9 costs the project a week of training time it cannot recover.

### Machine shape — the counter-intuitive part

The network is ~2.5M parameters and ~320 MFLOP per forward pass. Even a T4 dispatches a batch of 512 in single-digit milliseconds. **The GPU is not the bottleneck. Python MCTS tree operations are.** So the right machine is **many vCPUs with a modest GPU**, not a big GPU with few cores.

| Machine | GPU | vCPU | On-demand | Spot | Notes |
| --- | --- | --- | --- | --- | --- |
| n1-standard-16 + T4 | T4 16 GB | 16 | ~$1.11/h | **~$0.26/h** | **Recommended.** Best credit-hours per dollar |
| n1-standard-32 + T4 | T4 16 GB | 32 | ~$1.87/h | ~$0.46/h | If profiling shows CPU still saturated |
| g2-standard-16 | L4 24 GB | 16 | ~$1.10/h | ~$0.45/h | Faster GPU, but GPU is not the constraint |
| a2-highgpu-1g | A100 40 GB | 12 | ~$3.70/h | ~$1.20/h | **Do not.** Pays for FLOPs you cannot use |

Prices are indicative for us-central1 and move; check before committing.

### Throughput estimate

```
network evals per game = avg_plies × sims
sims = 100
```

**Game length is not constant across training, and this matters for the budget.** Early on, an untrained agent hangs its Raja almost immediately — expect games of 15–30 plies. As it learns king safety, length grows toward 100–150. Averaged across a full run, assume **~80 plies**, i.e. ~8,000 evaluations per game.

| Configuration | Est. games/hour |
| --- | --- |
| Naive: batch-1, single process | ~25 |
| **Parallel-games batched: 128 concurrent games, 12 workers, AMP** | **~900** |

A ~35× swing on identical hardware. This is why §5 Phase 4 is a gate and not a stretch goal.

### Budget

Assume **600 effective GPU-hours** after setup waste, spot preemption, and quota limits.

| Phase | Hours | Purpose |
| --- | --- | --- |
| 0 — Connect-4 pipeline proof | local 3060 | Costs nothing, burns no credit |
| 1–3 — Development and integration | local 3060 | Do **not** develop on GCP |
| 4 — Performance tuning and machine-shape sweep | 25 | |
| 4b — Cloud soak test | 15 | Preemption and resume validated |
| **5 — Main training** | **450** | The run |
| 6 — Arena, baselines, ablations | 90 | Elo ladder; more arena games than v1 because it is cheap now |
| Reserve | 20 | |
| **Total** | **600** | |

**Expected yield: 450 h × 900 games/h ≈ 400,000 games at 100 simulations.** Plan against a conservative 250,000.

For calibration: DeepMind used 44M games for chess on TPU clusters; published small-game AlphaZero reproductions reach strong play at 10k–100k games. 250k games at 100 sims with a 2.5M-parameter network is a genuinely serious run for a semester project and should comfortably clear "beats a 3-ply alpha-beta baseline."

### Iteration structure

```
1 iteration = 1,000 self-play games   (~65 min)
            + 2,000 gradient steps    (~3 min)
            + arena 60 games every 5th iteration (~5 min amortised)
            ≈ 75 min
450 h ÷ 75 min ≈ 360 iterations ≈ 360,000 games
```

Target **300 iterations** and hold the remainder as headroom.

**Replay ratio.** 2,000 steps × 512 = ~1M samples per iteration against ~80k new positions is a replay ratio of ~13 — each new position is trained on 13 times in its first iteration alone, and the very first iteration would train ~13 epochs on a random network's games. Instead, derive the step count from the data:

```
steps_per_iter = ceil(replay_ratio × new_positions / batch_size)     # replay_ratio ≈ 3
train only once the buffer holds ≥ min_buffer positions             # min_buffer ≈ 50k
```

At ~80k new positions per iteration this is ~470 steps (~1 min), which also shortens the iteration. `steps_per_iter` remains as an upper bound.

### Fallback tiers

Choose your tier at the **Phase 4 gate in Week 8**, from measured throughput — not now.

| Tier | Budget | Configuration |
| --- | --- | --- |
| **A — cloud** | 600 h GCP | 8 blocks / 128 ch, 100 sims, 300 iterations |
| **B — local only** | 150 h on the 3060 | 5 blocks / 64 ch, 40 sims, 200 iterations. Still a full learning curve. |
| **C — minimal** | 20 h | 6×6 board variant, complete pipeline, honest scope note in the report |

Tier B is the plan if GPU quota is denied. It is not a failure — it is v1's plan, which was already sound. Tier C is a legitimate outcome as long as you *choose* it in Week 8 rather than discovering it in Week 15.

### Precision and memory notes

- 16 GB on a T4 is far more than a 2.5M-parameter network needs. Batch 1024 fits easily; VRAM will never constrain you.
- Use **AMP / FP16 for the inference path** (`torch.autocast` + `torch.inference_mode()`). T4 tensor cores give ~2× here, and inference is where nearly all GPU time goes.
- Keep the training step in FP32. FP16 gradients on a network this small buy little and add instability.
- If profiling shows Python tree operations dominating, store node statistics as flat NumPy arrays (`N`, `W`, `P` indexed by edge) rather than one Python object per node. Numba on the inner loop is the escape hatch.

## 2.5 Gap 5: nothing proves the pipeline before the hard game arrives

**Do not debug AlphaZero and Chaturanga simultaneously.** When the agent fails to improve, you will have no way to distinguish a movegen bug from a flipped backup sign from a broken replay buffer from a bad learning rate.

**Phase 0 is mandatory:** get the entire loop working end-to-end on **Connect-4** first. Same `State` interface, same MCTS, same training loop, different game. A correct implementation reaches near-perfect Connect-4 play in under an hour on the local 3060. If it does not, the bug is in your pipeline and you have found it in Week 2 rather than Week 12.

This is the single highest-leverage week of the project and it does not appear in the original milestone list.

---

# PART 3 — Technical Specification

## 3.1 Action space — 4096, clean

```
action = from_square × 64 + to_square        # [0, 4096)
```

Complete, with no special cases, because:
- **no castling** → the Raja never moves two squares
- **no en passant** → a diagonal pawn move is always an ordinary capture
- **promotion is forced and unique** → a Padati reaching the last rank always becomes a Mantri, so the (from, to) pair fully determines the move

This is exactly what the original proposal specified. The authentic ruleset is what makes it correct — v1's chess-aligned promotion choice would have forced 4168.

Illegal actions are masked by setting their logits to `−inf` **before** the softmax, never by zeroing probabilities after it. Post-hoc zeroing silently corrupts the cross-entropy gradient and is a classic, hard-to-find bug.

## 3.2 Input encoding — 14 planes of 8×8

Always **canonical orientation**: the board is flipped so the side to move is always "us," moving up the board. The network then learns one player's game rather than two, roughly halving what it must represent.

| Planes | Contents |
| --- | --- |
| 0–5 | Our Padati, Ashva, Gaja, Ratha, Mantri, Raja |
| 6–11 | Their Padati, Ashva, Gaja, Ratha, Mantri, Raja |
| 12 | `ply_count / 300` (fixed normaliser, not the live cap), constant-filled — **required**, because adjudication depends on it (§2.2) |
| 13 | All ones — gives the CNN an explicit board-edge reference against zero padding |

**No castling planes, no en passant plane, no repetition plane, no 50-move plane, no history planes.** With no check and no repetition rule, the position plus side-to-move is a complete Markov state. AlphaZero stacks eight timesteps mainly to make repetition detectable; we have no repetition rule, so history buys nothing. This is a real simplification the chess version does not get to enjoy.

Note that plane 6–11 must include the enemy Raja even though it can be captured — its position is the single most important feature on the board.

## 3.3 Network — ~2.5M parameters

```
input  14 × 8 × 8
  ↓  conv 3×3, 14→128, BN, ReLU
  ↓  8 × residual block [conv3×3 128→128, BN, ReLU, conv3×3 128→128, BN, +skip, ReLU]
  ├─ policy head
  │     conv 1×1, 128→64  →  reshape (64,8,8) → 4096 logits
  │       interpreted as [to_square][from_rank][from_file]
  │     → mask → log_softmax
  └─ value head
        conv 1×1, 128→8 → flatten 512 → Linear(512→256) → ReLU → Linear(256→1) → tanh
```

**Do not** flatten the trunk into `Linear(8192, 4096)` — that is ~34M parameters in one layer, dwarfing the 2.4M trunk and guaranteeing overfit. The 1×1 convolution above produces the same 4096 outputs from ~8k parameters by reading spatial position as the *from*-square and channel index as the *to*-square. **This is the single most important architectural detail in the plan.**

The implementation assigns the transpose — **channel = from-square, spatial position = to-square** — which is equally valid. Either way each logit is tied to an *absolute* square, so the network cannot share "this is a knight jump" across the board.

**Optional ablation: relative move-type head.** AlphaZero-for-chess indexes the policy as *from-square × move type*. For Chaturanga that is 8 one-step directions (Raja, Mantri, Padati), 8 Ashva jumps, 4 Gaja jumps and 4 × 6 longer Ratha slides = **44 planes × 64 = 2,816 logits**, with no promotion planes because promotion is forced. It is translation-equivariant and typically learns faster. Worth one ablation run in Phase 6; not required.

`local.yaml` uses 5 blocks / 64 channels (~500k params) for development on the 3060.

## 3.4 The interface contract — freeze in Week 1

This is what makes a four-way split work. Agree these signatures before anyone writes real logic, commit stubs the same day, and let all four develop in parallel against them.

```python
# ---- Student 1 ----
class State:
    def legal_moves(self) -> list[int]:            # action ids in [0, 4096); PSEUDO-LEGAL == LEGAL
    def apply(self, action: int) -> "State":       # returns a new state
    def is_terminal(self) -> tuple[bool, float]:   # (done, z from side-to-move's view); §2.1 order
    def encode(self) -> np.ndarray:                # (14, 8, 8) float32, canonical
    def legal_mask(self) -> np.ndarray:            # (4096,) bool
    def key(self) -> Hashable:                     # transposition key (NOT repetition — no such rule)
    def mirror(self) -> tuple["State", np.ndarray] # augmentation: state + action index permutation
    @property
    def side_to_move(self) -> int: ...
    @property
    def ply_count(self) -> int: ...

# ---- Student 3 ----
net(x: Tensor[B, 14, 8, 8]) -> (policy_logits: Tensor[B, 4096],
                                value:         Tensor[B, 1])

# ---- Student 2 ----
mcts.search(state: State, evaluate: Callable, n_sims: int) -> np.ndarray  # (4096,) visit counts

# ---- Student 4 ----
evaluate(states: list[State]) -> tuple[np.ndarray, np.ndarray]  # batched inference across games
```

`mcts.search` takes an **`evaluate` callable**, not a network. That indirection is what lets Student 4 drop in the batched parallel-games inference server in Phase 4 without Student 2 changing a line. It is the most important line in the contract.

## 3.5 Free 2× on data — now available

With castling removed, a **left–right mirror of the board is a legally valid transformation** of the entire game tree. Every piece's movement is mirror-symmetric, so mirroring a position and permuting the policy target indices to match yields a second, equally valid training example.

**Enable mirror augmentation.** In v1 castling made this illegal — king-side and queen-side castling are not mirror images. Dropping castling buys it back for free.

Verify the action-index permutation with the `perft(mirror(s), d) == perft(s, d)` test in §2.3. A subtly wrong permutation poisons training in a way that is very difficult to diagnose from loss curves alone.

## 3.6 Repository layout

```
chaturangazero/
├── RULES.md                     # §2.1, frozen — changes need 4 sign-offs
├── configs/
│   ├── local.yaml               # 5×64, 40 sims — 3060 development
│   ├── cloud.yaml               # 8×128, 100 sims — GCP training
│   └── connect4.yaml            # Phase 0
├── chaturanga/                  # S1
│   ├── board.py  moves.py  encoding.py  terminal.py  perft.py  reference.py
├── search/                      # S2
│   ├── mcts.py  node.py  baselines.py  arena.py
├── net/                         # S3
│   ├── model.py  train.py  replay.py  augment.py
├── pipeline/                    # S4
│   ├── selfplay.py  inference_server.py  loop.py  gcs.py  resume.py  profile.py
├── infra/                       # S4
│   ├── startup.sh  create_vm.sh  Dockerfile
├── eval/                        # S2
│   ├── elo.py  plots.py  play_cli.py
└── tests/
    ├── perft_reference.json  test_rules.py  test_terminal.py
    ├── test_encoding.py  test_mirror.py  test_mcts.py
```

---

# PART 4 — Team Split (4 Students)

Each role owns one architectural boundary, carries a comparable mix of correctness burden, algorithmic difficulty and systems work, and has a Phase-0 slice so nobody is blocked in Weeks 1–2.

### Student 1 — Rules & Game Engine
*The correctness role.*

| | |
| --- | --- |
| **Owns** | `RULES.md`, `chaturanga/` |
| **Deliverables** | Board representation and make/unmake · pseudo-legal move generation for all six pieces · **the terminal ladder of §2.1 in exact order** · move cap enforcement as a hard invariant · the 4096-action codec · `mirror()` and its index permutation · perft harness · the 15-position test suite |
| **Hardest part** | The terminal ladder. Movegen is unusually easy here — no check, no castling, no en passant, no pins — but the ordering of Raja-capture, King-vs-King, bare-king and stalemate is subtle and order-dependent, and a King-vs-King position resolving as a bare-king loss is the kind of bug that quietly corrupts every endgame value in the buffer. Stalemate is also the one condition returning `+1`. Write the ordering and sign tests before the implementation. |
| **Phase 0** | Connect-4 `State` implementation |
| **Exit criterion** | Two independent implementations agree exactly to depth 6 on all 15 positions; 10,000 random playouts all terminate |
| **Report section** | Rule derivation, the no-check design, verification methodology |

### Student 2 — Search & Evaluation
*The algorithmic role.*

| | |
| --- | --- |
| **Owns** | `search/`, `eval/` |
| **Deliverables** | MCTS `Node` with per-edge N/W/Q/P · PUCT selection · expansion and backup with correct sign flipping · Dirichlet root noise (α ≈ 0.3, ε = 0.25) · temperature schedule (τ=1 for 20 plies, then τ→0) · transposition table on `State.key()` · subtree reuse between moves · baselines: random, greedy-material, **3-ply alpha-beta with material + mobility** · arena harness and gating logic · Elo computation from checkpoint round-robin · all plots · human-play CLI |
| **Hardest part** | Backup sign correctness. A flipped sign produces a system that runs cleanly, trains without error, and simply never improves — the worst failure mode in the project. Write the sign test first. Stalemate adds a second, sharper trap: it is the one terminal returning **+1** to the side to move (§2.1), which is the opposite of every AlphaZero reference implementation online. Copying reference code will get it wrong, and because stalemate almost never occurs in real games, no amount of self-play will surface it — only the hand-built unit test will. |
| **Phase 0** | MCTS driving Connect-4 |
| **Exit criterion** | With a perfect hand-written evaluator injected as `evaluate`, MCTS at 200 sims plays Connect-4 optimally from 20 test positions |
| **Report section** | Search algorithm, PUCT behaviour, sims-vs-strength ablation, Elo ladder, final evaluation |

### Student 3 — Network & Training
*The deep-learning role.*

| | |
| --- | --- |
| **Owns** | `net/` |
| **Deliverables** | ResNet trunk and dual heads per §3.3 · masked policy loss (`−inf` before softmax) · combined loss with L2 · Adam, LR schedule, gradient clipping · replay buffer (deque, uniform **position-level** sampling, not game-level) · mirror augmentation with verified permutation · AMP inference path · training diagnostics: policy/value loss split, value-head calibration, policy entropy, buffer age histogram · **the independent reference engine for S1's differential testing** |
| **Hardest part** | Diagnosing a run that is silently not learning. Falling loss is not evidence of improvement — only arena win-rate is. Build the diagnostics before the first long run, not after it fails. |
| **Phase 0** | Network and training loop on Connect-4 |
| **Exit criterion** | Connect-4 agent beats random >95% and greedy >80% after under one GPU-hour |
| **Report section** | Architecture, loss curves, hyperparameter study, augmentation ablation |

### Student 4 — Pipeline, Performance & Cloud
*The systems role.*

| | |
| --- | --- |
| **Owns** | `pipeline/`, `infra/`, `configs/` |
| **Deliverables** | **Parallel-games batched inference server** pooling MCTS leaves across ~128 concurrent games · virtual loss · multiprocessing worker pool · outer training loop and checkpoint management · **GCP: quota request, VM provisioning scripts, container image, GCS checkpoint sync, SIGTERM preemption handler, auto-resume startup script** · config system and seeding for reproducibility · profiling harness with before/after numbers |
| **Hardest part** | Batched leaf evaluation across concurrent trees with virtual loss. This single change decides whether the project trains on 400k games or 10k, and it is genuinely hard concurrency work. Second-hardest: making preemption recovery actually work — a spot VM dying at 3am must resume unattended, or you lose a night of credits every time. |
| **Phase 0** | Orchestration loop and config system on Connect-4 |
| **Exit criterion** | ≥25× games/hour versus the naive Phase-3 baseline, measured; and a VM killed mid-iteration resumes automatically with no data loss |
| **Report section** | System architecture, parallelisation, cloud infrastructure, profiling results |

### Shared work

- `RULES.md` review — all four sign off before Phase 1 begins
- The §3.4 interface contract — agreed and stubbed in Week 1
- Differential testing — S1 × S3, deliberately without reading each other's code
- Phase 3 integration debugging — all four in the room
- Final report — each writes their own section; S4 assembles

---

# PART 5 — Schedule (16 weeks)

## Phase 0 — Spec and pipeline proof · Weeks 1–2 · all four

The most important two weeks in the plan.

- **Day 1: S4 requests GCP GPU quota.** Upgrade to a paid account to retain the credit, then request `GPUS_ALL_REGIONS`. This has multi-day latency and can be denied; everything downstream depends on knowing the answer early.
- Write and ratify `RULES.md` (§2.1), with the terminal ladder written out in order.
- Freeze the §3.4 interface contract. Commit stubs so nobody blocks.
- **Build the whole loop on Connect-4.** Local 3060, small net, 25 sims, one GPU-hour.
- Repo, config system, seeding, CI running the test suite.

**Gate:** Connect-4 agent beats random >95%. If this fails, nothing downstream can work — stop and fix it.

## Phase 1 — Rules engine · Weeks 3–4 · S1 (+ S3 on the reference engine)

- Board representation. Plain arrays; bitboards only if profiling later demands them.
- Pseudo-legal move generation for all six pieces.
- The terminal ladder, in order, with the ordering test written first.
- 4096 codec with round-trip tests; `mirror()` with the permutation test.
- Perft: differential against S3's reference engine to depth 6 on 15 positions.

**Gate:** §2.3 exit criterion met; numbers frozen into `perft_reference.json`.

## Phase 2 — Search and network · Weeks 3–5 · S2, S3 (parallel with Phase 1)

Both work against the Phase-0 Connect-4 game and the frozen stubs. **Neither waits for Phase 1.**

- S2: full MCTS, PUCT, noise, temperature, transposition table, subtree reuse, three baselines.
- S3: architecture per §3.3, masked loss, replay buffer, augmentation, AMP, diagnostics.

**Gate:** S2's MCTS plays Connect-4 optimally with an injected perfect evaluator; S3's Connect-4 agent hits its win-rate targets.

## Phase 3 — Integration · Week 6 · all four

First real Chaturanga self-play with a random-weight network. It will play terribly — hanging its Raja within a dozen moves. That is the correct outcome, and games will be very short.

Assertion-heavy shakedown:
- Every game terminates; the cap assertion never fires
- Terminal-reason histogram is sane: mostly Raja-capture early on
- `π` sums to 1 over legal actions only, zero everywhere else
- Values stay in [−1, +1] and the sign flips correctly between plies
- `z` recorded in the buffer matches the game result from that position's perspective
- No memory growth across 500 games

**Gate:** 500 complete self-play games, zero crashes, zero cap-assertion failures.

## Phase 4 — Performance · Weeks 7–8 · S4 lead, all four

The phase student projects skip and then run out of semester.

- **Parallel-games batched leaf evaluation across ~128 concurrent games — the big one**
- Virtual loss so parallel simulations do not pile into one branch
- Multiprocessing self-play workers; sweep worker count
- Transposition table hit-rate tuning
- AMP inference, `torch.inference_mode()`
- NumPy-array node storage if Python object overhead dominates the profile

Profile before and after; the numbers go in the report. This is real, reportable engineering.

**Gate:** ≥25× games/hour vs Phase 3. **Choose your compute tier here (§2.4).**

## Phase 4b — Cloud deployment and soak · Week 9 · S4

- Container image, VM provisioning script, GCS bucket layout
- Checkpoint sync on every iteration; buffer snapshot every 10
- SIGTERM handler flushing state within the 30-second spot preemption window
- Startup script pulling repo and latest checkpoint, resuming unattended
- **Soak test: deliberately kill the VM mid-iteration three times and confirm clean resume each time**
- Machine-shape sweep: n1-standard-16 vs -32, T4 vs L4, measured in games per dollar

**Gate:** a VM killed at a random moment resumes automatically with no lost work.

## Phase 5 — Main training · Weeks 10–13 · S3 lead, S4 operating

Four weeks of near-continuous running:

```
repeat 300 times:
    generate 1,000 self-play games with the current best net
    train 2,000 gradient steps on the buffer
    every 5th iteration: arena challenger vs incumbent, same sims as self-play
                         promote if score ≥ 55% over 200 games
                         (or an SPRT with early stopping)
```

**Why 200 games.** At 60 games the standard error of the score is ~6.4 points, so a 55% threshold sits less than one standard error above 50% and an *equal-strength* challenger passes ~20–25% of the time. At 200 games the SE is ~3.5 points. The batched arena makes this cheap. Log the 95% confidence interval next to every gate result.

Keep arena gating even though AlphaZero itself dropped it — at this scale training is noisy, and one bad update can collapse the agent. Gating makes that recoverable.

Log every iteration: policy loss, value loss, **mean game length**, **terminal-reason histogram**, **cap-adjudication rate**, draw rate, policy entropy, buffer age. Two of these are the early-warning system: game length climbing toward the cap and cap-rate above 25% both mean the agent has found the shuffling attractor (§2.2).

**Never delete checkpoints** — they are the Elo ladder that becomes the report's headline figure. Sync every one to GCS.

**Gate:** a monotone-ish Elo curve across at least 30 checkpoints, rated from a round-robin over **all** iteration checkpoints, not only promoted ones. (With a gate every 5th of 300 iterations there are at most 60 gates, so "30 promotions" would need a ≥ 50% promotion rate — not a realistic target.)

## Phase 6 — Evaluation · Weeks 14–15 · S2 lead

- Round-robin among checkpoints → fit Elo → **the learning curve**
- Versus baselines: random, greedy-material, 3-ply alpha-beta
- Sims-vs-strength ablation: 10 / 30 / 100 / 400 sims at fixed weights, showing search buys strength independent of training
- Mirror-augmentation ablation
- **Rule-specific analysis, and this is where the project gets interesting:** does the agent learn to exploit Gaja orbit confinement? Does it keep the Raja active given there is no check — and does it learn to *bait* the opponent's Raja onto guarded squares? How often does the bare-king rule decide games versus outright Raja capture, and how does that ratio shift as the agent improves? How quickly did the missed-Raja-capture rate fall to zero? Did stalemate ever fire — and if so, was it deliberate?
- Human-play CLI — a terminal ASCII board is entirely sufficient
- One annotated sample game

## Phase 7 — Write-up · Week 16 · all four

Report, Elo plot, loss curves, profiling table, cloud cost accounting, annotated game, and an honest section on what did not work. That last section is worth more marks than most students expect.

---

# PART 6 — Configuration

`configs/cloud.yaml`. Every run records its exact config and git SHA.

```yaml
rules:
  castling:          false
  pawn_double_step:  false      # implies no en passant
  promotion:         mantri_only
  check_rule:        none       # game ends on Raja capture, played not automatic
  stalemate:         side_to_move_wins   # no legal move  ->  z = +1. See §2.1
  repetition_draw:   false
  fifty_move_rule:   false
  bare_king:         loss       # except K-v-K, which is a draw

net:
  blocks: 8
  channels: 128
  action_space: 4096
  input_planes: 14

mcts:
  sims: 100
  c_puct: 1.5
  dirichlet_alpha: 0.3
  dirichlet_eps: 0.25
  virtual_loss: 3
  reuse_subtree: true

play:
  temp_moves: 20
  move_cap: 300                 # SOLE termination guarantee — see §2.2
  adjudicate_margin: 4.0

train:
  optimizer: adamw              # decoupled weight decay; "adam" = literal c‖θ‖² loss
  weight_decay: 1.0e-2
  lr: 2.0e-3
  lr_milestones: [80, 200]      # iterations
  lr_gamma: 0.1
  batch_size: 512
  l2: 1.0e-4
  buffer_size: 1000000          # positions
  steps_per_iter: 2000          # upper bound; actual = ceil(replay_ratio × new positions / batch)
  replay_ratio: 3.0
  min_buffer: 50000
  grad_clip: 1.0
  amp_inference: true
  mirror_augment: true          # legal now that castling is gone

arena:
  games: 200
  sims: 100                     # same as mcts.sims
  win_threshold: 0.55
  every_n_iters: 5

loop:
  games_per_iter: 1000
  iterations: 300
  parallel_games: 128
  workers: 12

cloud:
  machine: n1-standard-16
  gpu: nvidia-tesla-t4
  spot: true
  bucket: gs://chaturangazero
  checkpoint_every: 1
  buffer_snapshot_every: 10

seed: 20260829
```

---

# PART 7 — Risk Register

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| **Infinite games / shuffling attractor** — no repetition rule, no 50-move rule, so the move cap is the only guarantee | **High** | Fatal to throughput | Cap as a hard assertion (§2.2); material adjudication; monitor cap-rate and mean game length from iteration 1; tighten cap to 200 if cap-rate exceeds 25% |
| **GPU quota denied on the GCP account** | **Medium-high** — common on trial accounts | Drops the project to Tier B | Request on **day 1 of Week 1**. Upgrade to paid to retain the credit. Tier B on the local 3060 is a complete plan. |
| **Value backup sign error** | Medium | Fatal, and silent | S2 writes the sign test before the implementation; Phase-0 Connect-4 catches it in Week 2 |
| **Stalemate sign error** — the `+1` terminal implemented as a loss or draw | **Medium-high** — every online reference has the opposite convention | Silently corrupting, and self-play will never surface it | Hand-built stalemate position in the test suite, written before the backup code. This is the single most important unit test in the project relative to how rarely the condition occurs. |
| **Terminal-ladder ordering bug** — K-v-K resolving as a bare-king loss | Medium | Corrupts endgame values | Ordering test written before the implementation; explicit K-v-K position in the perft set |
| **King-safety filtering creeping back in** — someone adds `is_square_attacked()` and wires it into legality | Medium | Silently changes the game | Do not write the function at all (§2.1); assert that `len(legal_moves())` equals the pseudo-legal count |
| **Spot preemption losing work** | **Certain** — it will happen repeatedly | Lost credits and nights | GCS sync every iteration; SIGTERM handler; unattended resume; validated by the Phase 4b soak test |
| **Phase 4 never happens** | Medium | Caps the run at ~10k games | Phase 4 is a gate with a numeric exit criterion, not a stretch goal |
| **Account suspended mid-run** | Low-medium | Loses the VM, not the work | All state in GCS from the start; local 3060 fallback |
| **Overfitting to a stale buffer** | Medium | Quiet stagnation | 1M-position buffer, position-level sampling, arena gating catches regressions |
| **No-check makes the Raja overvalued as an attacker** | Low | Nothing — it may be correct play | Observe and report; an interesting finding either way |

---

# PART 8 — Deliverables Checklist

**Code**
- [ ] `RULES.md` ratified by four students, terminal ladder written in order
- [ ] Rules engine: pseudo-legal movegen, four-condition terminal ladder plus move cap, move-cap invariant
- [ ] Perft frozen in `perft_reference.json`, two implementations agreeing to depth 6
- [ ] MCTS with PUCT, noise, temperature, transposition table, subtree reuse
- [ ] Dual-head ResNet with masked policy loss
- [ ] Replay buffer, mirror augmentation, training loop, arena gating
- [ ] Parallel-games batched self-play with documented ≥25× speedup
- [ ] GCP provisioning, GCS checkpointing, validated preemption recovery
- [ ] Baselines: random, greedy, 3-ply alpha-beta
- [ ] Human-play CLI
- [ ] Full test suite in CI

**Results**
- [ ] Elo curve across ≥30 checkpoints (round-robin over all iteration checkpoints)
- [ ] Win rates vs all three baselines
- [ ] Loss curves, policy/value split
- [ ] Sims-vs-strength ablation
- [ ] Mirror-augmentation ablation
- [ ] Profiling table before and after Phase 4; games-per-dollar by machine shape
- [ ] Terminal-reason distribution across training — how the game's ending changes as the agent learns
- [ ] One annotated sample game

**Report**
- [ ] Rule derivation and the no-check design rationale
- [ ] Architecture and search description
- [ ] Compute budget: estimated vs actual, in hours and dollars
- [ ] Rule-specific findings: Gaja orbits, Raja activity, bare-king vs capture endings
- [ ] What did not work

---

## The three things that decide this project

1. **GPU quota on day 1 of Week 1.** Everything in Tier A depends on an approval you do not control and cannot rush.
2. **Phase 0** — the entire pipeline proven on Connect-4 before Chaturanga arrives.
3. **Phase 4** — parallel-games batched self-play, worth ~35× in games per hour.

Only one of these appears in the original proposal's milestone list. All three are gates here, with numeric exit criteria, because a project that skips them reaches Week 15 with a correct implementation that has never trained on enough games to show a learning curve.
