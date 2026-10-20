# ChaturangaZero: Architecture, MCTS, and Reinforcement Learning

This document provides a comprehensive mathematical and architectural breakdown of **ChaturangaZero** — an autonomous AlphaZero implementation for 6th-century Chaturanga. It details the neural network architecture, the rationale for dual heads, Monte Carlo Tree Search (MCTS), the self-play reinforcement learning loop, the composite loss function, and high-throughput batched systems engineering.

---

## 1. Mathematical Formulation & Representation

ChaturangaZero models Chaturanga as a discrete, deterministic, two-player zero-sum Markov Decision Process (MDP) with perfect information:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma \rangle$$

* **State Space $\mathcal{S}$:** Fully observable configurations on the $8 \times 8$ uncheckered Ashtapada board.
* **Action Space $\mathcal{A}$:** Discrete space of $|\mathcal{A}| = 4,096$ possible moves.
* **Transition Function $\mathcal{T}(s, a)$:** Deterministic transition to the next state $s'$.
* **Reward $\mathcal{R}$:** Terminal only, where $\mathcal{R} \in \{-1, 0, +1\}$ from the perspective of the player to move.
* **Discount Factor $\gamma = 1$:** Undiscounted, as all credit assignment is linked to the final game outcome.

---

### 1.1 The 14-Plane State Tensor $\mathbf{s} \in \mathbb{R}^{14 \times 8 \times 8}$

In standard chess AlphaZero, DeepMind used 119 planes because chess is non-Markovian (requiring past move history to track 3-fold repetition, 50-move draw clocks, and en-passant/castling privileges).

Under authentic Chaturanga rules:
1. No castling.
2. No double-step pawn leap and no en-passant.
3. No 3-fold repetition rule.

Consequently, the game state is **strictly 1st-order Markovian**: the future state distribution depends *only* on the current board. The full board is completely described by a **14-channel binary/numerical tensor of shape $(14 \times 8 \times 8)$**:

| Channel Indices | Feature Representation | Value Type | Description |
| :---: | :--- | :---: | :--- |
| **Planes $0 - 5$** | Friendly Pieces (Padati, Ashva, Gaja, Ratha, Mantri, Raja) | $\{0, 1\}$ | Binary occupancy of our 6 piece types |
| **Planes $6 - 11$** | Enemy Pieces (Padati, Ashva, Gaja, Ratha, Mantri, Raja) | $\{0, 1\}$ | Binary occupancy of opponent's 6 piece types |
| **Plane $12$** | Normalized Game Clock: $\frac{\text{ply\_count}}{300}$ | $[0, 1]$ | Progress toward the 300-ply adjudication limit |
| **Plane $13$** | Boundary Reference Plane | $\{1\}$ | Uniform constant 1s across all squares |

#### Perspective Invariance (Canonical Flipping)
The state is always encoded from the perspective of the **side to move**:
* If Black is to move, the board is flipped vertically ($r \leftarrow 7 - r$).
* Friendly pieces are always in planes $0-5$, moving upwards towards rank 7.
* This ensures the neural network learns a single, universal perspective rather than learning separate strategies for White and Black.

#### Horizontal Reflection Invariance (Data Augmentation)
Chaturanga rules are horizontally symmetric. A board state reflected across the vertical axis (file $f \leftarrow 7 - f$) preserves all legal transitions and game outcomes:
$$s_{\text{mirrored}} = \text{flip\_files}(s)$$
$$\pi_{\text{mirrored}} = \mathcal{P}(\pi)$$
Where $\mathcal{P}$ is the permutation matrix mapping action $a = (\text{from} \times 64 + \text{to})$ to its reflected counterpart. This property allows us to **double our training data ($2\times$)** without playing additional games.

---

### 1.2 Action Space Codec ($|\mathcal{A}| = 4,096$)

Every move is parameterized by a source square and a destination square on the $8 \times 8$ board:
$$\text{from} \in [0, 63], \quad \text{to} \in [0, 63]$$
$$\text{action\_id} = \text{from} \times 64 + \text{to}$$
Because pawn promotion is forced strictly to Mantri, there are no underpromotion choices. Thus, $|\mathcal{A}| = 64 \times 64 = 4,096$ encompasses the entire universe of possible moves.

---

### 1.3 Terminal Rules and the Move Cap

Checked after every move, from the perspective of the side to move $S$, **in this order** (`RULES.md`, `chaturanga/terminal.py`):

| # | Condition | Result for $S$ |
|---|---|---|
| 1 | $S$ has no Raja (it was captured) | loss, $z = -1$ |
| 2 | Both sides have only their Raja | draw, $z = 0$ |
| 3 | $S$ has only its Raja, the opponent has more | loss, $z = -1$ |
| 4 | $S$ has **no legal move** (stalemate) | **win, $z = +1$**: the opposite of most AlphaZero code, guarded by a unit test |
| 5 | ply $\ge 300$ (move cap) | training: material adjudication (margin 4). Arena/evaluation: draw |

There is no check, so a Raja may step onto a guarded square; it is only lost when the opponent actually captures it. There is no repetition or 50-move rule, so the cap is the **only** guarantee that a game ends, and it is enforced as a hard assertion. The fraction of games reaching the cap is logged every iteration: above ~25% means the agent has found a "shuffling" loop. Plane 12 (ply / 300, a fixed scale so that tightening the cap doesn't re-scale the input) is in the input because adjudication depends on it.

---

## 2. ResNet Architecture Deep-Dive

The core neural network $f_\theta(s)$ is a **Dual-Head Deep Residual Convolutional Neural Network** (~2.5 million parameters).

```
                      Input Tensor: (B, 14, 8, 8)
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │              Conv Stem               │
               │  Conv2d(14 → 128, kernel=3, pad=1)   │
               │  BatchNorm2d(128)                    │
               │  ReLU                                │
               └──────────────────┬───────────────────┘
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │         8 × Residual Blocks          │
               │  ┌────────────────────────────────┐  │
               │  │  Conv2d(128 → 128, 3×3, pad=1) │  │
               │  │  BatchNorm2d(128)              │  │
               │  │  ReLU                          │  │
               │  │  Conv2d(128 → 128, 3×3, pad=1) │  │
               │  │  BatchNorm2d(128)              │  │
               │  └───────────────┬────────────────┘  │
               │                  │ + Skip Connection │
               │                  ▼                   │
               │                 ReLU                 │
               └──────────────────┬───────────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
     ┌─────────────────────────┐     ┌─────────────────────────┐
     │       POLICY HEAD       │     │       VALUE HEAD        │
     │ Conv2d(128 → 64, 1×1)   │     │ Conv2d(128 → 8, 1×1)    │
     │ BatchNorm2d(64)         │     │ BatchNorm2d(8)          │
     │ Flatten: (64×8×8)=4,096 │     │ Flatten: (8×8×8)=512    │
     │ Mask Illegal Moves (-∞) │     │ Linear(512 → 256), ReLU │
     │ Log-Softmax             │     │ Linear(256 → 1), Tanh   │
     │ Log-probabilities log p │     │ Scalar v ∈ [-1, +1]     │
     └─────────────────────────┘     └─────────────────────────┘
```

### 2.1 The Residual Block Mechanism
Each residual block computes:
$$\mathbf{x}_{l+1} = \text{ReLU}\left( \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l) \right)$$
Where $\mathcal{F}$ consists of two $3 \times 3$ convolutions with Batch Normalization:
$$\mathcal{F}(\mathbf{x}_l) = \text{BN}_2\left(\text{Conv}_2\left(\text{ReLU}\left(\text{BN}_1\left(\text{Conv}_1(\mathbf{x}_l)\right)\right)\right)\right)$$

#### Why Residual Blocks?
1. **Vanishing Gradient Elimination:** Standard deep CNNs degrade because backpropagated gradients shrink exponentially through repeated matrix multiplications. The additive identity connection $\mathbf{x}_l$ creates a direct gradient highway:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_{l+1}} \left( \mathbf{I} + \frac{\partial \mathcal{F}}{\partial \mathbf{x}_l} \right)$$
   Even if $\frac{\partial \mathcal{F}}{\partial \mathbf{x}_l}$ approaches zero, gradients still flow back through $\mathbf{I}$.
2. **Feature Preservation:** The network can learn complex tactical patterns while preserving raw spatial piece positions across deep layers.

---

### 2.2 Why We Need a Dual-Head Architecture

Instead of training two separate networks (one for choosing moves and one for evaluating positions), we use a single residual trunk with two task-specific heads:
$$f_\theta(s) = (\mathbf{p}, v)$$

#### 1. Shared Feature Representation
Evaluating a position ($v$) and choosing candidate moves ($\mathbf{p}$) rely on identical fundamental features:
* Piece placement and material balance.
* Piece mobility and lines of attack. (There are no pins: Chaturanga has no check, so a piece is never forbidden to move.)
* King exposure and attack geometry.

Sharing the 8-block trunk forces the network to learn rich, generalized representations of Chaturanga geometry that serve both evaluation and tactical selection simultaneously.

#### 2. Mutual Regularization
* If the value head alone were trained, it would suffer from reward sparsity (only winning or losing at the end of the game).
* The policy head receives rich, per-move supervision from MCTS visit distributions $\boldsymbol{\pi}$ at every single turn.
* **Policy gradients stabilize and regularize the value trunk**, while value gradients prevent the policy from hallucinating tactically reckless moves.

#### 3. Half the Inference Latency
In Monte Carlo Tree Search, every expanded leaf requires both prior probabilities $\mathbf{p}$ and an evaluation $v$. A dual-head architecture evaluates both in a **single GPU forward pass**, cutting inference latency by $\approx 50\%$ compared to two separate networks.

---

### 2.3 The $1 \times 1$ Convolution Policy Head Innovation

A naive implementation of the policy head flattens the trunk features $(128 \times 8 \times 8 = 8,192)$ and passes them through a fully connected dense layer to $4,096$ outputs:
$$\text{Linear}(8192 \to 4096): \quad 8,192 \times 4,096 \approx \mathbf{33,554,432} \text{ parameters}$$

This creates catastrophic problems:
* **Overparameterization:** A 33.5M parameter layer dwarfs the entire 2.5M trunk, causing extreme overfitting on self-play data.
* **Destruction of Spatial Locality:** A dense layer discards the 2D geometry of the board.

#### Our Engineered Solution:
We use a **$1 \times 1$ convolution** reducing channels from $128 \to 64$:
$$\text{Conv2d}(128 \to 64, \text{kernel}=1): \quad 128 \times 64 \times 1 \times 1 = \mathbf{8,192} \text{ parameters}$$
Since $64 \times 8 \times 8 = 4,096$, the output tensor of shape $(B, 64, 8, 8)$ is simply reshaped into $(B, 4096)$!
* The reshape is channel-major, so logit index $c \cdot 64 + (8r + f)$ is action `from_sq * 64 + to_sq`:
* The channel index $c \in [0, 63]$ indexes the **source** square (`from_sq = c`).
* The spatial position $(r, f)$ indexes the **destination** square (`to_sq = r * 8 + f`).
* (Plan §3.3 describes the transposed assignment. Either works, because every logit has its own 1×1 filter; this is what `net/model.py` actually does.)
* **Parameter reduction:** $33.5\text{M} \to 8.2\text{k}$ (**$4000\times$ fewer parameters**), with zero spatial information loss!

#### Illegal Move Masking:
Before the softmax layer, illegal actions are masked to $-\infty$:
$$\text{logits}_{\text{masked}}(a) = \begin{cases} \text{logits}(a) & \text{if } a \in \mathcal{A}_{\text{legal}}(s) \\ -\infty & \text{otherwise} \end{cases}$$
$$p(a) = \frac{\exp(\text{logits}_{\text{masked}}(a))}{\sum_{b \in \mathcal{A}_{\text{legal}}(s)} \exp(\text{logits}_{\text{masked}}(b))}$$
This guarantees that illegal actions receive exactly $0$ probability, preventing gradient waste on impossible moves.

---

## 3. Monte Carlo Tree Search (MCTS)

MCTS is the policy improvement operator. While the raw neural network provides heuristic intuition, MCTS conducts explicit lookahead search to compute tactical depth.

Each search edge $(s, a)$ in the search tree maintains four statistics:
1. $N(s, a)$: Visit count.
2. $W(s, a)$: Total accumulated action value.
3. $Q(s, a) = \frac{W(s, a)}{N(s, a)}$: Mean action value (expected payoff).
4. $P(s, a)$: Prior probability provided by the neural network's policy head.

---

### 3.1 The 4 Steps of MCTS Simulation

For every game turn, MCTS executes $M = 100$ simulations. Each simulation traverses the tree from root to leaf:

```
        (Root State s_0)
              │
    1. SELECT │ (PUCT formula balances Q and P)
              ▼
           (Node s_t)
              │
    2. EXPAND │ (Create leaf node for unvisited action)
              ▼
        (Leaf Node s_L)
              │
  3. EVALUATE │ (ResNet forward pass: outputs P and v)
              ▼
   4. BACKUP  ▲ (Propagate -v up the tree; update N and W)
```

#### Step 1: Selection (The PUCT Algorithm)
From root $s_0$, the search steps down the tree by picking actions that maximize the **Predictor Upper Confidence Bound applied to Trees (PUCT)** formula:
$$a_t = \arg\max_{a} \left[ Q(s, a) + U(s, a) \right]$$
Where the exploration term $U(s, a)$ is defined as:
$$U(s, a) = c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_{b} N(s, b)}}{1 + N(s, a)}$$

* **Exploitation term $Q(s, a)$:** Guides the search toward moves that proved successful in previous simulations.
* **Exploration term $U(s, a)$:** Encourages exploring unvisited moves ($N(s, a) \to 0$) that have a high network prior $P(s, a)$. As an action is explored repeatedly, the denominator $1 + N(s, a)$ grows, forcing the tree to explore other candidate branches.
* **Constant $c_{\text{puct}} = 1.5$:** Controls the balance between exploitation and exploration.

#### Dirichlet Noise at Root (Exploration Injection)
To ensure the search explores diverse opening options during self-play, Dirichlet noise is blended into the root node's prior probabilities:
$$P(s_{\text{root}}, a) \leftarrow (1 - \epsilon) P(s_{\text{root}}, a) + \epsilon \cdot \eta_a, \quad \boldsymbol{\eta} \sim \text{Dir}(\alpha)$$
* Parameters: $\alpha = 0.3$, $\epsilon = 0.25$.
* This prevents search paths from prematurely collapsing to a single move.

#### Step 2: Expansion
When selection reaches a state $s_L$ with unvisited actions, a new child node is allocated and initialized with $N(s_L, a) = 0, W(s_L, a) = 0, Q(s_L, a) = 0$.

#### Step 3: Evaluation
Instead of conducting expensive and inaccurate random rollouts to the end of the game, **the neural network evaluates the leaf node directly**:
$$(P(s_L, \cdot), v) = f_\theta(s_L)$$
* $P(s_L, \cdot)$ initializes the prior distribution for the child node.
* $v \in [-1, +1]$ serves as the leaf evaluation from the perspective of the player to move at $s_L$.
* If $s_L$ is terminal, the true game outcome $z \in \{-1, 0, +1\}$ replaces $v$.

#### Step 4: Backup (Minimax Backpropagation)
The evaluation $v$ is propagated back along the traversed trajectory $(s_0, a_0, s_1, a_1, \dots, s_L)$.
Because Chaturanga is a zero-sum two-player game, **the reward sign alternates at each ply**:
$$v_t = -v_{t+1}$$
For each edge on the search path:
$$N(s_t, a_t) \leftarrow N(s_t, a_t) + 1$$
$$W(s_t, a_t) \leftarrow W(s_t, a_t) + v_t$$
$$Q(s_t, a_t) \leftarrow \frac{W(s_t, a_t)}{N(s_t, a_t)}$$

---

### 3.2 Action Selection via Temperature Scheduling

After $M = 100$ simulations, the root visit count distribution forms a search-improved policy $\boldsymbol{\pi}$:
$$\pi(a) = \frac{N(s_{\text{root}}, a)^{1/\tau}}{\sum_{b} N(s_{\text{root}}, b)^{1/\tau}}$$
Where $\tau$ is the exploration temperature:
* **Opening Plies ($\text{ply} \le 20$):** We set $\tau = 1.0$. The agent samples actions stochastically proportional to visit counts:
  $$a \sim \boldsymbol{\pi}$$
  This ensures wide game diversity in the training buffer.
* **Late Game ($\text{ply} > 20$):** We anneal temperature $\tau \to 0$. The agent selects the greedy maximum-visit action:
  $$a = \arg\max_{a} N(s_{\text{root}}, a)$$
  This enforces disciplined endgame play.

---

## 4. The Self-Play Reinforcement Learning Loop

The training loop implements **Generalized Policy Iteration (GPI)**:

```
    ┌────────────────────────────────────────────────────────┐
    │                      SELF-PLAY                         │
    │  DualHeadNet f_θ guides MCTS across 128 parallel games │
    │  Records experience triples: (s_t, π_t, z)             │
    └───────────────────────────┬────────────────────────────┘
                                │
                                ▼
    ┌────────────────────────────────────────────────────────┐
    │                     REPLAY BUFFER                      │
    │  Capacity: 1,000,000 position-level transitions        │
    │  Augmented via horizontal mirror reflection (2×)       │
    └───────────────────────────┬────────────────────────────┘
                                │ Sample Mini-Batches (B=512 cloud / 256 local)
                                ▼
    ┌────────────────────────────────────────────────────────┐
    │                   NETWORK TRAINING                     │
    │  Minimize: L(θ) = MSE(z, v) + CrossEntropy(π, p) + L2  │
    │  Optimizer: AdamW (lr=2e-3, decoupled decay 1e-2)      │
    └───────────────────────────┬────────────────────────────┘
                                │ Produces Challenger θ_new
                                ▼
    ┌────────────────────────────────────────────────────────┐
    │                      ARENA GATE                        │
    │  Every 5th iteration: challenger vs incumbent, 60 games│
    │  Promote if score ≥ 55%; the incumbent plays self-play │
    └────────────────────────────────────────────────────────┘
```

### 4.1 Data Collection Triples
During self-play, at every move $t$, the system records:
$$(s_t, \boldsymbol{\pi}_t, z)$$
* $s_t \in \mathbb{R}^{14 \times 8 \times 8}$: Board state tensor.
* $\boldsymbol{\pi}_t \in \mathbb{R}^{4096}$: Normalized MCTS visit count distribution.
* $z \in \{-1, 0, +1\}$: True final game outcome from the perspective of the player to move at step $t$.

---

### 4.2 The Composite Loss Function

Parameters $\theta$ are optimized by minimizing the multi-objective loss:
$$\mathcal{L}(\theta) = \mathcal{L}_{\text{value}} + \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{reg}}$$
$$\mathcal{L}(\theta) = (z - v)^2 - \sum_{a=1}^{4096} \pi(a) \log p(a) + c \|\theta\|^2$$

#### 1. Value Loss: Mean Squared Error (MSE)
$$\mathcal{L}_{\text{value}} = (z - v)^2$$
* Compares the scalar prediction $v \in [-1, 1]$ against the actual terminal game winner $z$.
* Penalizes poor position assessment and drives the value head toward the true game-theoretic minimax value.

#### 2. Policy Loss: Categorical Cross-Entropy
$$\mathcal{L}_{\text{policy}} = -\sum_{a} \pi(a) \log p(a) = -\boldsymbol{\pi}^\top \log \mathbf{p}$$
* $\boldsymbol{\pi}$ is the target distribution from MCTS visit counts.
* $\mathbf{p}$ is the network's raw output probability vector.
* Minimizing cross-entropy is mathematically equivalent to minimizing the **Kullback-Leibler (KL) Divergence** between the network policy and the search policy:
  $$D_{\text{KL}}(\boldsymbol{\pi} \parallel \mathbf{p}) = \sum_a \pi(a) \log \frac{\pi(a)}{p(a)} = -\sum_a \pi(a) \log p(a) - \mathcal{H}(\boldsymbol{\pi})$$
  Because the entropy of the search distribution $\mathcal{H}(\boldsymbol{\pi})$ is constant with respect to $\theta$, minimizing cross-entropy directly forces the raw network to mimic the lookahead search!

#### 3. Weight Regularization ($L_2$)
$$\mathcal{L}_{\text{reg}} = c \|\theta\|^2 = c \sum_i w_i^2$$
* Default: **AdamW** with decoupled weight decay (`train.weight_decay`, 1e-2) on conv/linear weights only (not BatchNorm or biases). Putting $c\|\theta\|^2$ in the loss, or using Adam's `weight_decay` (equivalent), gets rescaled by Adam's per-parameter step sizes, so the most-updated weights are barely regularised. AdamW avoids that.
* `optimizer: adam` uses the formula above literally: the $c\|\theta\|^2$ term (`train.l2`, 1e-4) is added to the loss and logged as `l2_loss`.
* Guards against overfitting to early self-play trajectories.

---

### 4.3 Why Does Reinforcement Learning Converge? (The Distillation Cycle)

The AlphaZero learning cycle forms a **virtuous policy iteration flywheel**:
1. **MCTS is a Policy Improver:** Lookahead search with the network's own priors and values usually produces a visit distribution $\boldsymbol{\pi}$ that is stronger than the raw policy $\mathbf{p}$. This holds in practice, not as a theorem, and it is weakest when the value head is still poor.
2. **Supervised Distillation:** Training the ResNet on $(s, \boldsymbol{\pi})$ distills the tactical lookahead of MCTS into the network's convolutional parameters.
3. **Compound Strength:** In the next iteration, MCTS starts with a stronger baseline policy. Search explores deeper, more complex tactical lines, generating an even higher-quality target $\boldsymbol{\pi}'$.
4. **Iterated improvement:** Repeated over many iterations, the loop improves play without any human demonstrations. There is no guarantee of reaching optimal play. Progress is measured, not assumed: by the Elo ladder across checkpoints and by win rates against the random, greedy and 3-ply alpha-beta baselines (the plan's success criteria).

---

## 5. High-Throughput Systems Engineering: Batched Inference

In naive AlphaZero implementations, each MCTS worker runs its own neural network evaluation sequentially on the GPU:
$$\text{Latency} \approx M \times T \times \text{forward\_pass\_time}$$
For 100 simulations across 80 plies, a single game requires 8,000 sequential GPU kernel launches. Because batch size is 1, the GPU's Tensor Cores are severely under-utilized ($< 5\%$ compute capacity).

### 5.1 Parallel-games batched self-play (`pipeline/inference_server.py`)

The server is **single-threaded and lockstep**, not a queue of worker threads: one Python loop drives every concurrent game.

```
for each move:
    for each simulation step:
        for each active game (G of them):
            descend its tree with PUCT up to `leaves_per_step` times
            (virtual loss on each pending path so the descents spread out)
            leaf already terminal / in the evaluation cache?  -> back up now, no GPU
            otherwise                                         -> queue the leaf
        ONE forward pass on all queued leaves  (batch ≈ G × leaves_per_step, FP16 AMP on CUDA)
        expand + back up every queued leaf; revert virtual loss
    every game picks its move; finished games are recycled with fresh roots,
    and all new roots are expanded in one more batched pass
```

Pieces:

1. **Lockstep batching:** `loop.parallel_games` games share each forward pass.
2. **Virtual loss** (`mcts.leaves_per_step`, `mcts.virtual_loss`): more than one leaf per game per batch keeps the batch large when few games remain. The default of 1 is plain MCTS.
3. **Evaluation cache** (`search/cache.py`, `mcts.eval_cache_size`): positions that recur within a run (transpositions, shared openings across games) skip the network. Each tree keeps its own visit statistics.
4. **Worker processes** (`pipeline/workers.py`, `loop.workers`): the tree work is Python on one core, so self-play is split across processes. Each runs its own batched server and returns a compact replay buffer.
5. **Batched arena** (`run_batched_arena`): gating games also run concurrently, with one forward pass per network per simulation step.

### 5.2 Performance: what is and isn't measured

**Not yet measured:** end-to-end games/hour on a GPU, and GPU utilisation. Earlier versions of this section showed "~35×" and "650 games/hour". Those were the planning estimates from `IMPLEMENTATION_PLAN.md` §2.4, not measurements, and have been removed. Measure them on the training machine with a short self-play run before choosing a compute tier (Phase 4 gate).

**Measured:** the pure-Python core (no network), with `python tools/bench_core.py` on the development CPU, before and after the needtofix S1–S7 optimisations:

| Core metric (pure Python, no network) | Before | After | Speed-up |
| :--- | ---: | ---: | :---: |
| Legal-move generation (positions/s) | 16.7k | 83.2k | 4–7× |
| Terminal check (positions/s) | 12.7k | 41.0k | 3–4× |
| Board encoding (positions/s) | 43.0k | 90.0k | 2–3× |
| **MCTS simulations/s** (dummy evaluator) | 3,050 | 9,755 | **2.4–5.9×** |
| **Memory per MCTS node** | 50.7 KB | 2.3 KB | **22× smaller** |
| `perft(start, 4)` | 2.04 s | 0.12 s | 12–17× |

Best of two runs each, on a laptop CPU whose timings vary about 2× between runs. The speed-up column is the range of the per-run ratios. Search results are unchanged: identical MCTS visit counts on 40 test positions, identical move lists on 3,000, byte-identical encodings, and production perft still matches the frozen depth-5/6 reference numbers. These core numbers are an upper bound on the self-play speed-up, which also includes network time. Measure games/hour on the training machine.

---

## 6. Summary of Architectural Constants

| Hyperparameter | Symbol | Value | Rationale |
| :--- | :---: | :---: | :--- |
| **Input Shape** | $(C, H, W)$ | $14 \times 8 \times 8$ | Markovian state: 6 friendly + 6 enemy + clock + border |
| **Action Space** | $|\mathcal{A}|$ | 4,096 | Flat $64 \times 64$ (`from_sq * 64 + to_sq`) |
| **Residual Blocks** | $N$ | 8 | Sufficient depth for tactical patterns; fast inference |
| **Trunk Channels** | $C$ | 128 | High feature capacity without GPU memory bloat |
| **Policy Head Channels** | $C_{\text{policy}}$ | 64 | $1\times1$ Conv yields $64 \times 64 = 4,096$ with only ~8.2k weights |
| **Value Head Conv** | $C_{\text{value}}$ | 8 | $1\times1$ Conv compressing to 512 dense inputs |
| **MCTS Simulations** | $M$ | 100 | Balances search quality with self-play speed |
| **PUCT Constant** | $c_{\text{puct}}$ | 1.5 | Governs exploration vs. exploitation |
| **Dirichlet Noise** | $(\alpha, \epsilon)$ | $(0.3, 0.25)$ | Injects root exploration to diversify self-play |
| **Temperature** | $\tau$ | $1.0 \to 0.0$ | Stochastic in first 20 plies; greedy thereafter |
| **Replay Buffer Size** | $|\mathcal{B}|$ | $1,000,000$ | Uniform position-level sampling with mirror symmetry |
| **Learning Rate** | $\eta$ | $2 \times 10^{-3}$ | AdamW, gradient clipping 1.0, ×0.1 at iterations 80 and 200 |
| **Weight Decay** | — | $1 \times 10^{-2}$ | AdamW decoupled decay on conv/linear weights (`optimizer: adam` uses $c = 10^{-4}$ in the loss instead) |
| **Batch Size** | $B$ | 512 / 256 | cloud / local |
| **Move Cap** | — | 300 plies | Sole termination guarantee; training games adjudicated on material (margin 4) |
