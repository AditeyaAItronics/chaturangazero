"""
Parallel-Games Batched Inference & Self-Play Server (Phase 4).

Achieves ~25x-35x speedup by pooling leaf evaluations across concurrent games
into a single batched GPU forward pass with AMP (FP16) and torch.inference_mode().

Instead of:
    Game 1: traverse -> forward(batch_size=1) -> backup ... (GPU starved at 2% TDP)
We do:
    Games 1..B: traverse all trees -> forward(batch_size=B) -> backup all (GPU saturated)
"""

import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable, Optional, Type

from search.node import Node
from search.mcts import _select_edge, _backup, select_action_with_temperature
from search.cache import EvalCache, apply_entry
from net.replay import ReplayBuffer
from utils.device import get_device, use_amp


class BatchedEvaluator:
    """
    Batched neural network evaluator with AMP (FP16) and inference_mode.
    
    Can evaluate a batch of states in a single forward pass.
    """
    def __init__(self, net: torch.nn.Module, device=None, amp: bool = True):
        self.device = get_device(device)
        self.net = net.to(self.device)
        self.amp = use_amp(self.device, amp)
        self.net.eval()

    def evaluate_batch(self, states: list) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a list of game states in a single batched forward pass.
        
        Args:
            states: list of game states (Chaturanga State or Connect4State).
            
        Returns:
            policies: np.ndarray of shape (B, action_size) with softmax probabilities.
            values: np.ndarray of shape (B,) with scalar values in [-1, +1].
        """
        if not states:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)

        # Vectorized encoding across all states in the batch
        encodings = np.stack([s.encode() for s in states], axis=0)
        x = torch.from_numpy(encodings).to(self.device, non_blocking=self.device.type == 'cuda')

        with torch.inference_mode():
            if self.amp:
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    policy_logits, values = self.net(x)
            else:
                policy_logits, values = self.net(x)

            policies = F.softmax(policy_logits, dim=-1).cpu().numpy()
            vals = values.squeeze(-1).cpu().numpy()

        # Handle scalar output if B=1
        if vals.ndim == 0:
            vals = np.array([vals.item()], dtype=np.float32)

        # Network output is in the canonical frame; MCTS works in absolute actions
        for i, s in enumerate(states):
            if hasattr(s, 'orient_policy'):
                policies[i] = s.orient_policy(policies[i])

        return policies, vals

    def __call__(self, state) -> tuple[np.ndarray, float]:
        """Single-state evaluation for compatibility with sequential MCTS."""
        policies, vals = self.evaluate_batch([state])
        return policies[0], float(vals[0])


class GameSlot:
    """Tracks state and trajectory for one concurrent game in the batch."""
    def __init__(self, state_cls: Type, action_size: int):
        self.state_cls = state_cls
        self.action_size = action_size
        self.reset()

    def reset(self):
        self.state = self.state_cls()
        self.history = []  # list of (encoded_state, pi, side_to_move)
        self.root = None
        self.done = False
        self.outcome = 0.0
        self.terminal_reason = 'ongoing'


def run_batched_self_play(
    net: torch.nn.Module,
    state_cls: Type,
    total_games: int,
    num_parallel_games: int = 64,
    n_sims: int = 40,
    action_size: int = 4096,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    temp_threshold: int = 15,
    device=None,
    amp: bool = True,
    buffer: Optional[ReplayBuffer] = None,
    reuse_subtree: bool = True,
    leaves_per_step: int = 1,
    virtual_loss: float = 3.0,
    eval_cache_size: int = 50_000,
    fpu_reduction: Optional[float] = None,
) -> dict:
    """
    Generate self-play games using parallel-games batched MCTS.
    
    Args:
        net: Neural network.
        state_cls: State class (Chaturanga State or Connect4State).
        total_games: Total number of games to generate.
        num_parallel_games: Number of concurrent games in the batch (e.g. 32, 64, 128).
        n_sims: Number of MCTS simulations per move.
        action_size: 4096 for Chaturanga, 7 for Connect-4.
        c_puct: Exploration constant.
        dirichlet_alpha: Dirichlet noise parameter.
        dirichlet_eps: Weight of Dirichlet noise at root.
        temp_threshold: Number of plies with temperature=1.0.
        device: 'auto', 'cpu', 'cuda', 'cuda:N' or None (auto-detect).
        amp: Whether to use FP16 automatic mixed precision.
        buffer: Optional ReplayBuffer to add training examples directly.
        reuse_subtree: keep the chosen child's subtree as the next root (config mcts.reuse_subtree).
        leaves_per_step: leaves each game contributes to one network batch. With
            more than 1, virtual loss steers the extra descents onto different
            branches; the batch grows to games x leaves_per_step, which keeps the
            GPU fed when few games remain. 1 reproduces plain sequential MCTS.
        virtual_loss: visits-worth of temporary loss applied along a pending path.
        eval_cache_size: entries in the shared evaluation cache (transposition
            table); 0 disables it.
        fpu_reduction: value of unvisited edges (search.mcts._select_edge); None = 0.
        
    Returns:
        dict of run statistics (games, wins, losses, draws, mean_length, elapsed_time).
    """
    evaluator = BatchedEvaluator(net, device=device, amp=amp)
    batch_size = min(num_parallel_games, total_games)
    slots = [GameSlot(state_cls, action_size) for _ in range(batch_size)]

    games_completed = 0
    games_started = batch_size

    stats = {
        'games': 0,
        'p1_wins': 0,
        'p2_wins': 0,
        'draws': 0,
        'game_lengths': [],
        'terminal_reasons': {},
        'samples_generated': 0,
        'raja_capture_chances': 0,   # moves where a Raja capture was legal
        'raja_captures_missed': 0,   # ...and a different move was played
        # The same, for greedy (temperature 0) moves only: in the first temp_moves
        # plies the move is sampled, so a "miss" there can be sampling noise (N10)
        'raja_capture_chances_greedy': 0,
        'raja_captures_missed_greedy': 0,
        # Search itself missed it: the most-visited move wasn't a Raja capture
        'raja_captures_missed_by_search': 0,
        'virtual_loss_collisions': 0,
        'simulations': 0,            # completed simulations (each backs up one leaf)
        'moves': 0,                  # moves chosen by search
        'simulations_dropped': 0,    # unused budget; always 0 (kept as a tripwire)
        'concurrent_games': batch_size,   # games in flight at once (the starting batch)
    }
    cache = EvalCache(eval_cache_size) if eval_cache_size > 0 else None
    leaves_per_step = max(1, int(leaves_per_step))
    # Steps per move: ceil(n_sims / leaves_per_step) without collisions. A collision
    # is retried on a later step instead of being dropped (needtofixagain N6), so a
    # move can take more steps. The first descent of a step can never collide, so
    # every active game completes >= 1 simulation per step and n_sims steps suffice.
    max_steps = n_sims

    start_time = time.time()

    # Initialize roots: one batched evaluation for all slots
    for slot in slots:
        slot.root = Node(slot.state, action_size=action_size)
    _expand_roots([slot.root for slot in slots], evaluator, cache)
    for slot in slots:
        slot.root.add_root_noise(dirichlet_alpha, dirichlet_eps)

    while games_completed < total_games:
        # 1. Run n_sims simulations per game, in lockstep across games.
        #    Each step: every active game descends up to `leaves_per_step` times
        #    (virtual loss on pending paths), then ONE batched forward pass
        #    evaluates all new leaves from all games.
        sims_left = {i: n_sims for i, slot in enumerate(slots) if not slot.done}
        for _ in range(max_steps):
            if not any(v > 0 for v in sims_left.values()):
                break
            pending = []          # (slot_idx, leaf, path)
            pending_ids = set()

            for i, slot in enumerate(slots):
                if slot.done or sims_left.get(i, 0) <= 0:
                    continue
                for _ in range(min(leaves_per_step, sims_left[i])):
                    leaf, path = _descend(slot.root, c_puct, virtual_loss, action_size,
                                          fpu_reduction)

                    if id(leaf) in pending_ids:
                        # Two descents reached the same unevaluated leaf: undo it and
                        # retry on the next step, once that leaf has been evaluated
                        _revert_virtual_loss(path, virtual_loss)
                        stats['virtual_loss_collisions'] += 1
                        break
                    sims_left[i] -= 1
                    stats['simulations'] += 1

                    value = _try_expand_without_network(leaf, cache)
                    if value is not None:
                        # Terminal or cache hit: back up immediately, no GPU needed
                        _revert_virtual_loss(path, virtual_loss)
                        _backup(leaf, value)
                    else:
                        pending.append((i, leaf, path))
                        pending_ids.add(id(leaf))

            if not pending:
                continue

            # B. BATCHED EVALUATION: one forward pass for every pending leaf of every game
            batch_policies, batch_values = evaluator.evaluate_batch([leaf.state for _, leaf, _ in pending])

            # C. EXPAND + BACKUP
            for k, (slot_idx, leaf, path) in enumerate(pending):
                value = float(batch_values[k])
                legal = leaf.state.legal_moves()
                if not legal:   # unreachable: is_terminal() already caught stalemate
                    leaf.set_terminal(1.0)
                    value = 1.0
                else:
                    priors = _normalised_priors(batch_policies[k], legal)
                    leaf.expand(legal, priors)
                    if cache is not None:
                        cache.put_eval(leaf.state, legal, priors, value)
                _revert_virtual_loss(path, virtual_loss)
                _backup(leaf, value)

        stats['simulations_dropped'] += sum(max(v, 0) for v in sims_left.values())

        # 2. SELECT MOVES for each active slot
        new_roots = []    # roots that still need expanding, done in one batch below
        for i, slot in enumerate(slots):
            if slot.done:
                continue

            # Visit counts from root
            visit_dist = slot.root.visit_distribution()
            temp = 1.0 if slot.state.ply_count < temp_threshold else 0.0
            legal_now = slot.state.legal_moves()
            action, pi = select_action_with_temperature(visit_dist, temp, legal_now)
            stats['moves'] += 1

            # Diagnostic (§2.1): was a free Raja capture on the board, and missed?
            raja_captures = _raja_capture_actions(slot.state, legal_now)
            if raja_captures:
                _count_raja_capture(stats, raja_captures, action, visit_dist, temp)

            # Record trajectory step — π and legal moves in the same canonical frame as the encoding
            encoded = slot.state.encode()
            legal_canonical = np.flatnonzero(slot.state.orient_policy(slot.state.legal_mask()))
            slot.history.append((encoded, slot.state.orient_policy(pi), slot.state.side_to_move,
                                 legal_canonical))

            # Apply move
            slot.state = slot.state.apply(action)

            # Check if game is terminal after this move
            done, z = slot.state.is_terminal()
            if done:
                slot.done = True
                games_completed += 1
                stats['games'] += 1
                stats['game_lengths'].append(slot.state.ply_count)

                # Classify outcome and terminal reason
                # z is from perspective of side to move.
                # If z == 1, current side won. If z == -1, previous side won.
                if z == 0:
                    stats['draws'] += 1
                    winner = 0
                elif z == -1:
                    # Previous player won
                    winner = -slot.state.side_to_move
                else:
                    winner = slot.state.side_to_move

                if winner == 1:
                    stats['p1_wins'] += 1
                elif winner == -1:
                    stats['p2_wins'] += 1

                from pipeline.selfplay import _classify_terminal
                reason = _classify_terminal(slot.state, z)
                stats['terminal_reasons'][reason] = stats['terminal_reasons'].get(reason, 0) + 1

                step_interval = max(10, total_games // 4)
                if games_completed % step_interval == 0 or games_completed == total_games:
                    cur_elapsed = time.time() - start_time
                    cur_speed = games_completed / max(cur_elapsed, 1e-5)
                    cur_mean_len = float(np.mean(stats['game_lengths'])) if stats['game_lengths'] else 0.0
                    print(f"  [Self-play] {games_completed}/{total_games} games completed ({cur_speed:.2f} games/s, avg {cur_mean_len:.1f} plies)", flush=True)

                # Flush game positions to replay buffer
                if buffer is not None:
                    for s_enc, pi_dist, side, legal_canonical in slot.history:
                        # Target value: +1 if this side won, -1 if lost, 0 if draw
                        target_v = float(winner * side) if winner != 0 else 0.0
                        # Mirror augmentation happens in ReplayBuffer.sample()
                        buffer.add(s_enc, pi_dist, target_v, legal=legal_canonical)
                        stats['samples_generated'] += 1

                # If we still have games left to start, reset this slot
                if games_started < total_games:
                    slot.reset()
                    games_started += 1
                    slot.root = Node(slot.state, action_size=action_size)
                    new_roots.append(slot.root)
                else:
                    slot.root = None
            else:
                # Subtree reuse: advance root to chosen child
                child = slot.root.children.get(action) if reuse_subtree else None
                if child is not None and child.is_expanded:
                    slot.root = child
                    slot.root.parent = None
                    slot.root.parent_action = -1
                    slot.root.parent_edge = -1
                else:
                    slot.root = Node(slot.state, action_size=action_size)
                    new_roots.append(slot.root)

        # Expand every new root in one batched forward pass (S5), then add
        # exploration noise to every active root (new or reused)
        _expand_roots(new_roots, evaluator, cache)
        for slot in slots:
            if not slot.done and slot.root is not None:
                slot.root.add_root_noise(dirichlet_alpha, dirichlet_eps)

    elapsed = time.time() - start_time
    stats['elapsed_seconds'] = elapsed
    stats['games_per_second'] = stats['games'] / max(elapsed, 1e-5)
    stats['mean_length'] = float(np.mean(stats['game_lengths'])) if stats['game_lengths'] else 0.0
    stats['draw_rate'] = stats['draws'] / max(stats['games'], 1)
    stats['raja_capture_miss_rate'] = (stats['raja_captures_missed'] / stats['raja_capture_chances']
                                       if stats['raja_capture_chances'] else 0.0)
    stats['raja_capture_miss_rate_greedy'] = (
        stats['raja_captures_missed_greedy'] / stats['raja_capture_chances_greedy']
        if stats['raja_capture_chances_greedy'] else 0.0)
    if cache is not None:
        stats.update(cache.stats())

    return stats


def _descend(root: Node, c_puct: float, virtual_loss: float, action_size: int,
             fpu_reduction: Optional[float] = None):
    """
    PUCT descent from the root to a leaf, applying virtual loss to every edge on
    the way: each traversed edge looks `virtual_loss` visits worse (a loss for
    the player choosing it) until the leaf's real value is backed up.
    Returns (leaf, path) where path is the list of (node, edge index) pairs.
    """
    node = root
    path = []
    while node.is_expanded and not node.is_terminal:
        edge = _select_edge(node, c_puct, fpu_reduction)
        if virtual_loss:
            node.N[edge] += virtual_loss
            node.W[edge] -= virtual_loss
        path.append((node, edge))
        action = node.legal_actions[edge]
        child = node.children.get(action)
        if child is None:
            child = Node(node.state.apply(action), parent=node, parent_action=action,
                         action_size=action_size, parent_edge=edge)
            node.children[action] = child
            return child, path
        node = child
    return node, path


def _revert_virtual_loss(path, virtual_loss: float):
    if not virtual_loss:
        return
    for node, edge in path:
        node.N[edge] -= virtual_loss
        node.W[edge] += virtual_loss


def _try_expand_without_network(leaf: Node, cache: Optional[EvalCache],
                                adjudicate: bool = True) -> Optional[float]:
    """
    Resolve a leaf without the network if possible: an already-terminal node,
    a cache hit, or a new terminal position. Returns its value (side to move),
    or None if the network must evaluate it.

    adjudicate: score a move-cap leaf by material (self-play) or as a draw
    (arena/evaluation, §2.2).
    """
    if leaf.is_terminal:
        return leaf.terminal_value
    if cache is not None:
        entry = cache.get(leaf.state, adjudicate)
        if entry is not None:
            return apply_entry(leaf, entry)
    done, z = leaf.state.is_terminal(adjudicate=adjudicate)
    if done:
        leaf.set_terminal(z)
        if cache is not None:
            cache.put_terminal(leaf.state, z, adjudicate)
        return z
    return None


RAJA_CAPTURE_COUNTERS = ('raja_capture_chances', 'raja_captures_missed',
                         'raja_capture_chances_greedy', 'raja_captures_missed_greedy',
                         'raja_captures_missed_by_search')


def _count_raja_capture(stats: dict, raja_captures: set, action: int,
                        visits: np.ndarray, temperature: float) -> None:
    """
    Update the missed-Raja-capture counters (§2.1 diagnostic, split by N10).
    Call only when a Raja capture was legal.
    """
    missed = action not in raja_captures
    stats['raja_capture_chances'] += 1
    stats['raja_captures_missed'] += missed
    if temperature < 1e-8:
        stats['raja_capture_chances_greedy'] += 1
        stats['raja_captures_missed_greedy'] += missed
    if int(np.argmax(visits)) not in raja_captures:
        stats['raja_captures_missed_by_search'] += 1


def _raja_capture_actions(state, legal: list[int]) -> set:
    """Legal moves that capture the opponent's Raja (Chaturanga only)."""
    board = getattr(state, 'board', None)
    if board is None or getattr(board, 'ndim', 0) != 1:
        return set()
    from chaturanga.board import RAJA
    enemy_raja = -RAJA * state.side_to_move
    return {a for a in legal if board[a % 64] == enemy_raja}


def _normalised_priors(policy: np.ndarray, legal: list[int]) -> np.ndarray:
    """Network policy restricted to the legal moves and renormalised (uniform if all zero)."""
    priors = np.asarray(policy, dtype=np.float64)[legal]
    total = priors.sum()
    if total > 0:
        return priors / total
    return np.full(len(legal), 1.0 / len(legal))


def _expand_roots(nodes: list, evaluator, cache: Optional[EvalCache],
                  adjudicate: bool = True) -> None:
    """
    Expand a list of fresh root nodes with ONE batched forward pass (S5):
    cache hits and terminal roots need no network call at all.
    """
    pending = []
    for node in nodes:
        if _try_expand_without_network(node, cache, adjudicate) is None:
            pending.append(node)
    if not pending:
        return
    policies, values = evaluator.evaluate_batch([node.state for node in pending])
    for k, node in enumerate(pending):
        legal = node.state.legal_moves()
        if not legal:   # unreachable: is_terminal() already caught stalemate
            node.set_terminal(1.0)
            continue
        priors = _normalised_priors(policies[k], legal)
        node.expand(legal, priors)
        if cache is not None:
            cache.put_eval(node.state, legal, priors, float(values[k]), adjudicate)


def run_batched_arena(
    net_a: torch.nn.Module,
    net_b: torch.nn.Module,
    create_state: Callable,
    n_games: int,
    n_sims: int,
    action_size: int = 4096,
    c_puct: float = 1.5,
    device=None,
    amp: bool = True,
    opening_plies: int = 0,
    eval_cache_size: int = 50_000,
    verbose: bool = False,
    n_sims_b: Optional[int] = None,
    fpu_reduction: Optional[float] = None,
) -> dict:
    """
    Arena between two networks with every game played concurrently (needtofix S6).

    `n_sims` is net_a's search budget per move; `n_sims_b` is net_b's
    (default: the same). Different budgets with the SAME weights give the
    plan's sims-vs-strength ablation.

    Same match rules as search.arena.arena(): games come in pairs that share a
    random opening of `opening_plies` plies, once with each network moving
    first; moves are the argmax of MCTS visit counts with no root noise; a
    move-cap ending is a draw (no adjudication, §2.2). Each move builds a fresh
    tree owned by the network to move; all leaves of all games are evaluated
    with one forward pass per network per simulation step.

    Returns the same dict as arena(), from net_a's point of view
    ('p1_*' = net_a).
    """
    from search.arena import random_opening

    evaluators = (BatchedEvaluator(net_a, device=device, amp=amp),
                  BatchedEvaluator(net_b, device=device, amp=amp))
    caches = tuple(EvalCache(eval_cache_size) if eval_cache_size > 0 else None for _ in range(2))

    # games[g] = [state, side played by net_a, done, result for net_a]
    games = []
    opening = None
    for g in range(n_games):
        if g % 2 == 0 or opening is None:
            opening = random_opening(create_state, opening_plies)
        a_side = opening.side_to_move if g % 2 == 0 else -opening.side_to_move
        games.append([opening, a_side, False, 0.0])

    def player_of(game) -> int:
        return 0 if game[0].side_to_move == game[1] else 1

    while not all(g[2] for g in games):
        active = [g for g in games if not g[2]]
        roots = [Node(g[0], action_size=action_size) for g in active]
        owners = [player_of(g) for g in active]
        for p in (0, 1):
            _expand_roots([r for r, o in zip(roots, owners) if o == p], evaluators[p], caches[p],
                          adjudicate=False)

        budgets = (n_sims, n_sims if n_sims_b is None else n_sims_b)
        for step in range(max(budgets)):
            pending = ([], [])        # per player: (leaf, path)
            for root, owner in zip(roots, owners):
                if step >= budgets[owner]:
                    continue
                leaf, path = _descend(root, c_puct, 0.0, action_size, fpu_reduction)
                # The arena scores a capped game as a draw, so its search must too (N1)
                value = _try_expand_without_network(leaf, caches[owner], adjudicate=False)
                if value is not None:
                    _backup(leaf, value)
                else:
                    pending[owner].append(leaf)
            for p in (0, 1):
                if not pending[p]:
                    continue
                policies, values = evaluators[p].evaluate_batch([leaf.state for leaf in pending[p]])
                for k, leaf in enumerate(pending[p]):
                    legal = leaf.state.legal_moves()
                    priors = _normalised_priors(policies[k], legal)
                    leaf.expand(legal, priors)
                    if caches[p] is not None:
                        caches[p].put_eval(leaf.state, legal, priors, float(values[k]),
                                           adjudicate=False)
                    _backup(leaf, float(values[k]))

        for game, root in zip(active, roots):
            action = int(np.argmax(root.visit_distribution()))
            game[0] = game[0].apply(action)
            done, z = game[0].is_terminal(adjudicate=False)
            if done:
                # z is for the side to move; convert to net_a's point of view
                game[2] = True
                game[3] = z if game[0].side_to_move == game[1] else -z

    results = [g[3] for g in games]
    a_wins = sum(1 for r in results if r > 0)
    b_wins = sum(1 for r in results if r < 0)
    draws = len(results) - a_wins - b_wins
    if verbose:
        print(f"  [Arena] {n_games} games: A {a_wins}W - {draws}D - {b_wins}L B", flush=True)
    return {
        'p1_wins': a_wins,
        'p2_wins': b_wins,
        'draws': draws,
        'p1_win_rate': (a_wins + 0.5 * draws) / max(n_games, 1),
        'results': results,
    }
