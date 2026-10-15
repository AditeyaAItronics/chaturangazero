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
from search.mcts import _select_action, _backup, select_action_with_temperature
from net.replay import ReplayBuffer


class BatchedEvaluator:
    """
    Batched neural network evaluator with AMP (FP16) and inference_mode.
    
    Can evaluate a batch of states in a single forward pass.
    """
    def __init__(self, net: torch.nn.Module, device: str = 'cuda', amp: bool = True):
        self.device = torch.device(device if isinstance(device, str) else device)
        self.net = net.to(self.device)
        self.amp = amp and (self.device.type == 'cuda')
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
        x = torch.from_numpy(encodings).to(self.device, non_blocking=True)

        with torch.inference_mode():
            if self.amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    policy_logits, values = self.net(x)
            else:
                policy_logits, values = self.net(x)

            policies = F.softmax(policy_logits, dim=-1).cpu().numpy()
            vals = values.squeeze(-1).cpu().numpy()

        # Handle scalar output if B=1
        if vals.ndim == 0:
            vals = np.array([vals.item()], dtype=np.float32)

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
    device: str = 'cuda',
    amp: bool = True,
    buffer: Optional[ReplayBuffer] = None,
    mirror_augmentation: bool = True,
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
        device: 'cuda' or 'cpu'.
        amp: Whether to use FP16 automatic mixed precision.
        buffer: Optional ReplayBuffer to add training examples directly.
        mirror_augmentation: Whether to add mirrored positions to buffer.
        
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
    }

    start_time = time.time()

    # Initialize roots and run initial expansion for active slots
    _expand_and_init_roots(slots, evaluator, action_size, dirichlet_alpha, dirichlet_eps)

    while games_completed < total_games:
        # 1. Run n_sims simulations across all currently active slots in lockstep
        for _ in range(n_sims):
            # A. SELECT: Traverse tree for each active slot until an unexpanded/terminal leaf is found
            leaves = []
            eval_needed_indices = []
            eval_states = []

            for i, slot in enumerate(slots):
                if slot.done:
                    leaves.append(None)
                    continue

                node = slot.root
                while node.is_expanded and not node.is_terminal:
                    action = _select_action(node, c_puct)
                    if action in node.children:
                        node = node.children[action]
                    else:
                        # Create child
                        child_state = slot.state_cls.apply(node.state, action) if hasattr(slot.state_cls, 'apply') and not hasattr(node.state, 'apply') else node.state.apply(action)
                        child = Node(child_state, parent=node, parent_action=action, action_size=action_size)
                        node.children[action] = child
                        node = child
                        break

                leaves.append(node)

                # Check if this leaf needs neural network evaluation
                if not node.is_expanded and not node.is_terminal:
                    done, z = node.state.is_terminal()
                    if done:
                        node.is_terminal = True
                        node.terminal_value = z
                        node.is_expanded = True
                    else:
                        eval_needed_indices.append(i)
                        eval_states.append(node.state)

            # B. BATCHED EVALUATION: Send all non-terminal leaves to GPU in one forward pass
            if eval_states:
                batch_policies, batch_values = evaluator.evaluate_batch(eval_states)
                for k, slot_idx in enumerate(eval_needed_indices):
                    leaf = leaves[slot_idx]
                    p = batch_policies[k]
                    v = batch_values[k]

                    # Legal move masking
                    legal = leaf.state.legal_moves()
                    leaf.legal_actions = legal
                    if not legal:
                        leaf.is_terminal = True
                        leaf.terminal_value = 1.0  # Stalemate = win
                        leaf.is_expanded = True
                    else:
                        mask = np.zeros(action_size, dtype=bool)
                        for a in legal:
                            mask[a] = True
                        p = p * mask
                        psum = np.sum(p)
                        if psum > 0:
                            leaf.P[:] = p / psum
                        else:
                            leaf.P[:] = mask.astype(np.float32) / len(legal)
                        leaf.is_expanded = True

            # C. BACKUP: Propagate values back up the tree for each active slot
            eval_map = {slot_idx: k for k, slot_idx in enumerate(eval_needed_indices)}
            for i, slot in enumerate(slots):
                if slot.done or leaves[i] is None:
                    continue
                leaf = leaves[i]
                if leaf.is_terminal:
                    val = leaf.terminal_value
                elif i in eval_map:
                    val = float(batch_values[eval_map[i]])
                else:
                    val = 0.0
                _backup(leaf, val)

        # 2. SELECT MOVES for each active slot
        for i, slot in enumerate(slots):
            if slot.done:
                continue

            # Visit counts from root
            visit_dist = slot.root.visit_distribution()
            temp = 1.0 if slot.state.ply_count < temp_threshold else 0.0
            action, pi = select_action_with_temperature(
                visit_dist, temp, slot.state.legal_moves()
            )

            # Record trajectory step
            encoded = slot.state.encode()
            slot.history.append((encoded, pi, slot.state.side_to_move))

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

                reason = getattr(slot.state, 'terminal_reason', None)
                if reason is None:
                    if hasattr(slot.state, '_check_win'):
                        reason = 'board_full' if slot.state.ply_count >= 42 else 'four_in_a_row'
                    else:
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
                    for s_enc, pi_dist, side in slot.history:
                        # Target value: +1 if this side won, -1 if lost, 0 if draw
                        target_v = float(winner * side) if winner != 0 else 0.0
                        buffer.add(s_enc, pi_dist, target_v)
                        stats['samples_generated'] += 1

                        if mirror_augmentation:
                            # Horizontal mirror
                            # Connect-4: flip along width axis (axis 2), flip columns
                            # Chaturanga: flip along width axis (axis 2), apply action permutation
                            if hasattr(slot.state, 'mirror_action_permutation'):
                                perm = slot.state.mirror_action_permutation()
                                mir_s = np.flip(s_enc, axis=2).copy()
                                mir_pi = pi_dist[perm].copy()
                                buffer.add(mir_s, mir_pi, target_v)
                            elif hasattr(slot.state, 'mirror'):
                                _, perm = slot.state.mirror()
                                mir_s = np.flip(s_enc, axis=2).copy()
                                mir_pi = pi_dist[perm].copy()
                                buffer.add(mir_s, mir_pi, target_v)
                            else:
                                mir_s = np.flip(s_enc, axis=2).copy()
                                mir_pi = pi_dist[::-1].copy()
                                buffer.add(mir_s, mir_pi, target_v)
                            stats['samples_generated'] += 1

                # If we still have games left to start, reset this slot
                if games_started < total_games:
                    slot.reset()
                    games_started += 1
                    _init_single_slot_root(slot, evaluator, action_size, dirichlet_alpha, dirichlet_eps)
                else:
                    slot.root = None
            else:
                # Subtree reuse: advance root to chosen child
                if action in slot.root.children:
                    slot.root = slot.root.children[action]
                    slot.root.parent = None
                    slot.root.parent_action = -1
                else:
                    slot.root = Node(slot.state, action_size=action_size)
                    _expand_single_node(slot.root, evaluator, action_size)

                # Add exploration noise at new root
                if slot.root.legal_actions:
                    noise = np.random.dirichlet([dirichlet_alpha] * len(slot.root.legal_actions))
                    for k, a in enumerate(slot.root.legal_actions):
                        slot.root.P[a] = (1 - dirichlet_eps) * slot.root.P[a] + dirichlet_eps * noise[k]

    elapsed = time.time() - start_time
    stats['elapsed_seconds'] = elapsed
    stats['games_per_second'] = stats['games'] / max(elapsed, 1e-5)
    stats['mean_length'] = float(np.mean(stats['game_lengths'])) if stats['game_lengths'] else 0.0

    return stats


def _expand_and_init_roots(slots, evaluator, action_size, alpha, eps):
    """Batch-evaluates and initializes the roots for all initial slots."""
    states_to_eval = []
    active_indices = []

    for i, slot in enumerate(slots):
        slot.root = Node(slot.state, action_size=action_size)
        done, z = slot.state.is_terminal()
        if done:
            slot.root.is_terminal = True
            slot.root.terminal_value = z
            slot.root.is_expanded = True
        else:
            states_to_eval.append(slot.state)
            active_indices.append(i)

    if states_to_eval:
        policies, values = evaluator.evaluate_batch(states_to_eval)
        for k, slot_idx in enumerate(active_indices):
            slot = slots[slot_idx]
            legal = slot.state.legal_moves()
            slot.root.legal_actions = legal
            if not legal:
                slot.root.is_terminal = True
                slot.root.terminal_value = 1.0
                slot.root.is_expanded = True
                continue

            p = policies[k]
            mask = np.zeros(action_size, dtype=bool)
            for a in legal:
                mask[a] = True
            p = p * mask
            psum = np.sum(p)
            p = p / psum if psum > 0 else mask.astype(np.float32) / len(legal)

            # Dirichlet noise at root
            noise = np.random.dirichlet([alpha] * len(legal))
            for idx_a, a in enumerate(legal):
                slot.root.P[a] = (1 - eps) * p[a] + eps * noise[idx_a]
            slot.root.is_expanded = True


def _init_single_slot_root(slot, evaluator, action_size, alpha, eps):
    """Initializes a new root when a slot is recycled."""
    slot.root = Node(slot.state, action_size=action_size)
    done, z = slot.state.is_terminal()
    if done:
        slot.root.is_terminal = True
        slot.root.terminal_value = z
        slot.root.is_expanded = True
        return

    policies, _ = evaluator.evaluate_batch([slot.state])
    p = policies[0]
    legal = slot.state.legal_moves()
    slot.root.legal_actions = legal
    if not legal:
        slot.root.is_terminal = True
        slot.root.terminal_value = 1.0
        slot.root.is_expanded = True
        return

    mask = np.zeros(action_size, dtype=bool)
    for a in legal:
        mask[a] = True
    p = p * mask
    psum = np.sum(p)
    p = p / psum if psum > 0 else mask.astype(np.float32) / len(legal)

    noise = np.random.dirichlet([alpha] * len(legal))
    for idx_a, a in enumerate(legal):
        slot.root.P[a] = (1 - eps) * p[a] + eps * noise[idx_a]
    slot.root.is_expanded = True


def _expand_single_node(node, evaluator, action_size):
    """Expands a single node."""
    done, z = node.state.is_terminal()
    if done:
        node.is_terminal = True
        node.terminal_value = z
        node.is_expanded = True
        return

    policies, _ = evaluator.evaluate_batch([node.state])
    p = policies[0]
    legal = node.state.legal_moves()
    node.legal_actions = legal
    if not legal:
        node.is_terminal = True
        node.terminal_value = 1.0
        node.is_expanded = True
        return

    mask = np.zeros(action_size, dtype=bool)
    for a in legal:
        mask[a] = True
    p = p * mask
    psum = np.sum(p)
    node.P[:] = p / psum if psum > 0 else mask.astype(np.float32) / len(legal)
    node.is_expanded = True
