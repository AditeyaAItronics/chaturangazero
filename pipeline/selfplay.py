"""
Self-play: generate training data by playing games against oneself.

Produces (state_encoding, policy_target, value_target) triples
for the replay buffer.
"""

import numpy as np
from typing import Callable, Optional
import time


def self_play_game(
    create_state: Callable,
    evaluate: Callable,
    action_size: int = 4096,
    n_sims: int = 100,
    c_puct: float = 1.5,
    temp_moves: int = 20,
    move_cap: int = 300,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    reuse_subtree: bool = True,
    mirror_perm: Optional[np.ndarray] = None,
) -> dict:
    """
    Play one complete game of self-play, collecting training data.
    
    Returns:
        dict with:
        - 'states': list of encoded states
        - 'policies': list of π vectors
        - 'result': final z from player 1's perspective
        - 'ply_count': game length
        - 'terminal_reason': string describing how the game ended
    """
    from search.mcts import search, select_action_with_temperature, get_subtree
    
    state = create_state()
    initial_side = state.side_to_move
    
    states_encoded = []
    policies = []
    root_node = None
    
    while True:
        # Check terminal
        done, z = state.is_terminal()
        if done:
            # z is from current side-to-move's perspective
            # Convert to player 1 (initial_side) perspective
            if state.side_to_move == initial_side:
                result = z
            else:
                result = -z
            
            terminal_reason = _classify_terminal(state, z)
            break
        
        # Get encoded state for training data
        encoded = state.encode()
        
        # Run MCTS
        visit_counts = search(
            state=state,
            evaluate=evaluate,
            n_sims=n_sims,
            action_size=action_size,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            add_noise=True,
            root_node=root_node,
        )
        
        # Select action with temperature
        temperature = 1.0 if state.ply_count < temp_moves else 0.0
        
        legal_actions = None
        if hasattr(state, 'legal_moves'):
            legal_actions = state.legal_moves()
        else:
            from chaturanga.moves import generate_moves
            legal_actions = generate_moves(state)
        
        action, pi = select_action_with_temperature(
            visit_counts, temperature, legal_actions
        )
        
        # Store training data — π in the same canonical frame as the encoding
        states_encoded.append(encoded)
        policies.append(state.orient_policy(pi))
        
        # Subtree reuse
        if reuse_subtree:
            root_node = get_subtree(
                _get_root_from_search(visit_counts, state, action_size),
                action,
            )
        else:
            root_node = None
        
        # Apply action
        if hasattr(state, 'apply'):
            state = state.apply(action)
        else:
            from chaturanga.moves import apply_move
            state = apply_move(state, action)
    
    return {
        'states': states_encoded,
        'policies': policies,
        'result': result,
        'ply_count': state.ply_count,
        'terminal_reason': terminal_reason,
    }


def _get_root_from_search(visit_counts, state, action_size):
    """Helper to reconstruct root node for subtree reuse."""
    # For now, we skip subtree reuse in naive mode
    # The parallel-games server in Phase 4 will handle this properly
    return None


def _classify_terminal(state, z) -> str:
    """Classify how the game ended."""
    from chaturanga.board import MOVE_CAP
    
    if not state.has_raja(state.side_to_move):
        return 'raja_captured'
    
    side_count = state.piece_count(state.side_to_move)
    opp_count = state.piece_count(-state.side_to_move)
    
    if side_count == 1 and opp_count == 1:
        return 'king_vs_king'
    
    if side_count == 1 and opp_count > 1:
        return 'bare_king'
    
    if state.ply_count >= MOVE_CAP:
        return 'move_cap'
    
    # Must be stalemate
    return 'stalemate'


def run_self_play_batch(
    n_games: int,
    create_state: Callable,
    evaluate: Callable,
    action_size: int = 4096,
    n_sims: int = 100,
    c_puct: float = 1.5,
    temp_moves: int = 20,
    move_cap: int = 300,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    verbose: bool = True,
) -> list[dict]:
    """
    Run a batch of self-play games sequentially (naive mode).
    
    Phase 4 replaces this with parallel-games batched mode.
    """
    games = []
    stats = {
        'raja_captured': 0,
        'king_vs_king': 0,
        'bare_king': 0,
        'stalemate': 0,
        'move_cap': 0,
        'total_plies': 0,
    }
    
    start_time = time.time()
    
    for i in range(n_games):
        game = self_play_game(
            create_state=create_state,
            evaluate=evaluate,
            action_size=action_size,
            n_sims=n_sims,
            c_puct=c_puct,
            temp_moves=temp_moves,
            move_cap=move_cap,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
        )
        games.append(game)
        
        reason = game['terminal_reason']
        if reason in stats:
            stats[reason] += 1
        stats['total_plies'] += game['ply_count']
        
        if verbose and (i + 1) % max(1, n_games // 10) == 0:
            elapsed = time.time() - start_time
            avg_plies = stats['total_plies'] / (i + 1)
            print(f"  Game {i+1}/{n_games}: "
                  f"avg_plies={avg_plies:.1f}, "
                  f"games/h={3600*(i+1)/elapsed:.0f}, "
                  f"terminal: {stats}")
    
    return games
