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
    cache=None,
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
    legal_canonical = []
    raja_capture_chances = 0
    raja_captures_missed = 0
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
        
        # Run MCTS (keeping the root so the chosen subtree can be reused)
        visit_counts, root = search(
            state=state,
            evaluate=evaluate,
            n_sims=n_sims,
            action_size=action_size,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            add_noise=True,
            root_node=root_node,
            cache=cache,
            return_root=True,
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

        # Diagnostic (§2.1): a free Raja capture was available but not played
        from pipeline.inference_server import _raja_capture_actions
        raja_captures = _raja_capture_actions(state, legal_actions)
        if raja_captures:
            raja_capture_chances += 1
            raja_captures_missed += action not in raja_captures
        
        # Store training data — π in the same canonical frame as the encoding
        states_encoded.append(encoded)
        policies.append(state.orient_policy(pi))
        legal_canonical.append(np.flatnonzero(state.orient_policy(state.legal_mask())))
        
        # Subtree reuse: the chosen child's tree becomes the next root
        root_node = get_subtree(root, action) if reuse_subtree else None
        
        # Apply action
        if hasattr(state, 'apply'):
            state = state.apply(action)
        else:
            from chaturanga.moves import apply_move
            state = apply_move(state, action)
    
    return {
        'states': states_encoded,
        'policies': policies,
        'legal': legal_canonical,
        'raja_capture_chances': raja_capture_chances,
        'raja_captures_missed': int(raja_captures_missed),
        'result': result,
        'ply_count': state.ply_count,
        'terminal_reason': terminal_reason,
    }


def _classify_terminal(state, z=None) -> str:
    """
    Classify how the game ended. Delegates to the terminal ladder itself, so
    the reason can never disagree with is_terminal() (it used to check the
    move cap before stalemate — the reverse of §2.1).
    """
    if hasattr(state, 'board') and getattr(state.board, 'ndim', 0) == 1:
        from chaturanga.terminal import terminal_reason
        return terminal_reason(state)
    # Connect-4
    return 'board_full' if z == 0 else 'four_in_a_row'


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
    reuse_subtree: bool = True,
    eval_cache_size: int = 50_000,
) -> list[dict]:
    """
    Run a batch of self-play games sequentially (naive mode).

    Phase 4 replaces this with parallel-games batched mode.
    One evaluation cache (transposition table) is shared by all games in the
    batch — valid because the network is fixed for the whole batch.
    """
    from search.cache import EvalCache
    cache = EvalCache(eval_cache_size) if eval_cache_size > 0 else None
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
            reuse_subtree=reuse_subtree,
            cache=cache,
        )
        if cache is not None:
            game.update(cache.stats())   # cumulative for the batch so far
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
