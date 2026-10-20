"""
Arena — pit two players against each other and compute win rates.

Used for:
1. Gating: challenger vs incumbent after each training iteration
2. Evaluation: agent vs baselines
3. Elo computation from round-robin tournaments
"""

import numpy as np
from typing import Callable, Optional


def play_game(
    player1: Callable,
    player2: Callable,
    initial_state,
    action_size: int = 4096,
    verbose: bool = False,
) -> tuple[float, int, list]:
    """
    Play a single game between two players.
    
    Args:
        player1: Callable(state) -> action for player 1 (moves first)
        player2: Callable(state) -> action for player 2
        initial_state: Starting game state
        action_size: Size of action space
        verbose: Print moves if True
    
    Returns:
        (result, ply_count, move_history)
        result: +1 if player1 wins, -1 if player2 wins, 0 for draw
    """
    state = initial_state
    move_history = []
    
    while True:
        # No material adjudication in arena games (§2.2): a capped game is a draw
        done, z = state.is_terminal(adjudicate=False)
        if done:
            # z is from the current side-to-move's perspective
            # Convert to player1's perspective
            if state.side_to_move == initial_state.side_to_move:
                result = z
            else:
                result = -z
            return result, state.ply_count, move_history
        
        # Current player selects a move
        if state.side_to_move == initial_state.side_to_move:
            action = player1(state)
        else:
            action = player2(state)
        
        if verbose:
            side = "P1" if state.side_to_move == initial_state.side_to_move else "P2"
            print(f"  Ply {state.ply_count}: {side} plays action {action}")
        
        move_history.append(action)
        
        # Apply move
        if hasattr(state, 'apply'):
            state = state.apply(action)
        else:
            from chaturanga.moves import apply_move
            state = apply_move(state, action)


def arena(
    player1: Callable,
    player2: Callable,
    create_state: Callable,
    n_games: int = 60,
    action_size: int = 4096,
    verbose: bool = False,
    opening_plies: int = 0,
) -> dict:
    """
    Play n_games between two players, alternating colors.
    
    Args:
        player1, player2: player callables
        create_state: factory function returning a fresh starting state
        n_games: total games to play (should be even for fair color split)
        action_size: action space size
        verbose: print progress
        opening_plies: if > 0, each pair of games starts from the same random
            opening of this many plies, once with each player to move first.
            Without this, two deterministic players (argmax MCTS, no noise)
            replay the same 2 games for the whole match.
    
    Returns:
        dict with 'p1_wins', 'p2_wins', 'draws', 'p1_win_rate', 'results'
    """
    p1_wins = 0
    p2_wins = 0
    draws = 0
    results = []
    
    state = None
    for game_idx in range(n_games):
        # A new opening for every pair, shared by both games of the pair
        if game_idx % 2 == 0 or state is None:
            state = random_opening(create_state, opening_plies)
        
        # Alternate who goes first
        if game_idx % 2 == 0:
            result, plies, _ = play_game(player1, player2, state, action_size)
        else:
            result, plies, _ = play_game(player2, player1, state, action_size)
            result = -result  # Flip to be from player1's perspective
        
        if result > 0:
            p1_wins += 1
        elif result < 0:
            p2_wins += 1
        else:
            draws += 1
        
        results.append(result)
        
        if verbose and ((game_idx + 1) % max(10, n_games // 4) == 0 or (game_idx + 1) == n_games):
            print(f"  [Arena] Game {game_idx+1}/{n_games}: "
                  f"P1 {p1_wins}W - {draws}D - {p2_wins}L P2 "
                  f"(score: {(p1_wins + 0.5 * draws)/(game_idx+1):.1%})", flush=True)
    
    total = n_games
    return {
        'p1_wins': p1_wins,
        'p2_wins': p2_wins,
        'draws': draws,
        'p1_win_rate': (p1_wins + 0.5 * draws) / total,
        'results': results,
    }


def random_opening(create_state: Callable, n_plies: int, max_tries: int = 100):
    """
    Return a fresh state advanced by n_plies uniformly random legal moves,
    retrying if a random line happens to end the game.
    """
    for _ in range(max_tries):
        state = create_state()
        for _ in range(n_plies):
            done, _ = state.is_terminal()
            if done:
                break
            state = state.apply(int(np.random.choice(state.legal_moves())))
        if not state.is_terminal()[0]:
            return state
    return create_state()


def score_interval(results: list, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson confidence interval (default 95%) for a match score.

    results: per-game outcomes for player 1 (+1 win, 0 draw, -1 loss); the
    score is the mean of 1 / 0.5 / 0. Treating the score as a proportion
    over n games is slightly conservative with draws, and unlike a plain
    normal interval it never collapses to zero width at 0% or 100%. The
    standard error near 50% is ~6.4 points at 60 games and ~3.5 at 200
    (needtofixagain N12): a 55% gate at 60 games is less than one standard
    error above 50%.
    """
    n = len(results)
    if n == 0:
        return 0.0, 1.0
    p = float(np.mean((np.asarray(results, dtype=np.float64) + 1.0) / 2.0))
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def gate_model(
    challenger: Callable,
    incumbent: Callable,
    create_state: Callable,
    n_games: int = 60,
    win_threshold: float = 0.55,
    action_size: int = 4096,
    verbose: bool = True,
    opening_plies: int = 0,
) -> tuple[bool, dict]:
    """
    Arena gating: does the challenger beat the incumbent?
    
    Returns:
        (promoted, stats) — True if challenger's win rate >= threshold.
    """
    stats = arena(challenger, incumbent, create_state, n_games, action_size,
                  verbose=verbose, opening_plies=opening_plies)
    promoted = stats['p1_win_rate'] >= win_threshold
    return promoted, stats
