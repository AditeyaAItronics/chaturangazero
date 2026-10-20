"""
Elo computation and plotting utilities.

Used in Phase 6 for:
- Round-robin tournament Elo ratings
- Learning curves
- Loss curves
- Terminal reason distributions
"""

import numpy as np
import json
import os
from typing import Optional


def compute_elo(results: list[dict], k: float = None, initial_elo: float = 1500.0,
                prior_games: float = 2.0, max_iter: int = 10000, tol: float = 1e-10) -> dict[str, float]:
    """
    Fit Elo ratings to a set of match results by maximum likelihood
    (Bradley-Terry model, fitted with Hunter's MM algorithm).

    Unlike sequential K-factor updates, the result does not depend on the
    order the games were played in, which is what a checkpoint ladder needs.

    Each result should have:
        'player1': str, 'player2': str, 'score': float (1.0/0.5/0.0 for p1)

    Draws count as half a win for each side. A weak prior — `prior_games`
    virtual games, half won and half lost, against an anchor rated
    `initial_elo` — keeps ratings finite for a player who never lost (or
    never won) and sets the scale's origin.

    Args:
        k: ignored; kept so existing callers that pass it still work.

    Returns:
        dict mapping player name to Elo rating
    """
    names = sorted({r['player1'] for r in results} | {r['player2'] for r in results})
    if not names:
        return {}
    index = {name: i for i, name in enumerate(names)}
    n = len(names)

    wins = np.zeros(n)            # total score (draw = 0.5)
    games = np.zeros((n, n))      # games played between each pair
    for r in results:
        i, j = index[r['player1']], index[r['player2']]
        score = float(r['score'])
        wins[i] += score
        wins[j] += 1.0 - score
        games[i, j] += 1
        games[j, i] += 1

    # Virtual games against the anchor (strength 1, i.e. initial_elo)
    wins = wins + prior_games / 2.0
    gamma = np.ones(n)
    for _ in range(max_iter):
        pair = games / (gamma[:, None] + gamma[None, :])
        denom = pair.sum(axis=1) + prior_games / (gamma + 1.0)
        new_gamma = wins / denom
        if np.max(np.abs(np.log(new_gamma) - np.log(gamma))) < tol:
            gamma = new_gamma
            break
        gamma = new_gamma

    ratings = initial_elo + 400.0 * np.log10(gamma)
    return {name: float(ratings[index[name]]) for name in names}


def round_robin_elo(
    players: dict[str, object],
    create_state,
    play_fn=None,
    games_per_pair: int = 10,
    action_size: int = 4096,
    opening_plies: int = 4,
) -> dict[str, float]:
    """
    Run a round-robin tournament and compute Elo ratings.

    Args:
        players: dict mapping name -> player callable
        create_state: factory for initial states
        play_fn: unused (kept for backward compatibility)
        games_per_pair: games per player pair (colours alternate)
        opening_plies: each pair of games starts from the same random opening of
            this many plies, so deterministic players don't repeat one game

    Returns:
        dict mapping player name to Elo rating
    """
    from search.arena import play_game, random_opening

    names = list(players.keys())
    results = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p1_name, p2_name = names[i], names[j]
            p1, p2 = players[p1_name], players[p2_name]

            state = None
            for g in range(games_per_pair):
                if g % 2 == 0 or state is None:
                    state = random_opening(create_state, opening_plies)

                if g % 2 == 0:
                    result, _, _ = play_game(p1, p2, state, action_size)
                    score = (result + 1) / 2  # Convert from [-1,1] to [0,1]
                else:
                    result, _, _ = play_game(p2, p1, state, action_size)
                    score = (-result + 1) / 2
                results.append({
                    'player1': p1_name,
                    'player2': p2_name,
                    'score': score,
                })

    return compute_elo(results)


def generate_plots(metrics_path: str, output_dir: str):
    """
    Generate training plots from metrics JSON.
    
    Creates:
    - Loss curves (policy + value)
    - Policy entropy over time
    - Game length over time
    - Terminal reason distribution
    - Cap rate over time
    
    Uses matplotlib if available, otherwise prints text summaries.
    """
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        iterations = [m['iteration'] for m in metrics]
        
        # Loss curves
        if 'mean_policy_loss' in metrics[0]:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            p_loss = [m.get('mean_policy_loss', 0) for m in metrics]
            v_loss = [m.get('mean_value_loss', 0) for m in metrics]
            
            ax1.plot(iterations, p_loss, label='Policy Loss')
            ax1.plot(iterations, v_loss, label='Value Loss')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')
            ax1.legend()
            ax1.grid(True)
            
            entropy = [m.get('mean_entropy', 0) for m in metrics]
            ax2.plot(iterations, entropy, color='green')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Entropy')
            ax2.set_title('Policy Entropy')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=150)
            plt.close()
        
        # Game length
        if 'avg_plies' in metrics[0]:
            fig, ax = plt.subplots(figsize=(8, 5))
            plies = [m['avg_plies'] for m in metrics]
            ax.plot(iterations, plies)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Average Plies')
            ax.set_title('Game Length Over Training')
            ax.grid(True)
            plt.savefig(os.path.join(output_dir, 'game_length.png'), dpi=150)
            plt.close()
        
        # Cap rate
        if 'cap_rate' in metrics[0]:
            fig, ax = plt.subplots(figsize=(8, 5))
            cap = [m['cap_rate'] for m in metrics]
            ax.plot(iterations, cap)
            ax.axhline(y=0.25, color='r', linestyle='--', label='Warning threshold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Cap Rate')
            ax.set_title('Move Cap Adjudication Rate')
            ax.legend()
            ax.grid(True)
            plt.savefig(os.path.join(output_dir, 'cap_rate.png'), dpi=150)
            plt.close()
        
        print(f"Plots saved to {output_dir}")
        
    except ImportError:
        print("matplotlib not available, printing text summaries")
        for m in metrics[-5:]:
            print(f"  Iter {m['iteration']}: "
                  f"policy_loss={m.get('mean_policy_loss', '?'):.4f}, "
                  f"value_loss={m.get('mean_value_loss', '?'):.4f}, "
                  f"avg_plies={m.get('avg_plies', '?'):.1f}")


def plot_elo_curve(elo_history: list[tuple[int, float]], output_path: str):
    """Plot Elo rating over iterations."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        iters, elos = zip(*elo_history)
        plt.figure(figsize=(10, 6))
        plt.plot(iters, elos, 'b-o', markersize=3)
        plt.xlabel('Iteration')
        plt.ylabel('Elo Rating')
        plt.title('ChaturangaZero Learning Curve')
        plt.grid(True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Elo curve saved to {output_path}")
    except ImportError:
        print("matplotlib not available")
