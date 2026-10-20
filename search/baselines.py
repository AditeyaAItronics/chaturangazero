"""
Baseline opponents for evaluation.

1. Random player — uniform random over legal moves
2. Greedy material player — picks the move that maximizes material advantage
3. Alpha-beta with material + mobility — 3-ply lookahead

Greedy and alpha-beta break ties randomly (moves are shuffled before search),
so repeated games against a deterministic agent are not all identical.
Baselines see real game endings only: a move-cap position is a draw, never
materially adjudicated (§2.2).
"""

import numpy as np

WIN_SCORE = 1000.0     # terminal score scale; real evaluations stay far below it
MOBILITY_WEIGHT = 0.05


def _legal_moves(state) -> list[int]:
    if hasattr(state, 'legal_moves'):
        return state.legal_moves()
    from chaturanga.moves import generate_moves
    return generate_moves(state)


def _apply(state, action):
    """Apply action to state."""
    if hasattr(state, 'apply'):
        return state.apply(action)
    from chaturanga.moves import apply_move
    return apply_move(state, action)


def _is_chaturanga(state) -> bool:
    return hasattr(state, 'material_score')


def _shuffled(moves: list[int]) -> list[int]:
    """Random order, so ties are broken randomly (np.random respects the run seed)."""
    moves = list(moves)
    np.random.shuffle(moves)
    return moves


def _ordered(state, moves: list[int]) -> list[int]:
    """
    Captures first, most valuable victim first (MVV), random order otherwise.
    Good ordering is what makes alpha-beta prune.
    """
    moves = _shuffled(moves)
    if not _is_chaturanga(state):
        return moves
    from chaturanga.board import PIECE_VALUES, RAJA
    board = state.board

    def victim_value(action: int) -> float:
        target = abs(int(board[action % 64]))
        if target == 0:
            return 0.0
        return 100.0 if target == RAJA else PIECE_VALUES[target]

    return sorted(moves, key=victim_value, reverse=True)   # stable: ties stay shuffled


def random_player(state, action_size: int = 4096) -> int:
    """Play a random legal move."""
    moves = _legal_moves(state)
    if not moves:
        return -1
    return int(np.random.choice(moves))


def connect4_tactical_player(state, action_size: int = 7) -> int:
    """
    Connect-4 "greedy" baseline: win now if possible; otherwise avoid any move
    that hands the opponent an immediate win (which blocks a single threat);
    otherwise random. Material has no meaning in Connect-4, so this is the
    one-ply-tactics equivalent of the greedy-material player.
    """
    moves = _shuffled(_legal_moves(state))
    if not moves:
        return -1
    safe = []
    for a in moves:
        child = state.apply(a)
        done, z = child.is_terminal()
        if done and z < 0:          # opponent (to move in child) has lost: we win
            return int(a)
        if done:
            safe.append(a)          # a draw by filling the board
            continue
        opponent_wins = any(
            child.apply(b).is_terminal() == (True, -1.0) for b in child.legal_moves())
        if not opponent_wins:
            safe.append(a)
    return int(safe[0] if safe else moves[0])


def greedy_material_player(state, action_size: int = 4096) -> int:
    """
    Pick the move that maximizes material advantage.
    Greedy: look one move ahead, pick the best capture or any non-capture.
    For Connect-4 (no material) this delegates to connect4_tactical_player.
    """
    moves = _legal_moves(state)
    if not moves:
        return -1
    if not _is_chaturanga(state):
        return connect4_tactical_player(state, action_size)

    side = state.side_to_move
    best_action = None
    best_score = -float('inf')

    for action in _shuffled(moves):
        child = _apply(state, action)
        done, z = child.is_terminal(adjudicate=False)
        if done:
            # z is from the child's side to move (the opponent), so negate
            score = -z * WIN_SCORE
        elif _is_chaturanga(child):
            score = child.material_score(side) - child.material_score(-side)
        else:
            score = 0.0

        if score > best_score:
            best_score = score
            best_action = action

    return int(best_action)


def alpha_beta_player(state, depth: int = 3, action_size: int = 4096) -> int:
    """
    Alpha-beta search with material + mobility evaluation.
    Default depth = 3 plies.
    """
    moves = _legal_moves(state)
    if not moves:
        return -1

    best_action = None
    best_score = -float('inf')
    alpha = -float('inf')
    beta = float('inf')

    for action in _ordered(state, moves):
        child = _apply(state, action)
        score = -_alpha_beta(child, depth - 1, -beta, -alpha)
        if score > best_score:
            best_score = score
            best_action = action
        alpha = max(alpha, score)

    return int(best_action)


def _alpha_beta(state, depth: int, alpha: float, beta: float) -> float:
    """Negamax alpha-beta search. Scores are from the side to move's view."""
    done, z = state.is_terminal(adjudicate=False)
    if done:
        # Scale by remaining depth: a win found sooner scores higher, a loss
        # later scores higher. Without this a win now and a win in 2 plies
        # tie, and the search can keep postponing a Raja capture.
        return z * (WIN_SCORE + depth)

    if depth <= 0:
        return _evaluate_position(state)

    best = -float('inf')
    for action in _ordered(state, _legal_moves(state)):
        child = _apply(state, action)
        score = -_alpha_beta(child, depth - 1, -beta, -alpha)
        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break  # Cutoff

    return best


def _mobility_difference(state) -> int:
    """Our legal-move count minus the opponent's in the same position."""
    if not _is_chaturanga(state):
        return 0
    from chaturanga.board import State
    from chaturanga.moves import generate_moves
    opponent_view = State(board=state.board, side_to_move=-state.side_to_move,
                          ply_count=state.ply_count)
    return len(generate_moves(state)) - len(generate_moves(opponent_view))


def _evaluate_position(state) -> float:
    """
    Static evaluation: material + mobility, both as us-minus-them.
    Returns score from the side-to-move's perspective.

    Mobility used to count only the side to move's moves, which made leaf
    scores depend on whether the depth was odd or even.
    """
    side = state.side_to_move

    if _is_chaturanga(state):
        material = state.material_score(side) - state.material_score(-side)
    else:
        material = 0.0

    return material + MOBILITY_WEIGHT * _mobility_difference(state)
