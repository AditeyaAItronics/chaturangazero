"""
Pseudo-legal move generation for Chaturanga.

THERE IS NO CHECK. Every pseudo-legal move IS legal.
No is_square_attacked() function exists or should ever exist.
"""

import numpy as np
from .board import (
    State, WHITE, BLACK, EMPTY,
    PADATI, ASHVA, GAJA, RATHA, MANTRI, RAJA,
    rank_of, file_of, sq, on_board, get_move_cap,
)


def encode_action(from_sq: int, to_sq: int) -> int:
    """Encode a move as action = from_square * 64 + to_square."""
    return from_sq * 64 + to_sq


def decode_action(action: int) -> tuple[int, int]:
    """Decode action into (from_square, to_square)."""
    return action // 64, action % 64


# --- Precomputed move tables (needtofix S4) ---------------------------------
# For every square: the target squares of each leaper, and the four Ratha
# rays, in the same direction order the original per-piece generators used —
# so generate_moves() returns moves in exactly the same order as before
# (MCTS breaks ties by that order).

def _targets(deltas):
    table = []
    for s in range(64):
        r, f = rank_of(s), file_of(s)
        table.append(tuple(sq(r + dr, f + df) for dr, df in deltas if on_board(r + dr, f + df)))
    return tuple(table)


_ASHVA_TARGETS = _targets([(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)])
_GAJA_TARGETS = _targets([(-2, -2), (-2, 2), (2, -2), (2, 2)])      # jumps: the middle square is irrelevant
_MANTRI_TARGETS = _targets([(-1, -1), (-1, 1), (1, -1), (1, 1)])
_RAJA_TARGETS = _targets([(dr, df) for dr in (-1, 0, 1) for df in (-1, 0, 1) if (dr, df) != (0, 0)])
_LEAPER_TARGETS = {ASHVA: _ASHVA_TARGETS, GAJA: _GAJA_TARGETS,
                   MANTRI: _MANTRI_TARGETS, RAJA: _RAJA_TARGETS}


def _rays(s):
    r, f = rank_of(s), file_of(s)
    rays = []
    for dr, df in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        ray, nr, nf = [], r + dr, f + df
        while on_board(nr, nf):
            ray.append(sq(nr, nf))
            nr, nf = nr + dr, nf + df
        rays.append(tuple(ray))
    return tuple(rays)


_RATHA_RAYS = tuple(_rays(s) for s in range(64))

# Padati: forward square (None off-board) and diagonal-capture squares (file -1, then +1)
_PADATI_FORWARD = {
    side: tuple(sq(rank_of(s) + side, file_of(s)) if 0 <= rank_of(s) + side < 8 else None
                for s in range(64))
    for side in (WHITE, BLACK)
}
_PADATI_CAPTURES = {
    side: tuple(tuple(sq(rank_of(s) + side, file_of(s) + df) for df in (-1, 1)
                      if on_board(rank_of(s) + side, file_of(s) + df))
                for s in range(64))
    for side in (WHITE, BLACK)
}


def generate_moves(state: State) -> list[int]:
    """
    Generate all pseudo-legal moves for the side to move.

    PSEUDO-LEGAL == LEGAL in this ruleset (no check).
    Returns a list of action IDs in [0, 4096). The result is cached on the
    State (S3); each call returns a fresh copy, so callers may modify it.
    """
    cached = getattr(state, '_moves', None)
    if cached is not None:
        return list(cached)

    board = state.board.tolist()          # plain ints: much faster than numpy scalars
    side = state.side_to_move
    moves: list[int] = []
    append = moves.append

    for s, piece in enumerate(board):
        kind = piece * side               # > 0 only for the side to move's pieces
        if kind <= 0:
            continue
        base = s * 64
        if kind == PADATI:
            fwd = _PADATI_FORWARD[side][s]
            if fwd is not None:
                if board[fwd] == EMPTY:
                    append(base + fwd)
                for t in _PADATI_CAPTURES[side][s]:
                    if board[t] * side < 0:            # captures enemy pieces only
                        append(base + t)
        elif kind == RATHA:
            for ray in _RATHA_RAYS[s]:
                for t in ray:
                    target = board[t]
                    if target == EMPTY:
                        append(base + t)
                        continue
                    if target * side < 0:              # blocked: capture if enemy, then stop
                        append(base + t)
                    break
        else:
            # Ashva, Gaja, Mantri, Raja: empty or enemy-occupied target.
            # NO CHECK FILTERING — the Raja may step onto guarded squares.
            for t in _LEAPER_TARGETS[kind][s]:
                if board[t] * side <= 0:
                    append(base + t)

    if hasattr(state, '_moves'):
        state._moves = tuple(moves)
    return moves


def apply_move(state: State, action: int) -> 'State':
    """
    Apply a move and return a NEW state. Does not mutate the original.
    
    Handles:
    - Normal moves and captures
    - Padati promotion to Mantri on last rank (forced, no choice)
    """
    # §2.2 hard invariant: the move cap is the only termination guarantee, so a
    # move from a position at the cap means something skipped is_terminal().
    assert state.ply_count < get_move_cap(), \
        "game exceeded cap — termination logic is broken"
    
    from_sq, to_sq = decode_action(action)
    
    new_board = state.board.copy()
    piece = new_board[from_sq]
    
    # Move piece
    new_board[to_sq] = piece
    new_board[from_sq] = EMPTY
    
    # Check for Padati promotion
    piece_type = abs(int(piece))
    if piece_type == PADATI:
        to_rank = rank_of(to_sq)
        # White promotes on rank 7, Black promotes on rank 0
        if (state.side_to_move == WHITE and to_rank == 7) or \
           (state.side_to_move == BLACK and to_rank == 0):
            # Promote to Mantri
            new_board[to_sq] = MANTRI * state.side_to_move
    
    new_state = State.__new__(State)
    new_state.board = new_board
    new_state._side = -state.side_to_move
    new_state._ply = state.ply_count + 1
    new_state._hash = None
    new_state._moves = None
    new_state._terminal = None
        
    return new_state


def legal_mask(state: State) -> np.ndarray:
    """
    Return a boolean mask of shape (4096,) with True at legal action indices.
    """
    mask = np.zeros(4096, dtype=bool)
    mask[generate_moves(state)] = True
    return mask
