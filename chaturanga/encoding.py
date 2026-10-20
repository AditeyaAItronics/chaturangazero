"""
Board encoding for the neural network.

14 planes of 8x8, canonical orientation (side to move is always "us" moving up).

Planes 0-5:  Our Padati, Ashva, Gaja, Ratha, Mantri, Raja
Planes 6-11: Their Padati, Ashva, Gaja, Ratha, Mantri, Raja
Plane 12:    ply_count / move_cap, constant-filled
Plane 13:    All ones — board-edge reference
"""

import numpy as np
from .board import (
    State, WHITE, BLACK,
    PADATI, ASHVA, GAJA, RATHA, MANTRI, RAJA,
    rank_of, file_of, sq, MOVE_CAP,
)

NUM_PLANES = 14

# Piece type to plane index offset (0-indexed)
_PIECE_PLANE = {
    PADATI: 0,
    ASHVA: 1,
    GAJA: 2,
    RATHA: 3,
    MANTRI: 4,
    RAJA: 5,
}


def encode_state(state: State) -> np.ndarray:
    """
    Encode the board state as a (14, 8, 8) float32 tensor.
    
    Always canonical orientation: board is flipped so the side to move
    is always "us," moving up the board. The network learns one player's
    game rather than two.
    """
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    
    side = state.side_to_move
    board = state.board
    
    for s in range(64):
        piece = board[s]
        if piece == EMPTY:
            continue
        
        r, f = rank_of(s), file_of(s)
        
        # Canonical orientation: flip board if Black to move
        if side == BLACK:
            r = 7 - r
        
        piece_type = abs(int(piece))
        is_ours = (piece > 0 and side == WHITE) or (piece < 0 and side == BLACK)
        
        plane_offset = _PIECE_PLANE[piece_type]
        if is_ours:
            planes[plane_offset, r, f] = 1.0
        else:
            planes[6 + plane_offset, r, f] = 1.0
    
    # Plane 12: ply_count / move_cap
    planes[12, :, :] = state.ply_count / MOVE_CAP
    
    # Plane 13: all ones (board-edge reference)
    planes[13, :, :] = 1.0
    
    return planes


def mirror_action_permutation() -> np.ndarray:
    """
    Compute the action index permutation for left-right board mirroring.
    
    action = from_sq * 64 + to_sq
    Mirror: file f -> 7 - f, rank stays the same.
    
    Returns an array P of shape (4096,) such that:
        mirrored_action = P[original_action]
    """
    perm = np.zeros(4096, dtype=np.int32)
    for action in range(4096):
        from_sq = action // 64
        to_sq = action % 64
        
        from_r, from_f = rank_of(from_sq), file_of(from_sq)
        to_r, to_f = rank_of(to_sq), file_of(to_sq)
        
        # Mirror: flip files
        mirror_from = sq(from_r, 7 - from_f)
        mirror_to = sq(to_r, 7 - to_f)
        
        perm[action] = mirror_from * 64 + mirror_to
    
    return perm


# Cache the permutation at module level
MIRROR_PERM = mirror_action_permutation()


def canonical_action_permutation() -> np.ndarray:
    """
    Action permutation matching the canonical (side-to-move-up) board flip.

    encode_state() flips ranks when Black is to move, so the network sees
    Black's position from Black's side. Policy vectors must be flipped the
    same way: rank r -> 7 - r on both from- and to-square (sq ^ 56).

    The permutation is its own inverse, so the same array converts
    absolute -> canonical and canonical -> absolute.
    """
    perm = np.zeros(4096, dtype=np.int32)
    for action in range(4096):
        from_sq, to_sq = action // 64, action % 64
        perm[action] = (from_sq ^ 56) * 64 + (to_sq ^ 56)
    return perm


CANONICAL_PERM = canonical_action_permutation()


def orient_policy(state: State, policy: np.ndarray) -> np.ndarray:
    """
    Convert a (4096,) policy-shaped array between the absolute frame
    (State.legal_moves(), MCTS visit counts) and the canonical frame the
    network reads and writes. Identity for White, rank flip for Black.
    """
    if state.side_to_move == BLACK:
        return policy[CANONICAL_PERM]
    return policy


EMPTY = 0  # re-export for convenience


def mirror_state(state: State) -> 'State':
    """
    Create a left-right mirrored copy of the state.
    
    Files are flipped: a<->h, b<->g, c<->f, d<->e.
    Ranks and piece types remain the same.
    """
    new_board = np.zeros(64, dtype=np.int8)
    for s in range(64):
        r, f = rank_of(s), file_of(s)
        mirror_sq = sq(r, 7 - f)
        new_board[mirror_sq] = state.board[s]
    
    return State(board=new_board, side_to_move=state.side_to_move,
                 ply_count=state.ply_count)
