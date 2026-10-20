"""
Deliberately slow, obviously-correct reference engine for differential testing.

Written independently from the production engine (moves.py, terminal.py).
Uses naive 8x8 iteration, one function per piece, no optimization.
Written directly from RULES.md.

Only the State container (board array, side to move, ply count) and the
square/piece constants are shared with production. Move generation, move
application (including promotion), the terminal ladder and material
adjudication are all re-implemented here, so differential perft exercises
every rule rather than only piece movement.
"""

import numpy as np
from .board import (
    State, WHITE, BLACK, EMPTY,
    PADATI, ASHVA, GAJA, RATHA, MANTRI, RAJA,
    rank_of, file_of, sq, on_board,
)

# RULES.md values, deliberately restated rather than imported
REF_MOVE_CAP = 300
REF_ADJUDICATE_MARGIN = 4.0
REF_PIECE_VALUES = {RATHA: 5.0, ASHVA: 3.0, GAJA: 1.5, MANTRI: 1.5, PADATI: 1.0}


def encode_action(from_sq: int, to_sq: int) -> int:
    """action = from_square * 64 + to_square (RULES §3.1)."""
    return from_sq * 64 + to_sq


def ref_generate_moves(state: State) -> list[int]:
    """
    Reference move generation — deliberately simple and obviously correct.
    One function per piece type, no optimization.
    """
    board = state.board
    side = state.side_to_move
    moves = []
    
    for square in range(64):
        piece = board[square]
        if piece == EMPTY:
            continue
        
        # Check ownership
        if side == WHITE and piece <= 0:
            continue
        if side == BLACK and piece >= 0:
            continue
        
        piece_type = abs(int(piece))
        
        if piece_type == PADATI:
            _ref_padati(board, square, side, moves)
        elif piece_type == ASHVA:
            _ref_ashva(board, square, side, moves)
        elif piece_type == GAJA:
            _ref_gaja(board, square, side, moves)
        elif piece_type == RATHA:
            _ref_ratha(board, square, side, moves)
        elif piece_type == MANTRI:
            _ref_mantri(board, square, side, moves)
        elif piece_type == RAJA:
            _ref_raja(board, square, side, moves)
    
    return sorted(moves)


def _is_enemy(board, sq_idx, side):
    """Check if the piece at sq_idx is an enemy piece."""
    p = board[sq_idx]
    if p == EMPTY:
        return False
    if side == WHITE:
        return p < 0
    else:
        return p > 0


def _is_friendly(board, sq_idx, side):
    """Check if the piece at sq_idx is a friendly piece."""
    p = board[sq_idx]
    if p == EMPTY:
        return False
    if side == WHITE:
        return p > 0
    else:
        return p < 0


def _ref_padati(board, s, side, moves):
    """Padati: one step forward, diagonal capture. No double step. No en passant."""
    r, f = rank_of(s), file_of(s)
    direction = 1 if side == WHITE else -1
    
    # Forward (non-capture)
    nr = r + direction
    if 0 <= nr < 8:
        target = sq(nr, f)
        if board[target] == EMPTY:
            moves.append(encode_action(s, target))
    
    # Diagonal captures
    for df in [-1, 1]:
        nf = f + df
        nr = r + direction
        if 0 <= nr < 8 and 0 <= nf < 8:
            target = sq(nr, nf)
            if _is_enemy(board, target, side):
                moves.append(encode_action(s, target))


def _ref_ashva(board, s, side, moves):
    """Ashva: standard knight leap."""
    r, f = rank_of(s), file_of(s)
    knight_offsets = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1),
    ]
    for dr, df in knight_offsets:
        nr, nf = r + dr, f + df
        if 0 <= nr < 8 and 0 <= nf < 8:
            target = sq(nr, nf)
            if not _is_friendly(board, target, side):
                moves.append(encode_action(s, target))


def _ref_gaja(board, s, side, moves):
    """Gaja: exactly two squares diagonally, jumping. Cannot move one square."""
    r, f = rank_of(s), file_of(s)
    for dr, df in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
        nr, nf = r + dr, f + df
        if 0 <= nr < 8 and 0 <= nf < 8:
            target = sq(nr, nf)
            if not _is_friendly(board, target, side):
                moves.append(encode_action(s, target))


def _ref_ratha(board, s, side, moves):
    """Ratha: any distance orthogonally, blocked by first occupied square."""
    r, f = rank_of(s), file_of(s)
    for dr, df in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        cr, cf = r + dr, f + df
        while 0 <= cr < 8 and 0 <= cf < 8:
            target = sq(cr, cf)
            p = board[target]
            if p == EMPTY:
                moves.append(encode_action(s, target))
            elif _is_enemy(board, target, side):
                moves.append(encode_action(s, target))
                break
            else:
                # Friendly piece, blocked
                break
            cr += dr
            cf += df


def _ref_mantri(board, s, side, moves):
    """Mantri: exactly one square diagonally."""
    r, f = rank_of(s), file_of(s)
    for dr, df in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nr, nf = r + dr, f + df
        if 0 <= nr < 8 and 0 <= nf < 8:
            target = sq(nr, nf)
            if not _is_friendly(board, target, side):
                moves.append(encode_action(s, target))


def _ref_raja(board, s, side, moves):
    """Raja: one square in any direction. NO CHECK FILTERING."""
    r, f = rank_of(s), file_of(s)
    for dr in [-1, 0, 1]:
        for df in [-1, 0, 1]:
            if dr == 0 and df == 0:
                continue
            nr, nf = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                target = sq(nr, nf)
                if not _is_friendly(board, target, side):
                    moves.append(encode_action(s, target))


def ref_apply_move(state: State, action: int) -> State:
    """Apply a move: displace any piece on the target, promote a Padati on the last rank."""
    from_square = action // 64
    to_square = action % 64
    board = [int(x) for x in state.board]
    piece = board[from_square]
    board[from_square] = EMPTY
    board[to_square] = piece
    
    last_rank = 7 if state.side_to_move == WHITE else 0
    if abs(piece) == PADATI and to_square // 8 == last_rank:
        board[to_square] = MANTRI if piece > 0 else -MANTRI
    
    return State(board=np.array(board, dtype=np.int8),
                 side_to_move=-state.side_to_move,
                 ply_count=state.ply_count + 1)


def _ref_pieces(board, side) -> list[int]:
    """Piece types (absolute values) belonging to `side`."""
    if side == WHITE:
        return [int(p) for p in board if p > 0]
    return [int(-p) for p in board if p < 0]


def ref_is_terminal(state: State) -> tuple[bool, float]:
    """Terminal ladder from RULES.md, in order, from the side to move's view."""
    me = state.side_to_move
    mine = _ref_pieces(state.board, me)
    theirs = _ref_pieces(state.board, -me)
    
    # 1. Raja captured
    if RAJA not in mine:
        return True, -1.0
    # 2. King vs King
    if mine == [RAJA] and theirs == [RAJA]:
        return True, 0.0
    # 3. Bare king
    if mine == [RAJA]:
        return True, -1.0
    # 4. Stalemate: the player with no legal move WINS
    if len(ref_generate_moves(state)) == 0:
        return True, 1.0
    # 5. Move cap: adjudicate on material
    if state.ply_count >= REF_MOVE_CAP:
        diff = (sum(REF_PIECE_VALUES.get(p, 0.0) for p in mine)
                - sum(REF_PIECE_VALUES.get(p, 0.0) for p in theirs))
        if diff >= REF_ADJUDICATE_MARGIN:
            return True, 1.0
        if diff <= -REF_ADJUDICATE_MARGIN:
            return True, -1.0
        return True, 0.0
    return False, 0.0


def ref_perft(state: State, depth: int) -> int:
    """Reference perft — no production code on the path except the State container."""
    if depth == 0:
        return 1
    
    done, _ = ref_is_terminal(state)
    if done:
        return 0
    
    count = 0
    for action in ref_generate_moves(state):
        count += ref_perft(ref_apply_move(state, action), depth - 1)
    return count
