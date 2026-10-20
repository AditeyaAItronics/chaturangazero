"""
Deliberately slow, obviously-correct reference engine for differential testing.

Written independently from the production engine (board.py, moves.py,
terminal.py) and directly from RULES.md. Naive code, one function per piece,
no optimisation, no precomputed tables, no caching.

It has its own representation: an 8x8 grid indexed grid[rank][file] (rank 0 =
rank 1, file 0 = file a), positive pieces White, negative Black, using the
RULES.md piece codes restated below. Production code is touched only at the
boundary — a production State is read (board / side_to_move / ply_count) on the
way in, and ref_apply_move() builds one on the way out so results can be
compared. Move generation, move application (including promotion), the
terminal ladder and material adjudication are all re-implemented here, so
differential perft exercises every rule (needtofix C6, L7).
"""

# RULES.md values, deliberately restated rather than imported
WHITE, BLACK = 1, -1
EMPTY, PADATI, ASHVA, GAJA, RATHA, MANTRI, RAJA = 0, 1, 2, 3, 4, 5, 6
REF_MOVE_CAP = 300
REF_ADJUDICATE_MARGIN = 4.0
REF_PIECE_VALUES = {RATHA: 5.0, ASHVA: 3.0, GAJA: 1.5, MANTRI: 1.5, PADATI: 1.0}


class RefPosition:
    """The reference engine's own position: grid[rank][file], side to move, ply."""
    __slots__ = ('grid', 'side', 'ply')

    def __init__(self, grid, side: int, ply: int):
        self.grid = grid
        self.side = side
        self.ply = ply


def _to_ref(position) -> RefPosition:
    """Accept a RefPosition, or read a production State at the boundary."""
    if isinstance(position, RefPosition):
        return position
    flat = [int(x) for x in position.board]
    grid = [flat[rank * 8:(rank + 1) * 8] for rank in range(8)]
    return RefPosition(grid, int(position.side_to_move), int(position.ply_count))


def encode_action(from_sq: int, to_sq: int) -> int:
    """action = from_square * 64 + to_square (RULES §3.1), square = rank * 8 + file."""
    return from_sq * 64 + to_sq


def _square(rank: int, file: int) -> int:
    return rank * 8 + file


def _inside(rank: int, file: int) -> bool:
    return 0 <= rank < 8 and 0 <= file < 8


def _is_enemy(grid, rank, file, side) -> bool:
    piece = grid[rank][file]
    return piece != EMPTY and (piece > 0) != (side == WHITE)


def _is_friendly(grid, rank, file, side) -> bool:
    piece = grid[rank][file]
    return piece != EMPTY and (piece > 0) == (side == WHITE)


def _step_targets(grid, rank, file, side, deltas, moves):
    """One-shot targets (leapers and one-step pieces): empty or enemy squares."""
    for dr, df in deltas:
        r, f = rank + dr, file + df
        if _inside(r, f) and not _is_friendly(grid, r, f, side):
            moves.append(encode_action(_square(rank, file), _square(r, f)))


def _ref_padati(grid, rank, file, side, moves):
    """Padati: one step forward onto an empty square; captures one step diagonally forward."""
    forward = 1 if side == WHITE else -1
    r = rank + forward
    if not (0 <= r < 8):
        return
    if grid[r][file] == EMPTY:
        moves.append(encode_action(_square(rank, file), _square(r, file)))
    for df in (-1, 1):
        f = file + df
        if 0 <= f < 8 and _is_enemy(grid, r, f, side):
            moves.append(encode_action(_square(rank, file), _square(r, f)))


def _ref_ashva(grid, rank, file, side, moves):
    """Ashva: standard knight leap."""
    _step_targets(grid, rank, file, side,
                  [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)], moves)


def _ref_gaja(grid, rank, file, side, moves):
    """Gaja: exactly two squares diagonally, jumping. The middle square does not matter."""
    _step_targets(grid, rank, file, side, [(-2, -2), (-2, 2), (2, -2), (2, 2)], moves)


def _ref_mantri(grid, rank, file, side, moves):
    """Mantri: exactly one square diagonally."""
    _step_targets(grid, rank, file, side, [(-1, -1), (-1, 1), (1, -1), (1, 1)], moves)


def _ref_raja(grid, rank, file, side, moves):
    """Raja: one square in any direction. NO CHECK FILTERING."""
    _step_targets(grid, rank, file, side,
                  [(dr, df) for dr in (-1, 0, 1) for df in (-1, 0, 1) if (dr, df) != (0, 0)], moves)


def _ref_ratha(grid, rank, file, side, moves):
    """Ratha: any distance orthogonally, stopped by the first occupied square (captured if enemy)."""
    for dr, df in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        r, f = rank + dr, file + df
        while _inside(r, f):
            if grid[r][f] == EMPTY:
                moves.append(encode_action(_square(rank, file), _square(r, f)))
            else:
                if _is_enemy(grid, r, f, side):
                    moves.append(encode_action(_square(rank, file), _square(r, f)))
                break
            r, f = r + dr, f + df


_PIECE_GENERATORS = {
    PADATI: _ref_padati, ASHVA: _ref_ashva, GAJA: _ref_gaja,
    RATHA: _ref_ratha, MANTRI: _ref_mantri, RAJA: _ref_raja,
}


def ref_generate_moves(position) -> list[int]:
    """
    Reference move generation — deliberately simple and obviously correct.
    Accepts a RefPosition or a production State. Returns sorted action ids.
    """
    pos = _to_ref(position)
    moves = []
    for rank in range(8):
        for file in range(8):
            piece = pos.grid[rank][file]
            if piece == EMPTY or (piece > 0) != (pos.side == WHITE):
                continue
            _PIECE_GENERATORS[abs(piece)](pos.grid, rank, file, pos.side, moves)
    return sorted(moves)


def ref_apply_position(position, action: int) -> RefPosition:
    """Apply a move: displace any piece on the target, promote a Padati on the last rank."""
    pos = _to_ref(position)
    from_sq, to_sq = action // 64, action % 64
    fr, ff, tr, tf = from_sq // 8, from_sq % 8, to_sq // 8, to_sq % 8
    grid = [row[:] for row in pos.grid]
    piece = grid[fr][ff]
    grid[fr][ff] = EMPTY
    grid[tr][tf] = piece
    last_rank = 7 if pos.side == WHITE else 0
    if abs(piece) == PADATI and tr == last_rank:
        grid[tr][tf] = MANTRI if piece > 0 else -MANTRI
    return RefPosition(grid, -pos.side, pos.ply + 1)


def ref_apply_move(position, action: int):
    """ref_apply_position(), returned as a production State so results can be compared."""
    import numpy as np
    from .board import State
    pos = ref_apply_position(position, action)
    board = np.array([pos.grid[r][f] for r in range(8) for f in range(8)], dtype=np.int8)
    return State(board=board, side_to_move=pos.side, ply_count=pos.ply)


def _ref_pieces(grid, side) -> list[int]:
    """Piece types (absolute values) belonging to `side`."""
    return [abs(p) for row in grid for p in row if p != EMPTY and (p > 0) == (side == WHITE)]


def ref_is_terminal(position) -> tuple[bool, float]:
    """Terminal ladder from RULES.md, in order, from the side to move's view."""
    pos = _to_ref(position)
    mine = _ref_pieces(pos.grid, pos.side)
    theirs = _ref_pieces(pos.grid, -pos.side)

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
    if len(ref_generate_moves(pos)) == 0:
        return True, 1.0
    # 5. Move cap: adjudicate on material
    if pos.ply >= REF_MOVE_CAP:
        diff = (sum(REF_PIECE_VALUES.get(p, 0.0) for p in mine)
                - sum(REF_PIECE_VALUES.get(p, 0.0) for p in theirs))
        if diff >= REF_ADJUDICATE_MARGIN:
            return True, 1.0
        if diff <= -REF_ADJUDICATE_MARGIN:
            return True, -1.0
        return True, 0.0
    return False, 0.0


def ref_perft(position, depth: int) -> int:
    """Reference perft on the reference engine's own representation."""
    return _ref_perft(_to_ref(position), depth)


def _ref_perft(pos: RefPosition, depth: int) -> int:
    if depth == 0:
        return 1
    if ref_is_terminal(pos)[0]:
        return 0
    moves = ref_generate_moves(pos)
    if depth == 1:
        return len(moves)   # every child is a depth-0 leaf
    return sum(_ref_perft(ref_apply_position(pos, a), depth - 1) for a in moves)
