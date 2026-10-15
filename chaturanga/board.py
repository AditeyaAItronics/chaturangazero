"""
Chaturanga board representation and core game state.

Board is 8x8, stored as a flat numpy array of int8.
Piece encoding:
    0 = empty
    +1 Padati, +2 Ashva, +3 Gaja, +4 Ratha, +5 Mantri, +6 Raja  (White)
    -1 Padati, -2 Ashva, -3 Gaja, -4 Ratha, -5 Mantri, -6 Raja  (Black)

Square indexing: rank-major, 0 = a1, 1 = b1, ..., 63 = h8.
    sq = rank * 8 + file  (rank 0 = rank 1 in human notation)

Side to move: WHITE = 1, BLACK = -1.
"""

import numpy as np
from typing import Hashable, Optional
import copy

# --- Constants ---
WHITE = 1
BLACK = -1

EMPTY = 0
PADATI = 1   # Pawn
ASHVA = 2    # Knight
GAJA = 3     # Elephant (2-sq diagonal leaper)
RATHA = 4    # Rook (sliding orthogonal)
MANTRI = 5   # Minister (1-sq diagonal)
RAJA = 6     # King

PIECE_NAMES = {
    0: '.',
    1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'M', 6: 'K',
    -1: 'p', -2: 'n', -3: 'b', -4: 'r', -5: 'm', -6: 'k',
}

PIECE_VALUES = {
    PADATI: 1.0,
    ASHVA: 3.0,
    GAJA: 1.5,
    RATHA: 5.0,
    MANTRI: 1.5,
    RAJA: 0.0,  # Not counted in material for adjudication
}

MOVE_CAP = 300

# --- Board Geometry ---
def sq(rank: int, file: int) -> int:
    """Convert rank, file (0-indexed) to square index."""
    return rank * 8 + file

def rank_of(square: int) -> int:
    return square >> 3

def file_of(square: int) -> int:
    return square & 7

def on_board(rank: int, file: int) -> bool:
    return 0 <= rank < 8 and 0 <= file < 8

# --- Initial Position ---
INITIAL_BOARD = np.zeros(64, dtype=np.int8)

# White back rank (rank 0)
_WHITE_BACK = [RATHA, ASHVA, GAJA, MANTRI, RAJA, GAJA, ASHVA, RATHA]
for i, piece in enumerate(_WHITE_BACK):
    INITIAL_BOARD[sq(0, i)] = piece

# White pawns (rank 1)
for i in range(8):
    INITIAL_BOARD[sq(1, i)] = PADATI

# Black pawns (rank 6)
for i in range(8):
    INITIAL_BOARD[sq(6, i)] = -PADATI

# Black back rank (rank 7)
for i, piece in enumerate(_WHITE_BACK):
    INITIAL_BOARD[sq(7, i)] = -piece


class State:
    """
    Complete game state for Chaturanga.
    
    Conforms to the §3.4 interface contract.
    """
    __slots__ = ('board', '_side', '_ply', '_hash')
    
    def __init__(self, board: Optional[np.ndarray] = None,
                 side_to_move: int = WHITE, ply_count: int = 0):
        if board is None:
            self.board = INITIAL_BOARD.copy()
        else:
            self.board = board.copy()
        self._side = side_to_move
        self._ply = ply_count
        self._hash = None
    
    @property
    def side_to_move(self) -> int:
        return self._side
    
    @property
    def ply_count(self) -> int:
        return self._ply
    
    def piece_at(self, square: int) -> int:
        """Return piece code at the given square."""
        return int(self.board[square])
    
    def find_pieces(self, side: int) -> list[tuple[int, int]]:
        """Return list of (square, piece_type) for all pieces of the given side."""
        pieces = []
        if side == WHITE:
            for s in range(64):
                p = self.board[s]
                if p > 0:
                    pieces.append((s, int(p)))
        else:
            for s in range(64):
                p = self.board[s]
                if p < 0:
                    pieces.append((s, int(-p)))
        return pieces
    
    def has_raja(self, side: int) -> bool:
        """Check if the given side has a Raja on the board."""
        target = RAJA * side
        for s in range(64):
            if self.board[s] == target:
                return True
        return False
    
    def piece_count(self, side: int) -> int:
        """Count total pieces for the given side."""
        if side == WHITE:
            return int(np.sum(self.board > 0))
        else:
            return int(np.sum(self.board < 0))
    
    def material_score(self, side: int) -> float:
        """Compute material value for the given side (excluding Raja)."""
        score = 0.0
        if side == WHITE:
            for s in range(64):
                p = self.board[s]
                if p > 0:
                    score += PIECE_VALUES.get(int(p), 0.0)
        else:
            for s in range(64):
                p = self.board[s]
                if p < 0:
                    score += PIECE_VALUES.get(int(-p), 0.0)
        return score
    
    def display(self) -> str:
        """Return a human-readable board string."""
        lines = []
        for rank in range(7, -1, -1):
            row = [f"{rank+1} "]
            for file in range(8):
                p = self.board[sq(rank, file)]
                row.append(f" {PIECE_NAMES[int(p)]} ")
            lines.append("".join(row))
        lines.append("   a  b  c  d  e  f  g  h")
        side_str = "White" if self._side == WHITE else "Black"
        lines.append(f"Side to move: {side_str}, Ply: {self._ply}")
        return "\n".join(lines)
    
    def key(self) -> Hashable:
        """
        Transposition key. Position + side-to-move is a complete Markov state
        (no castling rights, no en passant, no repetition counter).
        """
        if self._hash is None:
            self._hash = (self.board.tobytes(), self._side)
        return self._hash
    
    def copy(self) -> 'State':
        """Return a deep copy of this state."""
        s = State.__new__(State)
        s.board = self.board.copy()
        s._side = self._side
        s._ply = self._ply
        s._hash = None
        return s

    def __repr__(self) -> str:
        return self.display()
    
    # --- §3.4 Interface Contract Methods ---
    # These delegate to the standalone functions so that MCTS and the pipeline
    # can call state.is_terminal(), state.legal_moves(), etc. uniformly.
    
    def is_terminal(self) -> tuple[bool, float]:
        """Check terminal conditions. Returns (done, z) from side-to-move's view."""
        from .terminal import is_terminal as _is_terminal
        return _is_terminal(self)
    
    def legal_moves(self) -> list[int]:
        """Generate all legal moves (pseudo-legal == legal in this ruleset)."""
        from .moves import generate_moves
        return generate_moves(self)
    
    def apply(self, action: int) -> 'State':
        """Apply a move and return a new State."""
        from .moves import apply_move
        return apply_move(self, action)
    
    def encode(self) -> 'np.ndarray':
        """Encode as (14, 8, 8) float32 tensor for the neural network."""
        from .encoding import encode_state
        return encode_state(self)
    
    def legal_mask(self) -> 'np.ndarray':
        """Return (4096,) boolean mask of legal actions."""
        from .moves import legal_mask as _legal_mask
        return _legal_mask(self)
    
    def mirror(self) -> tuple['State', 'np.ndarray']:
        """Return (mirrored_state, action_index_permutation) for augmentation."""
        from .encoding import mirror_state, MIRROR_PERM
        return mirror_state(self), MIRROR_PERM
    
    def mirror_action_permutation(self) -> 'np.ndarray':
        """Return the (4096,) action permutation array for left-right mirroring."""
        from .encoding import MIRROR_PERM
        return MIRROR_PERM
