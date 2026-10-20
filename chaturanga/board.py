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

# Material value by piece type (index = abs(piece code)); Raja counts 0
_VALUE_BY_TYPE = np.array([0.0, PIECE_VALUES[PADATI], PIECE_VALUES[ASHVA], PIECE_VALUES[GAJA],
                           PIECE_VALUES[RATHA], PIECE_VALUES[MANTRI], PIECE_VALUES[RAJA]])

MOVE_CAP = 300               # default; the live value comes from get_move_cap()
ADJUDICATE_MARGIN = 4.0      # default; the live value comes from get_adjudicate_margin()

# Runtime rule settings (§2.2). The training loop sets these from
# config['play'] so `move_cap: 200` in the YAML actually takes effect.
_RULES = {'move_cap': MOVE_CAP, 'adjudicate_margin': ADJUDICATE_MARGIN}


# The ruleset the engine implements (RULES.md §2.1). These are not settings:
# changing a game rule means changing the engine and RULES.md (4 sign-offs).
# A config's `rules:` block is checked against this, so editing the YAML can
# never silently claim a rule the engine doesn't play (needtofixagain N8).
RULESET = {
    'castling': False,
    'pawn_double_step': False,
    'promotion': 'mantri_only',
    'check_rule': 'none',
    'stalemate': 'side_to_move_wins',
    'repetition_draw': False,
    'fifty_move_rule': False,
    'bare_king': 'loss',
}


def check_ruleset(rules: Optional[dict]) -> None:
    """Raise ValueError if a config's `rules:` block disagrees with RULESET. None is allowed."""
    if rules is None:
        return
    problems = []
    for key, value in rules.items():
        if key not in RULESET:
            problems.append(f"unknown rule {key!r}")
        elif value != RULESET[key]:
            problems.append(f"{key}: config says {value!r}, the engine plays {RULESET[key]!r}")
    if problems:
        raise ValueError("config 'rules:' block does not match the engine (RULES.md §2.1): "
                         + "; ".join(problems) + ". Game rules are fixed in code, not config.")


def configure_rules(move_cap: Optional[int] = None,
                    adjudicate_margin: Optional[float] = None) -> None:
    """Set the move cap and material-adjudication margin; None keeps the current value."""
    if move_cap is not None:
        if int(move_cap) <= 0:
            raise ValueError(f"move_cap must be positive, got {move_cap}")
        _RULES['move_cap'] = int(move_cap)
    if adjudicate_margin is not None:
        _RULES['adjudicate_margin'] = float(adjudicate_margin)


def get_move_cap() -> int:
    return _RULES['move_cap']


def get_adjudicate_margin() -> float:
    return _RULES['adjudicate_margin']

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


_FEN_PIECES = {'P': PADATI, 'N': ASHVA, 'B': GAJA, 'R': RATHA, 'M': MANTRI, 'K': RAJA}


def to_fen(state: 'State') -> str:
    """
    FEN-style text for a position: ranks 8 -> 1 separated by '/', letters as in
    PIECE_NAMES (uppercase White), digits for runs of empty squares, then side
    to move ('w'/'b') and ply count. Example start position:
        'rnbmkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBMKBNR w 0'
    """
    rows = []
    for rank in range(7, -1, -1):
        row, empty = '', 0
        for file in range(8):
            p = int(state.board[sq(rank, file)])
            if p == 0:
                empty += 1
                continue
            if empty:
                row += str(empty)
                empty = 0
            row += PIECE_NAMES[p]
        rows.append(row + (str(empty) if empty else ''))
    return f"{'/'.join(rows)} {'w' if state.side_to_move == WHITE else 'b'} {state.ply_count}"


def from_fen(fen: str) -> 'State':
    """Inverse of to_fen(). The ply count is optional (default 0)."""
    fields = fen.split()
    rows = fields[0].split('/')
    if len(rows) != 8:
        raise ValueError(f"expected 8 ranks in {fen!r}")
    board = np.zeros(64, dtype=np.int8)
    for i, row in enumerate(rows):
        rank, file = 7 - i, 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
                continue
            piece = _FEN_PIECES[ch.upper()]
            board[sq(rank, file)] = piece if ch.isupper() else -piece
            file += 1
        if file != 8:
            raise ValueError(f"rank {rank + 1} of {fen!r} has {file} files")
    side = WHITE if len(fields) < 2 or fields[1] == 'w' else BLACK
    ply = int(fields[2]) if len(fields) > 2 else 0
    return State(board=board, side_to_move=side, ply_count=ply)


class State:
    """
    Complete game state for Chaturanga.
    
    Conforms to the §3.4 interface contract.

    A State is treated as immutable once built: legal moves and the terminal
    result are computed at most once and cached (needtofix S3). Never modify
    `board` in place after calling legal_moves()/is_terminal(); build a new
    State (apply(), copy(), or State(board=...)) instead.
    """
    __slots__ = ('board', '_side', '_ply', '_hash', '_moves', '_terminal')
        
    def __init__(self, board: Optional[np.ndarray] = None,
                 side_to_move: int = WHITE, ply_count: int = 0):
        if board is None:
            self.board = INITIAL_BOARD.copy()
        else:
            self.board = board.copy()
        self._side = side_to_move
        self._ply = ply_count
        self._hash = None
        self._moves = None       # cached generate_moves() result
        self._terminal = None    # cached (settings, evaluate_terminal() result)
        
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
        signed = self.board * side
        squares = np.flatnonzero(signed > 0)
        return [(int(s), int(signed[s])) for s in squares]

    def has_raja(self, side: int) -> bool:
        """Check if the given side has a Raja on the board."""
        return bool(np.any(self.board == RAJA * side))
        
    def piece_count(self, side: int) -> int:
        """Count total pieces for the given side."""
        if side == WHITE:
            return int(np.sum(self.board > 0))
        else:
            return int(np.sum(self.board < 0))
    
    def material_score(self, side: int) -> float:
        """Compute material value for the given side (excluding Raja)."""
        signed = self.board * side
        return float(_VALUE_BY_TYPE[signed[signed > 0]].sum())
    
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
        s._moves = None
        s._terminal = None
        return s

    def __repr__(self) -> str:
        return self.display()
    
    # --- §3.4 Interface Contract Methods ---
    # These delegate to the standalone functions so that MCTS and the pipeline
    # can call state.is_terminal(), state.legal_moves(), etc. uniformly.
    
    def is_terminal(self, adjudicate: bool = True) -> tuple[bool, float]:
        """
        Check terminal conditions. Returns (done, z) from side-to-move's view.
        adjudicate=False scores a move-cap ending as a draw (arena/evaluation, §2.2).
        """
        from .terminal import is_terminal as _is_terminal
        return _is_terminal(self, adjudicate=adjudicate)
    
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
    
    def orient_policy(self, policy: 'np.ndarray') -> 'np.ndarray':
        """
        Convert a (4096,) policy between the absolute frame (legal_moves, MCTS)
        and the canonical frame used by encode() and the network. Its own
        inverse, so it is applied to network output and to training targets alike.
        """
        from .encoding import orient_policy
        return orient_policy(self, policy)

    def mirror_action_permutation(self) -> 'np.ndarray':
        """Return the (4096,) action permutation array for left-right mirroring."""
        from .encoding import MIRROR_PERM
        return MIRROR_PERM
