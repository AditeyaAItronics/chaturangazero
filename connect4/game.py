"""
Connect-4 game implementation — Phase 0 pipeline proof.

Same State interface as Chaturanga so the entire MCTS / training pipeline
can be validated on a game with known optimal play before Chaturanga arrives.

Board: 6 rows x 7 columns, gravity-based.
Action space: 7 (column index 0-6).
Input encoding: 3 planes of 6x7 (our pieces, their pieces, all-ones).
"""

import numpy as np
from typing import Hashable, Optional

ROWS = 6
COLS = 7
ACTION_SPACE = 7
INPUT_PLANES = 3

# Players
PLAYER_1 = 1   # Goes first
PLAYER_2 = -1


class Connect4State:
    """
    Connect-4 game state conforming to the same interface contract as Chaturanga.
    """
    __slots__ = ('board', '_side', '_ply', '_hash', '_last_action', '_terminal')
    
    def __init__(self, board: Optional[np.ndarray] = None,
                 side_to_move: int = PLAYER_1, ply_count: int = 0,
                 last_action: int = -1):
        if board is None:
            self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        else:
            self.board = board.copy()
        self._side = side_to_move
        self._ply = ply_count
        self._hash = None
        self._last_action = last_action
        self._terminal = None    # cached is_terminal() result (needtofixagain N19)
    
    @property
    def side_to_move(self) -> int:
        return self._side
    
    @property
    def ply_count(self) -> int:
        return self._ply
    
    def legal_moves(self) -> list[int]:
        """Return list of column indices that are not full."""
        moves = []
        for col in range(COLS):
            if self.board[ROWS - 1, col] == 0:
                moves.append(col)
        return moves
    
    def apply(self, action: int) -> 'Connect4State':
        """Drop a piece in the given column. Returns new state."""
        new_board = self.board.copy()
        # Find lowest empty row
        for row in range(ROWS):
            if new_board[row, action] == 0:
                new_board[row, action] = self._side
                break
        return Connect4State(
            board=new_board,
            side_to_move=-self._side,
            ply_count=self._ply + 1,
            last_action=action,
        )
    
    def is_terminal(self, adjudicate: bool = True) -> tuple[bool, float]:
        """
        Check terminal conditions.
        Returns (done, z) where z is from side-to-move's perspective.
        """
        # Only the last move can have made four in a row, so only the lines
        # through it are checked, and the result is cached: MCTS asks on every
        # expansion (needtofixagain N19). _check_win() is the full-board scan.
        if self._terminal is None:
            if self._last_action >= 0 and self._last_move_won():
                self._terminal = (True, -1.0)   # the last player won: side to move lost
            elif self._ply >= ROWS * COLS:
                self._terminal = (True, 0.0)    # draw: board full
            else:
                self._terminal = (False, 0.0)
        return self._terminal

    def _last_move_won(self) -> bool:
        """Four in a row through the last piece dropped (by the opponent of the side to move)."""
        col = self._last_action
        column = self.board[:, col]
        row = int(np.flatnonzero(column)[-1])       # the last piece is the top one in its column
        player = int(column[row])
        board = self.board
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < ROWS and 0 <= c < COLS and board[r, c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 4:
                return True
        return False
    
    def _check_win(self, player: int) -> bool:
        """Check if the given player has 4 in a row."""
        board = self.board
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                if (board[r, c] == player and board[r, c+1] == player and
                    board[r, c+2] == player and board[r, c+3] == player):
                    return True
        # Vertical
        for r in range(ROWS - 3):
            for c in range(COLS):
                if (board[r, c] == player and board[r+1, c] == player and
                    board[r+2, c] == player and board[r+3, c] == player):
                    return True
        # Diagonal (up-right)
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if (board[r, c] == player and board[r+1, c+1] == player and
                    board[r+2, c+2] == player and board[r+3, c+3] == player):
                    return True
        # Diagonal (down-right)
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                if (board[r, c] == player and board[r-1, c+1] == player and
                    board[r-2, c+2] == player and board[r-3, c+3] == player):
                    return True
        return False
    
    def encode(self) -> np.ndarray:
        """Encode as (3, 6, 7) float32 tensor."""
        planes = np.zeros((INPUT_PLANES, ROWS, COLS), dtype=np.float32)
        
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board[r, c]
                if piece == self._side:
                    planes[0, r, c] = 1.0
                elif piece == -self._side:
                    planes[1, r, c] = 1.0
        
        planes[2, :, :] = 1.0  # All-ones reference plane
        return planes
    
    def legal_mask(self) -> np.ndarray:
        """Return boolean mask of shape (7,)."""
        mask = np.zeros(ACTION_SPACE, dtype=bool)
        for col in self.legal_moves():
            mask[col] = True
        return mask
    
    def key(self) -> Hashable:
        """Transposition key."""
        if self._hash is None:
            self._hash = (self.board.tobytes(), self._side)
        return self._hash
    
    def orient_policy(self, policy: np.ndarray) -> np.ndarray:
        """Canonical policy frame == absolute frame (encode() never flips columns)."""
        return policy

    def mirror(self) -> tuple['Connect4State', np.ndarray]:
        """Left-right mirror (Connect-4 is symmetric)."""
        mirrored_board = np.flip(self.board, axis=1).copy()
        mirrored_state = Connect4State(
            board=mirrored_board,
            side_to_move=self._side,
            ply_count=self._ply,
            last_action=COLS - 1 - self._last_action if self._last_action >= 0 else -1,
        )
        # Action permutation: column c -> COLS - 1 - c
        perm = np.array([COLS - 1 - c for c in range(COLS)], dtype=np.int32)
        return mirrored_state, perm
    
    def display(self) -> str:
        """Human-readable board."""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        lines = []
        for r in range(ROWS - 1, -1, -1):
            row = ' '.join(symbols[int(self.board[r, c])] for c in range(COLS))
            lines.append(row)
        lines.append('0 1 2 3 4 5 6')
        side_str = "X" if self._side == PLAYER_1 else "O"
        lines.append(f"Side to move: {side_str}, Ply: {self._ply}")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return self.display()
