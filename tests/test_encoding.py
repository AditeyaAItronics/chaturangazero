"""
Action codec, legal mask, encoding and state-immutability property tests (§2.3, §3.1, §3.2).
"""

import numpy as np
import pytest

from tests.conftest import N_POSITIONS
from chaturanga.board import State, WHITE, BLACK, from_fen, get_move_cap
from chaturanga.encoding import encode_state, CANONICAL_PERM, PLY_NORMALIZER
from chaturanga.moves import encode_action, decode_action, legal_mask, generate_moves


def _positions(n, seed=0):
    rng = np.random.default_rng(seed)
    out, s = [], State()
    while len(out) < n:
        if s.is_terminal()[0]:
            s = State()
        out.append(s)
        moves = s.legal_moves()
        s = s.apply(moves[int(rng.integers(len(moves)))])
    return out


POSITIONS = _positions(N_POSITIONS)


def test_decode_encode_roundtrip_for_every_legal_move_in_10k_positions():
    count = 0
    for s in POSITIONS:
        for m in s.legal_moves():
            assert 0 <= m < 4096
            assert encode_action(*decode_action(m)) == m
            count += 1
    assert count > N_POSITIONS


def test_legal_mask_has_exactly_len_legal_moves_bits_in_10k_positions():
    for s in POSITIONS:
        mask = legal_mask(s)
        moves = s.legal_moves()
        assert mask.dtype == bool and mask.shape == (4096,)
        assert int(mask.sum()) == len(moves) == len(set(moves))
        assert set(np.flatnonzero(mask)) == set(moves)


def test_apply_never_mutates_the_original_state():
    """The 'apply then undo restores the hash' property: apply() returns a new
    State, so the original's key must be unchanged after every move."""
    for s in POSITIONS[:500]:
        key, board = s.key(), s.board.copy()
        for m in s.legal_moves():
            s.apply(m)
        assert s.key() == key and np.array_equal(s.board, board)


def test_encoding_planes():
    for s in POSITIONS[:2000]:
        x = encode_state(s)
        assert x.shape == (14, 8, 8) and x.dtype == np.float32
        assert set(np.unique(x[:12])) <= {0.0, 1.0}
        # one piece plane per occupied square, none on empty squares
        assert int(x[:12].sum()) == int(np.count_nonzero(s.board))
        assert np.all(x[:12].sum(axis=0) <= 1)
        # our pieces in 0-5, theirs in 6-11
        assert int(x[:6].sum()) == s.piece_count(s.side_to_move)
        assert int(x[6:12].sum()) == s.piece_count(-s.side_to_move)
        assert np.all(x[12] == pytest.approx(s.ply_count / PLY_NORMALIZER))
        assert np.all(x[13] == 1.0)


def test_canonical_flip_puts_our_pawns_moving_up():
    """Black to move: Black's Padatis appear in plane 0 on the rank-1 side (flipped)."""
    s = State().apply(encode_action(8, 16))        # 1. a2-a3, Black to move
    x = encode_state(s)
    assert s.side_to_move == BLACK
    assert x[0, 1].sum() == 8          # Black's 8 pawns, canonical rank 2
    assert x[5, 0, 4] == 1.0           # Black's Raja on canonical e1
    # White's pawns, seen from Black's side: real rank 2 -> canonical row 6, a3 -> row 5
    assert x[6, 6].sum() == 7 and x[6, 5, 0] == 1.0


def test_canonical_action_perm_maps_legal_masks_between_colour_flipped_twins():
    for s in POSITIONS[:2000]:
        twin_board = np.zeros(64, dtype=np.int8)
        for sq_ in range(64):
            twin_board[sq_ ^ 56] = -s.board[sq_]
        twin = State(board=twin_board, side_to_move=-s.side_to_move, ply_count=s.ply_count)
        if s.side_to_move == BLACK:
            assert np.array_equal(s.orient_policy(legal_mask(s)), legal_mask(twin))
            assert np.array_equal(encode_state(s), encode_state(twin))
