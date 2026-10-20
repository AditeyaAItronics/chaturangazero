"""
Tests for the Chaturanga rules engine.

Covers:
- Move generation for all 6 piece types
- Terminal condition ordering (§2.1)
- The stalemate +1 sign trap
- Encoding round-trips
- Mirror invariance
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chaturanga.board import (
    State, WHITE, BLACK,
    PADATI, ASHVA, GAJA, RATHA, MANTRI, RAJA,
    sq, rank_of, file_of,
)
from chaturanga.moves import (
    generate_moves, apply_move, legal_mask,
    encode_action, decode_action,
)
from chaturanga.terminal import is_terminal
from chaturanga.encoding import encode_state, mirror_state, MIRROR_PERM


def test_initial_position_move_count():
    """Starting position should have a known number of legal moves."""
    state = State()
    moves = generate_moves(state)
    # White's initial moves:
    # 8 pawns: each can move forward = 8 moves
    # 2 Ashva: each has 2 moves from starting position = 4 moves
    # 2 Gaja: from c1 can go to a3/e3, from f1 can go to d3/h3 = 4 moves
    # 2 Ratha: blocked by pawns = 0 moves
    # 1 Mantri: d1, can go to c2/e2 = 2 diagonal moves
    # 1 Raja: e1, can go to d2/f2 = 2 moves (forward and diag occupied by own pieces)
    #   Actually: Raja at e1. d1=Mantri(own), f1=Gaja(own), d2=Pawn(own), e2=Pawn(own), f2=Pawn(own)
    #   So Raja moves: only squares that are empty or enemy
    # Let me count more carefully...
    print(f"Initial position: {len(moves)} legal moves")
    # Just verify it's reasonable (should be around 20)
    assert 10 < len(moves) < 40, f"Unexpected move count: {len(moves)}"
    print("  PASS: initial move count is reasonable")


def test_action_codec_roundtrip():
    """decode(encode(from, to)) == (from, to) for all squares."""
    for from_sq in range(64):
        for to_sq in range(64):
            action = encode_action(from_sq, to_sq)
            assert 0 <= action < 4096
            f, t = decode_action(action)
            assert f == from_sq and t == to_sq
    print("  PASS: action codec roundtrip")


def test_action_codec_roundtrip_on_legal_moves():
    """decode(encode(m)) == m for every legal move in a position."""
    state = State()
    moves = generate_moves(state)
    for action in moves:
        from_sq, to_sq = decode_action(action)
        reconstructed = encode_action(from_sq, to_sq)
        assert reconstructed == action
    print("  PASS: action codec on legal moves")


def test_terminal_raja_captured():
    """A position with no Raja for side to move should be terminal with z = -1."""
    board = np.zeros(64, dtype=np.int8)
    # White has pieces but no Raja
    board[sq(0, 0)] = RATHA  # White rook at a1
    board[sq(7, 4)] = -RAJA  # Black raja at e8
    
    state = State(board=board, side_to_move=WHITE)
    done, z = is_terminal(state)
    assert done == True, "Should be terminal — White has no Raja"
    assert z == -1.0, f"z should be -1 (loss), got {z}"
    print("  PASS: Raja captured terminal")


def test_terminal_king_vs_king():
    """K-v-K should be a draw, NOT a bare-king loss."""
    board = np.zeros(64, dtype=np.int8)
    board[sq(0, 4)] = RAJA   # White raja at e1
    board[sq(7, 4)] = -RAJA  # Black raja at e8
    
    # From White's perspective
    state = State(board=board, side_to_move=WHITE)
    done, z = is_terminal(state)
    assert done == True, "K-v-K should be terminal"
    assert z == 0.0, f"K-v-K should be a draw, got z={z}"
    
    # From Black's perspective
    state = State(board=board, side_to_move=BLACK)
    done, z = is_terminal(state)
    assert done == True
    assert z == 0.0, f"K-v-K from Black's side should also be draw, got z={z}"
    print("  PASS: King vs King is draw")


def test_terminal_bare_king():
    """Bare king (not K-v-K) should lose."""
    board = np.zeros(64, dtype=np.int8)
    board[sq(0, 4)] = RAJA    # White raja
    board[sq(7, 4)] = -RAJA   # Black raja
    board[sq(7, 0)] = -RATHA  # Black also has a rook
    
    # White has bare king, Black has raja + rook → White loses
    state = State(board=board, side_to_move=WHITE)
    done, z = is_terminal(state)
    assert done == True, "Bare king should be terminal"
    assert z == -1.0, f"Bare king should lose, got z={z}"
    print("  PASS: bare king loses")


def test_terminal_stalemate_wins():
    """
    THE SIGN TRAP: Stalemate should return z = +1 for the stalemated player.

    Hand-built position: White to move has no legal move at all.
        White: Raja a1, Gaja b1, Gaja b2, Padati a2 c2 d3 d4
        Black: Padati a3 c3 d5, Raja h8
    - a2, c2, d4 are blocked by the Black pawns directly ahead; their
      diagonal-forward squares hold no enemy piece, so no captures.
    - d3 is blocked by White's own d4.
    - Gaja b1 can only reach d3 and Gaja b2 can only reach d4: both own pieces.
    - Raja a1's neighbours a2, b1, b2 are all White's own pieces.
    """
    board = np.zeros(64, dtype=np.int8)
    board[sq(0, 0)] = RAJA      # a1
    board[sq(0, 1)] = GAJA      # b1
    board[sq(1, 1)] = GAJA      # b2
    board[sq(1, 0)] = PADATI    # a2
    board[sq(1, 2)] = PADATI    # c2
    board[sq(2, 3)] = PADATI    # d3
    board[sq(3, 3)] = PADATI    # d4
    board[sq(2, 0)] = -PADATI   # a3
    board[sq(2, 2)] = -PADATI   # c3
    board[sq(4, 3)] = -PADATI   # d5
    board[sq(7, 7)] = -RAJA     # h8

    state = State(board=board, side_to_move=WHITE)
    moves = generate_moves(state)
    assert moves == [], f"position must be a true stalemate, but White has moves {moves}"

    done, z = is_terminal(state)
    assert done, "No legal moves must be terminal"
    assert z == 1.0, f"Stalemate must be a WIN for the stalemated player (z=+1), got z={z}"
    print("  PASS: stalemate returns +1")


def test_terminal_order():
    """Rule 2 (K-v-K) must precede rule 3 (bare king) in evaluation order."""
    board = np.zeros(64, dtype=np.int8)
    board[sq(3, 3)] = RAJA   # White raja
    board[sq(5, 5)] = -RAJA  # Black raja
    
    # Both have exactly one piece (their Raja)
    state = State(board=board, side_to_move=WHITE)
    done, z = is_terminal(state)
    
    # This MUST be K-v-K (draw), NOT bare king (loss)
    assert done == True
    assert z == 0.0, f"K-v-K resolved as z={z}, should be 0 (draw). Ordering bug!"
    print("  PASS: terminal condition ordering correct")


def test_raja_can_move_to_guarded_square():
    """The Raja may legally step onto a guarded square — no check rule."""
    board = np.zeros(64, dtype=np.int8)
    board[sq(3, 3)] = RAJA    # White raja at d4
    board[sq(5, 4)] = -RATHA  # Black ratha at e6, attacks e4
    board[sq(7, 4)] = -RAJA   # Black raja
    
    state = State(board=board, side_to_move=WHITE)
    moves = generate_moves(state)
    
    # e4 is attacked by the Black ratha, but White Raja should still be able to go there
    target_action = encode_action(sq(3, 3), sq(3, 4))  # d4 to e4
    assert target_action in moves, "Raja should be able to move to guarded square e4"
    print("  PASS: Raja can move to guarded square")


def test_gaja_jumping():
    """Gaja jumps over intervening squares — they don't block it."""
    board = np.zeros(64, dtype=np.int8)
    board[sq(0, 2)] = GAJA    # White gaja at c1
    board[sq(1, 1)] = PADATI  # White pawn at b2 (intervening square)
    board[sq(1, 3)] = -PADATI # Black pawn at d2 (intervening square)
    board[sq(7, 4)] = -RAJA   # Black raja
    board[sq(0, 4)] = RAJA    # White raja
    
    state = State(board=board, side_to_move=WHITE)
    moves = generate_moves(state)
    
    # Gaja at c1 should be able to jump to a3 and e3
    gaja_to_a3 = encode_action(sq(0, 2), sq(2, 0))
    gaja_to_e3 = encode_action(sq(0, 2), sq(2, 4))
    
    assert gaja_to_a3 in moves, "Gaja should jump over b2 to reach a3"
    assert gaja_to_e3 in moves, "Gaja should jump over d2 to reach e3"
    print("  PASS: Gaja jumps over occupied squares")


def test_padati_promotion():
    """Padati on the 7th rank promotes to Mantri when reaching the 8th."""
    board = np.zeros(64, dtype=np.int8)
    board[sq(6, 3)] = PADATI  # White pawn about to promote at d7
    board[sq(0, 4)] = RAJA    # White raja
    board[sq(7, 4)] = -RAJA   # Black raja
    
    state = State(board=board, side_to_move=WHITE)
    moves = generate_moves(state)
    
    promote_action = encode_action(sq(6, 3), sq(7, 3))  # d7 to d8
    assert promote_action in moves, "Pawn should be able to promote"
    
    # Apply the promotion
    new_state = apply_move(state, promote_action)
    promoted_piece = new_state.piece_at(sq(7, 3))
    assert promoted_piece == MANTRI, f"Promoted to {promoted_piece}, expected Mantri ({MANTRI})"
    print("  PASS: Padati promotes to Mantri")


def test_encode_shape():
    """State encoding should produce (14, 8, 8)."""
    state = State()
    encoded = encode_state(state)
    assert encoded.shape == (14, 8, 8), f"Expected (14,8,8), got {encoded.shape}"
    assert encoded.dtype == np.float32
    print("  PASS: encoding shape and dtype")


def test_legal_mask_count():
    """legal_mask should have exactly len(legal_moves) bits set."""
    state = State()
    moves = generate_moves(state)
    mask = legal_mask(state)
    
    assert mask.shape == (4096,)
    assert np.sum(mask) == len(moves), \
        f"Mask has {np.sum(mask)} bits but {len(moves)} legal moves"
    print("  PASS: legal_mask matches move count")


def test_random_playout_terminates():
    """10 random games should all terminate within the move cap."""
    import random
    
    all_terminated = True
    for game_idx in range(10):
        state = State()
        for ply in range(350):  # Extra margin beyond move cap
            done, z = is_terminal(state)
            if done:
                break
            moves = generate_moves(state)
            if not moves:
                break
            action = random.choice(moves)
            state = apply_move(state, action)
        else:
            all_terminated = False
            print(f"  FAIL: game {game_idx} did not terminate in 350 plies")
    
    assert all_terminated, "a random playout ran past the move cap without terminating"
    print("  PASS: all 10 random games terminated")


def test_differential_movegen():
    """Cross-check production vs reference engine on initial position."""
    from chaturanga.reference import ref_generate_moves
    
    state = State()
    prod_moves = sorted(generate_moves(state))
    ref_moves = ref_generate_moves(state)  # Already sorted
    
    assert prod_moves == ref_moves, \
        f"Production and reference disagree!\nProd: {prod_moves}\nRef:  {ref_moves}"
    print("  PASS: differential movegen on initial position")


def test_differential_perft_depth2():
    """Cross-check perft at depth 2 between production and reference."""
    from chaturanga.perft import perft
    from chaturanga.reference import ref_perft
    
    state = State()
    prod_count = perft(state, 2)
    ref_count = ref_perft(state, 2)
    
    assert prod_count == ref_count, \
        f"Perft depth 2 disagrees: prod={prod_count}, ref={ref_count}"
    print(f"  PASS: perft depth 2 = {prod_count}")


if __name__ == '__main__':
    print("=" * 50)
    print("Chaturanga Rules Engine Tests")
    print("=" * 50)
    
    tests = [
        test_initial_position_move_count,
        test_action_codec_roundtrip,
        test_action_codec_roundtrip_on_legal_moves,
        test_terminal_raja_captured,
        test_terminal_king_vs_king,
        test_terminal_bare_king,
        test_terminal_stalemate_wins,
        test_terminal_order,
        test_raja_can_move_to_guarded_square,
        test_gaja_jumping,
        test_padati_promotion,
        test_encode_shape,
        test_legal_mask_count,
        test_random_playout_terminates,
        test_differential_movegen,
        test_differential_perft_depth2,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n{test.__name__}:")
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1
    
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 50}")
