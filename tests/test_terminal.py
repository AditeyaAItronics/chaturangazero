"""
Terminal ladder (§2.1) and termination (§2.2, §2.3) tests.
"""

import numpy as np
import pytest

from tests.conftest import N_PLAYOUTS
from chaturanga.board import (State, from_fen, get_move_cap, configure_rules,
                              get_adjudicate_margin)
from chaturanga.terminal import is_terminal, terminal_reason, evaluate_terminal
from chaturanga.reference import ref_is_terminal

# One hand-built position per rung of the ladder: (FEN, expected (done, z), reason)
LADDER = [
    ("4k3/8/8/8/8/8/8/R7 w 0", (True, -1.0), 'raja_captured'),       # White has no Raja
    ("8/8/5k2/8/3K4/8/8/8 w 0", (True, 0.0), 'king_vs_king'),
    ("8/8/5k2/8/3K4/8/8/8 b 0", (True, 0.0), 'king_vs_king'),
    ("r3k3/8/8/8/8/8/8/4K3 w 0", (True, -1.0), 'bare_king'),         # White bare, Black has a Ratha
    ("7k/8/8/3p4/3P4/p1pP4/PBP5/KB6 w 0", (True, 1.0), 'stalemate'), # the +1 sign trap
    ("4k3/7p/8/8/8/8/8/R3K3 w 0", (False, 0.0), 'ongoing'),
]


@pytest.fixture(autouse=True)
def restore_rules():
    cap, margin = get_move_cap(), get_adjudicate_margin()
    yield
    configure_rules(move_cap=cap, adjudicate_margin=margin)


@pytest.mark.parametrize('fen,expected,reason', LADDER)
def test_each_rung_of_the_ladder(fen, expected, reason):
    s = from_fen(fen)
    assert is_terminal(s) == expected
    assert terminal_reason(s) == reason
    assert ref_is_terminal(s) == expected          # the independent engine agrees


def test_rule_order_raja_capture_beats_everything():
    # White has no Raja AND no moves AND is at the cap: rule 1 decides
    s = from_fen(f"4k3/8/8/8/8/8/8/8 w {get_move_cap()}")
    assert evaluate_terminal(s) == (True, -1.0, 'raja_captured')


def test_rule_order_king_vs_king_before_bare_king():
    s = from_fen("8/8/5k2/8/3K4/8/8/8 w 0")
    assert evaluate_terminal(s)[2] == 'king_vs_king'


def test_rule_order_stalemate_before_move_cap():
    s = from_fen(f"7k/8/8/3p4/3P4/p1pP4/PBP5/KB6 w {get_move_cap()}")
    assert evaluate_terminal(s) == (True, 1.0, 'stalemate')


@pytest.mark.parametrize('diff_fen,expected', [
    ("4k3/7p/8/8/8/8/8/RN2K3 w", 1.0),     # +8 - 1 = +7  -> White wins
    ("rn2k3/8/8/8/8/8/7P/4K3 w", -1.0),    # -7           -> White loses
    ("4k3/7p/8/8/8/8/8/1N2K3 w", 0.0),     # +3 - 1 = +2  -> draw (< 4)
])
def test_move_cap_adjudication(diff_fen, expected):
    s = from_fen(f"{diff_fen} {get_move_cap()}")
    assert is_terminal(s) == (True, expected)
    assert is_terminal(s, adjudicate=False) == (True, 0.0)      # evaluation games: draw
    assert terminal_reason(s) == 'move_cap'


def test_random_playouts_terminate_with_a_sane_reason_distribution():
    """§2.3: every random playout terminates (10,000 in CI) within the cap."""
    rng = np.random.default_rng(2026)
    cap = get_move_cap()
    reasons = {}
    for _ in range(N_PLAYOUTS):
        s = State()
        while not s.is_terminal()[0]:
            assert s.ply_count < cap
            moves = s.legal_moves()
            s = s.apply(moves[int(rng.integers(len(moves)))])
        assert s.ply_count <= cap
        r = terminal_reason(s)
        reasons[r] = reasons.get(r, 0) + 1
    total = sum(reasons.values())
    assert total == N_PLAYOUTS
    # Random play hangs the Raja: capture dominates; King-vs-King is unreachable
    assert reasons.get('raja_captured', 0) / total > 0.8, reasons
    assert 'king_vs_king' not in reasons, reasons
    assert reasons.get('move_cap', 0) / total < 0.15, reasons
