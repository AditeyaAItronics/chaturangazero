"""
Perft regression suite against the frozen numbers in tests/perft_reference.json
(§2.3, needtofix M9). Regenerate the file with tools/perft_freeze.py.

By default every position is checked with BOTH engines up to the depths that
stay under ~200k leaves (a few seconds in total). Set CHATURANGA_FULL_PERFT=1
to check every frozen depth (up to 6 — this takes a long time).
"""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from chaturanga.board import from_fen, to_fen
from chaturanga.perft import perft
from chaturanga.reference import ref_perft, ref_is_terminal

with open(os.path.join(ROOT, 'tests', 'perft_reference.json'), encoding='utf-8') as f:
    REFERENCE = json.load(f)['positions']

FULL = os.environ.get('CHATURANGA_FULL_PERFT') == '1'
QUICK_LEAF_LIMIT = 200_000


def test_fifteen_positions_with_numbers():
    assert len(REFERENCE) == 15
    for p in REFERENCE:
        assert p['expected'], f"{p['name']} has no frozen perft numbers"
        assert '4' in p['expected'], f"{p['name']} is not frozen to at least depth 4"


@pytest.mark.parametrize('pos', REFERENCE, ids=[p['name'] for p in REFERENCE])
def test_root_terminal_status(pos):
    state = from_fen(pos['fen'])
    assert to_fen(state) == pos['fen']
    expected = (pos['root_terminal']['done'], pos['root_terminal']['z'])
    assert state.is_terminal() == expected
    assert ref_is_terminal(state) == expected


@pytest.mark.parametrize('pos', REFERENCE, ids=[p['name'] for p in REFERENCE])
def test_perft_matches_frozen_numbers(pos):
    state = from_fen(pos['fen'])
    for depth, expected in sorted(pos['expected'].items(), key=lambda kv: int(kv[0])):
        if not FULL and expected > QUICK_LEAF_LIMIT:
            continue
        d = int(depth)
        assert perft(state, d) == expected, f"production perft({pos['name']}, {d})"
        assert ref_perft(state, d) == expected, f"reference perft({pos['name']}, {d})"


def test_initial_position_numbers_unchanged():
    """The original hand-entered start-position values are still reproduced."""
    start = next(p for p in REFERENCE if p['name'] == 'initial_position')
    for depth, n in {'1': 16, '2': 256, '3': 4176, '4': 68133}.items():
        assert start['expected'][depth] == n
