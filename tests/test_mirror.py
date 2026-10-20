"""
Mirror augmentation (§3.5): left-right mirroring is a valid transformation of the
whole game tree, and MIRROR_PERM is the matching action permutation.

The key check from §2.3: perft(mirror(s), d) == perft(s, d) for every test
position — this validates movegen symmetry and the permutation at once.
"""

import json
import os

import numpy as np
import pytest

from chaturanga.board import State, from_fen
from chaturanga.encoding import MIRROR_PERM, encode_state, mirror_state
from chaturanga.moves import legal_mask
from chaturanga.perft import perft

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, 'tests', 'perft_reference.json'), encoding='utf-8') as f:
    REFERENCE = json.load(f)['positions']
LEAF_LIMIT = 200_000


def test_mirror_perm_is_an_involution():
    assert np.array_equal(MIRROR_PERM[MIRROR_PERM], np.arange(4096))


@pytest.mark.parametrize('pos', REFERENCE, ids=[p['name'] for p in REFERENCE])
def test_perft_is_mirror_invariant(pos):
    s = from_fen(pos['fen'])
    m = mirror_state(s)
    for depth, expected in sorted(pos['expected'].items(), key=lambda kv: int(kv[0])):
        if expected > LEAF_LIMIT:
            continue
        assert perft(m, int(depth)) == expected == perft(s, int(depth))


def _positions(n, seed=5):
    rng = np.random.default_rng(seed)
    out, s = [], State()
    while len(out) < n:
        if s.is_terminal()[0]:
            s = State()
        out.append(s)
        s = s.apply(int(rng.choice(s.legal_moves())))
    return out


def test_mirror_maps_legal_moves_and_encodings():
    for s in _positions(2000):
        m = mirror_state(s)
        # The mirrored position's legal moves are exactly the permuted ones
        assert np.array_equal(legal_mask(m), legal_mask(s)[MIRROR_PERM])
        # Encoding of the mirror = file-flipped encoding (what the buffer does)
        assert np.array_equal(encode_state(m), encode_state(s)[..., ::-1])
        # Mirroring preserves the game result
        assert m.is_terminal() == s.is_terminal()


def test_mirror_commutes_with_canonical_orientation():
    """Buffer mirroring happens in the canonical frame; both flips must commute."""
    for s in _positions(500, seed=6):
        pi = np.random.default_rng(0).random(4096).astype(np.float32)
        a = s.orient_policy(pi)[MIRROR_PERM]
        b = mirror_state(s).orient_policy(pi[MIRROR_PERM])
        assert np.array_equal(a, b)
