"""
Regression tests for the critical bugs listed in needtofix.md (C1–C7).
"""

import os
import sys
import json
import hashlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chaturanga.board import State, WHITE, BLACK, PADATI, RATHA, RAJA, MANTRI, sq
from chaturanga.encoding import CANONICAL_PERM, MIRROR_PERM
from chaturanga.moves import generate_moves, apply_move
from chaturanga.perft import perft
from chaturanga.reference import ref_perft, ref_apply_move, ref_is_terminal
from chaturanga.terminal import is_terminal
from connect4.game import Connect4State
from net.model import create_chaturanga_net
from net.replay import ReplayBuffer
from search.mcts import search


def _random_position(n_plies: int, seed: int) -> State:
    rng = np.random.default_rng(seed)
    state = State()
    for _ in range(n_plies):
        if state.is_terminal()[0]:
            break
        state = state.apply(int(rng.choice(state.legal_moves())))
    return state


def _color_flip(state: State) -> State:
    """Same position with colours swapped and ranks flipped: Black-to-move -> White-to-move."""
    board = np.zeros(64, dtype=np.int8)
    for s in range(64):
        board[s ^ 56] = -state.board[s]
    return State(board=board, side_to_move=-state.side_to_move, ply_count=state.ply_count)


# --- C1: Black's policy is oriented like its board -------------------------

def test_c1_canonical_perm_is_involution():
    assert np.array_equal(CANONICAL_PERM[CANONICAL_PERM], np.arange(4096))


def test_c1_black_policy_matches_canonical_board():
    """
    For a Black-to-move position, the canonical encoding equals the encoding of
    the colour-flipped White-to-move position — so the oriented legal mask must
    equal that White position's legal mask. Before the fix, Black's targets
    stayed in absolute squares and this failed.
    """
    for seed in range(20):
        state = _random_position(2 * seed + 1, seed)   # odd ply -> Black to move
        if state.side_to_move != BLACK or state.is_terminal()[0]:
            continue
        twin = _color_flip(state)
        assert np.array_equal(state.encode(), twin.encode())
        assert np.array_equal(state.orient_policy(state.legal_mask()), twin.legal_mask())


def test_c1_white_policy_unchanged():
    state = State()
    pi = np.random.rand(4096).astype(np.float32)
    assert np.array_equal(state.orient_policy(pi), pi)


# --- C2: the first visit from a node follows the prior ---------------------

def test_c2_first_simulation_follows_prior():
    state = Connect4State()
    prior = np.zeros(7, dtype=np.float32)
    prior[6] = 1.0   # all prior mass on the LAST legal action

    def evaluate(s):
        return prior, 0.0

    visits = search(state, evaluate, n_sims=1, action_size=7, add_noise=False)
    assert visits[6] == 1, f"first visit went to {int(np.argmax(visits))}, not the prior's choice"


# --- C3: policy logits can be negative -------------------------------------

def test_c3_policy_logits_not_clamped():
    torch.manual_seed(0)
    net = create_chaturanga_net(n_blocks=1, channels=16).eval()
    with torch.no_grad():
        logits, _ = net(torch.randn(4, 14, 8, 8))
    assert logits.min().item() < 0, "policy logits are clamped to >= 0 (ReLU on the logits)"


# --- C4 / C5: incumbent self-play, resume keeps best.pt and metrics ---------

def _tiny_connect4_config(iterations: int, arena_every: int) -> dict:
    return {
        'game': 'connect4', 'seed': 1, 'device': 'cpu',
        'net': {'blocks': 1, 'channels': 8, 'action_space': 7, 'input_planes': 3},
        'mcts': {'sims': 4, 'c_puct': 1.5, 'dirichlet_alpha': 1.0, 'dirichlet_eps': 0.25},
        'play': {'temp_moves': 4, 'move_cap': 42},
        'train': {'lr': 1e-3, 'l2': 1e-4, 'batch_size': 16, 'buffer_size': 1000,
                  'steps_per_iter': 2, 'grad_clip': 1.0, 'amp_inference': False,
                  'mirror_augment': True},
        'arena': {'games': 2, 'win_threshold': 0.55, 'every_n_iters': arena_every,
                  'opening_plies': 2},
        'loop': {'games_per_iter': 4, 'iterations': iterations, 'parallel_games': 4},
    }


def _sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_c4_c5_resume_keeps_incumbent_and_metrics(tmp_path):
    from pipeline.loop import training_loop

    out = str(tmp_path / 'run')
    training_loop(_tiny_connect4_config(iterations=2, arena_every=1), output_dir=out)
    best = os.path.join(out, 'best.pt')
    best_hash = _sha(best)

    # Resume for one more iteration with no arena: best.pt must be untouched
    # and metrics.json must keep the two earlier entries.
    training_loop(_tiny_connect4_config(iterations=3, arena_every=99), output_dir=out,
                  resume_from=os.path.join(out, 'checkpoint_0001.pt'))
    assert _sha(best) == best_hash, "resume overwrote the incumbent best.pt"
    with open(os.path.join(out, 'metrics.json')) as f:
        iterations = [m['iteration'] for m in json.load(f)]
    assert iterations == [0, 1, 2], f"metrics history lost on resume: {iterations}"


# --- C6: reference engine is independent and agrees -------------------------

def _rules_positions() -> list[State]:
    positions = [State()]
    # Promotion available for both colours
    b = np.zeros(64, dtype=np.int8)
    b[sq(6, 3)] = PADATI; b[sq(1, 5)] = -PADATI
    b[sq(0, 4)] = RAJA; b[sq(7, 4)] = -RAJA; b[sq(3, 0)] = RATHA; b[sq(4, 7)] = -RATHA
    positions += [State(board=b, side_to_move=WHITE), State(board=b, side_to_move=BLACK)]
    # Raja capture available, plus a capture that leaves the opponent bare
    b = np.zeros(64, dtype=np.int8)
    b[sq(0, 0)] = RAJA; b[sq(0, 7)] = RATHA; b[sq(7, 7)] = -RAJA; b[sq(7, 0)] = -MANTRI
    b[sq(5, 7)] = -PADATI
    positions.append(State(board=b, side_to_move=WHITE))
    # A few random middlegames
    positions += [_random_position(12 + k, 100 + k) for k in range(4)]
    return positions


def test_c6_reference_engine_does_not_import_production_rules():
    import chaturanga.reference as ref
    src = open(ref.__file__, encoding='utf-8').read()
    for forbidden in ('from .moves', 'from .terminal', 'import moves', 'import terminal'):
        assert forbidden not in src, f"reference.py still uses production code: {forbidden!r}"


def test_c6_differential_perft_depth3():
    for i, state in enumerate(_rules_positions()):
        if state.is_terminal()[0]:
            continue
        for depth in (1, 2, 3):
            assert perft(state, depth) == ref_perft(state, depth), \
                f"position {i} depth {depth}: prod={perft(state, depth)} ref={ref_perft(state, depth)}"


def test_c6_reference_apply_and_terminal_agree():
    for state in _rules_positions():
        if state.is_terminal()[0]:
            continue
        for a in generate_moves(state):
            prod, ref = apply_move(state, a), ref_apply_move(state, a)
            assert np.array_equal(prod.board, ref.board)
            assert is_terminal(prod) == ref_is_terminal(ref)


# --- C7: compact replay buffer ----------------------------------------------

def _chaturanga_samples(n: int):
    out, state = [], State()
    rng = np.random.default_rng(0)
    while len(out) < n:
        if state.is_terminal()[0]:
            state = State()
        legal = state.legal_moves()
        pi = np.zeros(4096, dtype=np.float32)
        pi[legal] = rng.dirichlet(np.ones(len(legal)))
        out.append((state.encode(), state.orient_policy(pi), float(rng.uniform(-1, 1))))
        state = state.apply(int(rng.choice(legal)))
    return out


def test_c7_buffer_is_lossless_without_mirroring():
    samples = _chaturanga_samples(30)
    buf = ReplayBuffer(max_size=30)
    for s, p, v in samples:
        buf.add(s, p, v)
    # Reproduce the buffer's index draw (positions can repeat, e.g. the start
    # position after a game restarts, so match by slot rather than by state)
    np.random.seed(0)
    expected_idx = np.random.randint(0, 30, size=30)
    np.random.seed(0)
    states, policies, values = buf.sample(30)
    for k, j in enumerate(expected_idx):
        exp_s, exp_p, exp_v = samples[j]
        assert np.array_equal(states[k], exp_s)
        np.testing.assert_allclose(policies[k], exp_p, rtol=0, atol=1e-7)
        assert abs(values[k][0] - exp_v) < 1e-6


def test_c7_buffer_mirrors_state_and_policy_together():
    s, p, v = _chaturanga_samples(1)[0]
    buf = ReplayBuffer(max_size=4, mirror_perm=MIRROR_PERM)
    buf.add(s, p, v)
    np.random.seed(1)
    states, policies, _ = buf.sample(64)
    mirrored = 0
    for st, pol in zip(states, policies):
        if np.array_equal(st, s):
            np.testing.assert_allclose(pol, p, atol=1e-7)
        else:
            mirrored += 1
            assert np.array_equal(st, s[..., ::-1])
            np.testing.assert_allclose(pol, p[MIRROR_PERM], atol=1e-7)
    assert 0 < mirrored < 64


def test_c7_buffer_ring_save_load(tmp_path):
    samples = _chaturanga_samples(25)
    buf = ReplayBuffer(max_size=10)
    for s, p, v in samples:
        buf.add(s, p, v)
    assert len(buf) == 10
    path = str(tmp_path / 'buf.npz')
    buf.save(path)
    loaded = ReplayBuffer(max_size=10)
    loaded.load(path)
    assert len(loaded) == 10
    # Only the 10 newest positions survive the ring
    newest = {s.tobytes() for s, _, _ in samples[-10:]}
    states, _, _ = loaded.sample(50)
    assert {st.tobytes() for st in states} <= newest


def test_c7_buffer_is_compact():
    samples = _chaturanga_samples(200)
    buf = ReplayBuffer(max_size=200)
    for s, p, v in samples:
        buf.add(s, p, v)
    per_position = (buf._bits.nbytes + buf._consts.nbytes + buf._is_const.nbytes
                    + buf._values.nbytes
                    + sum(a.nbytes + b.nbytes for a, b in zip(buf._pi_idx, buf._pi_val))) / 200
    dense = 14 * 8 * 8 * 4 + 4096 * 4
    assert per_position < dense / 20, f"{per_position:.0f} B/position vs {dense} dense"
