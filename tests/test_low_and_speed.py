"""
Regression tests for the low-priority issues (L1–L8) and the speed/memory
optimisations (S1–S7) in needtofix.md.
"""

import ast
import os
import sys
import tracemalloc

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from chaturanga.board import State, from_fen
from chaturanga.moves import generate_moves
from chaturanga.terminal import terminal_reason
from connect4.game import Connect4State
from net.model import create_chaturanga_net, create_connect4_net
from net.train import Trainer, create_evaluator
from search.arena import arena
from search.mcts import search
from pipeline.inference_server import run_batched_arena


def _random_states(n, seed=0):
    rng = np.random.default_rng(seed)
    out, s = [], State()
    while len(out) < n:
        if s.is_terminal()[0]:
            s = State()
        out.append(s)
        s = s.apply(int(rng.choice(s.legal_moves())))
    return out


# --- L1: requirements --------------------------------------------------------

def test_l1_requirements_list_flask_and_pytest():
    req = open(os.path.join(ROOT, 'requirements.txt'), encoding='utf-8').read().lower()
    dev = open(os.path.join(ROOT, 'requirements-dev.txt'), encoding='utf-8').read().lower()
    assert 'flask' in req
    assert 'pytest' in dev and '-r requirements.txt' in dev


# --- L3: optimiser ------------------------------------------------------------

def _batch(n=8):
    rng = np.random.default_rng(0)
    states = np.stack([s.encode() for s in _random_states(n)])
    pis = rng.dirichlet(np.ones(4096), size=n).astype(np.float32)
    return states, pis, rng.uniform(-1, 1, (n, 1)).astype(np.float32)


def test_l3_adamw_decays_only_conv_and_linear_weights():
    net = create_chaturanga_net(1, 8)
    trainer = Trainer(net, device='cpu', optimizer='adamw', weight_decay=0.05)
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    decay_group, no_decay_group = trainer.optimizer.param_groups
    assert decay_group['weight_decay'] == 0.05 and no_decay_group['weight_decay'] == 0.0
    assert all(p.ndim > 1 for p in decay_group['params'])        # conv / linear weights
    assert all(p.ndim == 1 for p in no_decay_group['params'])    # BN scales/shifts, biases
    stats = trainer.train_step(*_batch())
    assert np.isfinite(stats['total_loss']) and 'l2_loss' not in stats


def test_l3_adam_uses_the_plans_explicit_l2_term():
    net = create_chaturanga_net(1, 8)
    trainer = Trainer(net, device='cpu', optimizer='adam', l2=1e-3)
    assert trainer.optimizer.param_groups[0]['weight_decay'] == 0.0
    expected = 1e-3 * sum(float((p.detach() ** 2).sum()) for p in net.parameters() if p.ndim > 1)
    stats = trainer.train_step(*_batch())
    assert stats['l2_loss'] == pytest.approx(expected, rel=1e-5)
    assert stats['total_loss'] == pytest.approx(stats['policy_loss'] + stats['value_loss'] + expected, rel=1e-5)


# --- L4: web session lock ----------------------------------------------------

def test_l4_web_routes_hold_the_session_lock():
    import web.app as app
    for name in ('get_state', 'new_game', 'make_move', 'ai_move', 'undo_move'):
        assert hasattr(getattr(app, name), '__wrapped__'), f"{name} is not wrapped by the session lock"
    client = app.app.test_client()
    assert client.post('/api/new_game', json={'game': 'chaturanga', 'p1_type': 'human',
                                              'p2_type': 'random'}).status_code == 200
    assert client.get('/api/state').status_code == 200


# --- L5: policy-head index layout (as documented in net/model.py) -------------

def test_l5_policy_logit_index_is_channel_times_64_plus_square():
    net = create_chaturanga_net(1, 8).eval()
    captured = {}
    net.policy_bn.register_forward_hook(lambda m, i, o: captured.setdefault('planes', o.detach()))
    with torch.no_grad():
        logits, _ = net(torch.randn(1, 14, 8, 8))
    planes = captured['planes'][0]               # (64 channels, 8, 8)
    c, r, f = 37, 2, 5
    assert logits[0, c * 64 + (r * 8 + f)] == planes[c, r, f]   # channel = from, (r, f) = to


# --- L7: reference engine independence ---------------------------------------

def test_l7_reference_engine_has_no_module_level_production_imports():
    import chaturanga.reference as ref
    tree = ast.parse(open(ref.__file__, encoding='utf-8').read())
    top_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert top_imports == [], [ast.dump(n) for n in top_imports]


def test_l7_reference_moves_match_production():
    from chaturanga.reference import ref_generate_moves
    for s in _random_states(300, seed=4):
        assert ref_generate_moves(s) == sorted(generate_moves(s))


# --- L8: King-vs-King is unreachable from play ------------------------------

def test_l8_a_move_never_reduces_the_movers_own_pieces():
    """The invariant behind L8: you can't lose material on your own move."""
    for s in _random_states(400, seed=8):
        side = s.side_to_move
        mine = s.piece_count(side)
        for a in s.legal_moves():
            assert s.apply(a).piece_count(side) == mine


def test_l8_random_games_never_end_king_vs_king():
    rng = np.random.default_rng(1)
    reasons = {}
    for _ in range(150):
        s = State()
        while not s.is_terminal()[0]:
            s = s.apply(int(rng.choice(s.legal_moves())))
        r = terminal_reason(s)
        reasons[r] = reasons.get(r, 0) + 1
    assert 'king_vs_king' not in reasons, reasons


# --- S1: node size -----------------------------------------------------------

def test_s1_nodes_are_sized_to_legal_moves():
    def uniform(state):
        return np.full(4096, 1 / 4096, dtype=np.float32), 0.0
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    _, root = search(State(), uniform, n_sims=300, add_noise=False, return_root=True)
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    def count(n):
        return 1 + sum(count(c) for c in n.children.values())
    per_node = sum(s.size_diff for s in after.compare_to(before, 'filename')) / count(root)
    assert per_node < 4096, f"{per_node:.0f} bytes per node (was ~49,000 with dense 4096 arrays)"
    assert len(root.N) == len(root.legal_actions) == len(State().legal_moves())
    assert root.visit_distribution().shape == (4096,)


# --- S3: cached legal moves and terminal result -------------------------------

def test_s3_legal_moves_are_cached_and_safe_to_modify():
    s = State()
    first = s.legal_moves()
    first.append(-1)                      # callers may modify the returned list
    assert s.legal_moves() == first[:-1]
    s2 = State()
    s2.is_terminal()                      # the stalemate check generates the moves...
    assert s2._moves is not None          # ...and they are reused by legal_moves()


def test_s3_terminal_cache_respects_rule_settings():
    from chaturanga.board import configure_rules, get_move_cap
    cap = get_move_cap()
    s = from_fen("4k3/7p/8/8/8/8/8/R3K3 w 150")
    try:
        assert s.is_terminal() == (False, 0.0)
        configure_rules(move_cap=150)
        assert s.is_terminal()[0]          # same State object, new cap: re-evaluated
    finally:
        configure_rules(move_cap=cap)
    assert s.is_terminal() == (False, 0.0)


# --- S6: batched arena reproduces the sequential arena -----------------------

def test_s6_batched_arena_matches_sequential_arena():
    torch.manual_seed(1); a = create_connect4_net(1, 8).eval()
    torch.manual_seed(2); b = create_connect4_net(1, 8).eval()
    ea, eb = create_evaluator(a, 'cpu', 7), create_evaluator(b, 'cpu', 7)

    def pa(s):
        return int(np.argmax(search(s, ea, n_sims=16, action_size=7, add_noise=False)))

    def pb(s):
        return int(np.argmax(search(s, eb, n_sims=16, action_size=7, add_noise=False)))

    np.random.seed(9)
    sequential = arena(pa, pb, Connect4State, n_games=10, action_size=7, opening_plies=2)
    np.random.seed(9)
    batched = run_batched_arena(a, b, Connect4State, n_games=10, n_sims=16, action_size=7,
                                device='cpu', amp=False, opening_plies=2)
    assert batched['results'] == sequential['results']
    assert batched['p1_win_rate'] == sequential['p1_win_rate']
