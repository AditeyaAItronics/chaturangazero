"""
Regression tests for the medium-priority issues in needtofix.md (M1–M8).
M9 (perft reference numbers) is covered by tests/test_perft_reference.py.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chaturanga.board import State, from_fen, get_move_cap, configure_rules, get_adjudicate_margin
from chaturanga.encoding import MIRROR_PERM
from connect4.game import Connect4State
from connect4.solver import Connect4Solver
from net.model import create_chaturanga_net, create_connect4_net
from net.replay import ReplayBuffer
from net.train import Trainer, create_evaluator, value_calibration
from pipeline.inference_server import (run_batched_self_play, _descend, _revert_virtual_loss)
from search.baselines import connect4_tactical_player
from search.cache import EvalCache
from search.mcts import search, get_subtree
from search.node import Node

STALEMATE_FEN = "7k/8/8/3p4/3P4/p1pP4/PBP5/KB6 w"
RAJA_CAPTURE_FEN = "8/p3k3/8/8/8/8/4R3/K7 w 0"


@pytest.fixture(autouse=True)
def restore_rules():
    cap, margin = get_move_cap(), get_adjudicate_margin()
    yield
    configure_rules(move_cap=cap, adjudicate_margin=margin)


def _uniform_eval(action_size):
    def evaluate(state):
        return np.ones(action_size, dtype=np.float32) / action_size, 0.0
    return evaluate


# --- M1: sequential self-play really reuses the subtree ----------------------

def test_m1_sequential_selfplay_passes_the_chosen_subtree(monkeypatch):
    import search.mcts as mcts
    from pipeline.selfplay import self_play_game

    reused_visits = []
    real_search = mcts.search

    def spy(*args, **kwargs):
        root = kwargs.get('root_node')
        if root is not None:
            reused_visits.append(root.total_visits)
        return real_search(*args, **kwargs)

    monkeypatch.setattr(mcts, 'search', spy)
    np.random.seed(0)
    self_play_game(Connect4State, _uniform_eval(7), action_size=7, n_sims=16, temp_moves=0)
    assert reused_visits, "no search was given a reused root"
    assert max(reused_visits) > 0, "reused roots carried no visits from the previous search"


def test_m1_search_can_return_its_root():
    visits, root = search(Connect4State(), _uniform_eval(7), n_sims=10, action_size=7,
                          add_noise=False, return_root=True)
    assert np.array_equal(visits, root.visit_distribution())
    child = get_subtree(root, int(np.argmax(visits)))
    assert child is not None and child.parent is None


# --- M2: evaluation cache (transposition table) ------------------------------

def test_m2_cache_does_not_change_search_results():
    calls = {'n': 0}
    rng = np.random.default_rng(0)
    table = {}

    def evaluate(state):     # deterministic but position-dependent
        calls['n'] += 1
        key = state.key()
        if key not in table:
            p = rng.random(7).astype(np.float32)
            table[key] = (p / p.sum(), float(rng.uniform(-1, 1)))
        return table[key]

    state = Connect4State()
    plain = search(state, evaluate, n_sims=60, action_size=7, add_noise=False)
    calls['n'] = 0
    cache = EvalCache()
    cached = search(state, evaluate, n_sims=60, action_size=7, add_noise=False, cache=cache)
    first_calls = calls['n']
    again = search(state, evaluate, n_sims=60, action_size=7, add_noise=False, cache=cache)
    assert np.array_equal(plain, cached) and np.array_equal(plain, again)
    assert calls['n'] == first_calls, "second search with a warm cache called the network"
    assert cache.hits > 0


def test_m2_cache_key_includes_ply():
    a = State()
    b = State(board=a.board, side_to_move=a.side_to_move, ply_count=10)
    assert EvalCache.key(a) != EvalCache.key(b)    # the network input differs (ply plane)


def test_m2_batched_selfplay_reports_cache_hits():
    torch.manual_seed(0); np.random.seed(0)
    stats = run_batched_self_play(create_connect4_net(1, 8), Connect4State, total_games=8,
                                  num_parallel_games=8, n_sims=8, action_size=7, device='cpu',
                                  amp=False)
    assert stats['eval_cache_hit_rate'] > 0      # every game shares the opening positions


# --- M3: virtual loss and worker processes -----------------------------------

def test_m3_virtual_loss_is_fully_reverted():
    root = Node(Connect4State(), action_size=7)
    search(Connect4State(), _uniform_eval(7), n_sims=20, action_size=7, add_noise=False,
           root_node=None)
    _, root = search(Connect4State(), _uniform_eval(7), n_sims=20, action_size=7,
                     add_noise=False, return_root=True)
    before = [(n.N.copy(), n.W.copy()) for n in [root] + list(root.children.values())]
    leaf, path = _descend(root, 1.5, 3.0, 7)
    assert path and root.N[path[0][1]] >= 3      # virtual loss applied
    _revert_virtual_loss(path, 3.0)
    after = [(n.N, n.W) for n in [root] + list(root.children.values())]
    for (n0, w0), (n1, w1) in zip(before, after):
        assert np.array_equal(n0, n1) and np.allclose(w0, w1)


@pytest.mark.parametrize('leaves', [1, 4])
def test_m3_batched_selfplay_with_virtual_loss(leaves):
    torch.manual_seed(0); np.random.seed(0)
    buf = ReplayBuffer(5000)
    stats = run_batched_self_play(create_connect4_net(1, 8), Connect4State, total_games=6,
                                  num_parallel_games=3, n_sims=12, action_size=7, device='cpu',
                                  amp=False, buffer=buf, leaves_per_step=leaves)
    assert stats['games'] == 6
    assert len(buf) == stats['samples_generated'] == sum(stats['game_lengths'])


def test_m3_buffer_merge_matches_direct_adds():
    rng = np.random.default_rng(0)
    rows = []
    s = State()
    for _ in range(12):
        legal = s.legal_moves()
        pi = np.zeros(4096, np.float32); pi[legal] = 1.0 / len(legal)
        rows.append((s.encode(), pi, float(rng.uniform(-1, 1)), legal))
        s = s.apply(int(rng.choice(legal)))
    direct, part_a, part_b = ReplayBuffer(20), ReplayBuffer(20), ReplayBuffer(20)
    for k, (st, pi, v, legal) in enumerate(rows):
        direct.add(st, pi, v, legal=legal)
        (part_a if k < 5 else part_b).add(st, pi, v, legal=legal)
    merged = ReplayBuffer(20)
    merged.merge(part_a)
    merged.merge(part_b)
    np.random.seed(5); x = direct.sample(30, return_masks=True)
    np.random.seed(5); y = merged.sample(30, return_masks=True)
    for a, b in zip(x, y):
        assert np.array_equal(a, b)


def test_m3_parallel_worker_processes(tmp_path):
    from pipeline.workers import run_parallel_self_play
    net = create_connect4_net(1, 8)
    buf = ReplayBuffer(5000)
    stats = run_parallel_self_play(net, game='connect4', total_games=5, workers=2, buffer=buf,
                                   seed=3, device='cpu', num_parallel_games=3, n_sims=4,
                                   action_size=7, amp=False)
    assert stats['games'] == 5 and stats['workers'] == 2
    assert len(buf) == stats['samples_generated'] == sum(stats['game_lengths'])


# --- M4: the end reason follows the terminal ladder ---------------------------

def test_m4_stalemate_at_the_cap_is_classified_as_stalemate():
    from pipeline.selfplay import _classify_terminal
    state = from_fen(f"{STALEMATE_FEN} {get_move_cap()}")
    assert state.is_terminal() == (True, 1.0)
    assert _classify_terminal(state) == 'stalemate'     # used to say 'move_cap'


# --- M5 / M6 / M8: run records, diagnostics, snapshots -----------------------

def _tiny_run(tmp_path, workers=1):
    from pipeline.loop import training_loop
    config = {
        'game': 'connect4', 'seed': 4, 'device': 'cpu',
        'net': {'blocks': 1, 'channels': 8, 'action_space': 7, 'input_planes': 3},
        'mcts': {'sims': 4, 'c_puct': 1.5, 'dirichlet_alpha': 1.0, 'dirichlet_eps': 0.25},
        'play': {'temp_moves': 4, 'move_cap': 42},
        'train': {'lr': 1e-3, 'l2': 1e-4, 'batch_size': 16, 'buffer_size': 1000,
                  'steps_per_iter': 2, 'grad_clip': 1.0, 'amp_inference': False,
                  'mirror_augment': True},
        'arena': {'games': 2, 'win_threshold': 0.55, 'every_n_iters': 99},
        'loop': {'games_per_iter': 4, 'iterations': 1, 'parallel_games': 4, 'workers': workers},
    }
    out = tmp_path / 'run'
    training_loop(config, output_dir=str(out))
    return out


def test_m5_m6_m8_run_records(tmp_path):
    out = _tiny_run(tmp_path)

    # M5: config.yaml and checkpoints record the code revision
    with open(out / 'config.yaml') as f:
        assert 'git_sha' in yaml.safe_load(f)['run_info']
    ckpt = torch.load(out / 'checkpoint_0000.pt', weights_only=False)
    assert 'git_sha' in ckpt

    # M6: diagnostics are in metrics.json
    with open(out / 'metrics.json') as f:
        m = json.load(f)[0]
    for key in ('draw_rate', 'raja_capture_miss_rate', 'eval_cache_hit_rate', 'games_per_hour',
                'value_calibration_error', 'value_sign_accuracy', 'value_mean_abs',
                'buffer_age_histogram'):
        assert key in m, key

    # M8: the buffer is snapshotted every iteration, atomically
    assert (out / 'buffer.npz').exists() and not (out / 'buffer.tmp.npz').exists()


def test_m3_training_loop_with_workers(tmp_path):
    out = _tiny_run(tmp_path, workers=2)
    with open(out / 'metrics.json') as f:
        assert json.load(f)[0]['games_per_hour'] > 0


def test_m6_value_calibration():
    z = np.array([1, -1, 1, -1, 0, 1], dtype=float)
    perfect = value_calibration(z, z)
    assert perfect['value_calibration_error'] == pytest.approx(0.0)
    assert perfect['value_sign_accuracy'] == 1.0
    always_draw = value_calibration(np.zeros(6), z)
    assert always_draw['value_mean_abs'] == 0.0
    assert always_draw['value_calibration_error'] > 0.0


def test_m6_missed_raja_capture_counter():
    torch.manual_seed(0); np.random.seed(0)
    stats = run_batched_self_play(create_chaturanga_net(1, 8), lambda: from_fen(RAJA_CAPTURE_FEN),
                                  total_games=4, num_parallel_games=4, n_sims=2, device='cpu',
                                  amp=False)
    assert stats['raja_capture_chances'] >= 4          # every game starts with a capture available
    assert 0 <= stats['raja_captures_missed'] <= stats['raja_capture_chances']


# --- M7: Phase-0 / Phase-2 gates ---------------------------------------------

def test_m7_connect4_tactical_player_wins_and_blocks():
    # X (to move) has three in a row on the bottom: must win in column 3
    s = Connect4State()
    for a in (0, 0, 1, 1, 2, 6):
        s = s.apply(a)
    for seed in range(5):
        np.random.seed(seed)
        assert connect4_tactical_player(s) == 3
    # O to move must block X's three on the bottom row
    s = Connect4State()
    for a in (0, 6, 1, 6, 2):
        s = s.apply(a)
    for seed in range(5):
        np.random.seed(seed)
        assert connect4_tactical_player(s) == 3


def _late_positions(n=20):
    solver, found, seed = Connect4Solver(), [], 0
    while len(found) < n:
        rng = np.random.default_rng(seed); seed += 1
        s = Connect4State()
        for _ in range(30):
            if s.is_terminal()[0]:
                break
            s = s.apply(int(rng.choice(s.legal_moves())))
        if s.is_terminal()[0]:
            continue
        if len(set(solver.move_values(s).values())) > 1:   # some move is strictly worse
            found.append(s)
    return solver, found


def test_m7_mcts_with_perfect_evaluator_plays_optimally():
    """Part 4, S2 exit criterion: 200 sims + exact values -> optimal play from 20 positions."""
    solver, positions = _late_positions(20)

    def perfect(state):
        return np.ones(7, dtype=np.float32) / 7, float(solver.value(state))

    for i, s in enumerate(positions):
        visits = search(s, perfect, n_sims=200, action_size=7, add_noise=False)
        move = int(np.argmax(visits))
        assert move in solver.best_moves(s), \
            f"position {i}: played {move}, optimal {sorted(solver.best_moves(s))}, values {solver.move_values(s)}"


def test_m7_train_connect4_checks_both_gates():
    from train_connect4 import evaluate_gates, RANDOM_GATE, GREEDY_GATE
    assert (RANDOM_GATE, GREEDY_GATE) == (0.95, 0.80)
    g = evaluate_gates(create_connect4_net(1, 8), 'cpu', n_sims=2, n_games=2)
    assert {'random_rate', 'greedy_rate', 'passed'} <= set(g)
