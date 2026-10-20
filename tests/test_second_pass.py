"""
Regression tests for the second-pass review (needtofixagain.md, N1–N20).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chaturanga.board import (
    State, WHITE, BLACK, PADATI, RATHA, RAJA, sq,
    configure_rules, get_move_cap, get_adjudicate_margin,
)
from search.cache import EvalCache
from search.mcts import search
from search.node import Node


@pytest.fixture(autouse=True)
def restore_rules():
    """configure_rules() is process-global; never leak a changed cap into other tests."""
    cap, margin = get_move_cap(), get_adjudicate_margin()
    yield
    configure_rules(move_cap=cap, adjudicate_margin=margin)


def _uniform(state):
    return np.full(4096, 1.0 / 4096, dtype=np.float32), 0.0


def _materially_won_one_ply_before_cap() -> State:
    """
    White to move at ply cap-1, a Ratha up against a Padati (+4 = the default
    margin), with no capture of anything available to either side. Every
    White move reaches the cap, where adjudication scores a White win but a
    real game (arena, evaluation) scores a draw.
    """
    configure_rules(move_cap=10)
    board = np.zeros(64, dtype=np.int8)
    board[sq(0, 0)] = RAJA          # a1
    board[sq(0, 1)] = RATHA         # b1
    board[sq(7, 7)] = -RAJA         # h8
    board[sq(6, 0)] = -PADATI       # a7
    return State(board=board, side_to_move=WHITE, ply_count=9)


# --- N1: no material adjudication inside non-training search ---------------

def test_n1_search_adjudicates_only_when_asked():
    state = _materially_won_one_ply_before_cap()
    _, train_root = search(state, _uniform, n_sims=50, add_noise=False,
                           return_root=True, adjudicate=True)
    _, arena_root = search(state, _uniform, n_sims=50, add_noise=False,
                           return_root=True, adjudicate=False)
    # Self-play: every capped leaf is a material win for White
    assert np.allclose(train_root.W, train_root.N)
    # Arena/evaluation: a capped game is a draw, so the search must back up 0
    assert arena_root.N.sum() == 50 and np.allclose(arena_root.W, 0.0)


def test_n1_default_is_still_training_mode():
    """Self-play callers don't pass the flag; their behaviour must not change."""
    state = _materially_won_one_ply_before_cap()
    _, root = search(state, _uniform, n_sims=20, add_noise=False, return_root=True)
    assert np.allclose(root.W, root.N)


def test_n1_cache_keeps_modes_apart():
    before_cap = _materially_won_one_ply_before_cap()
    state = before_cap.apply(before_cap.legal_moves()[0])     # at the cap
    cache = EvalCache()
    cache.put_terminal(state, -1.0, adjudicate=True)
    assert cache.get(state, adjudicate=False) is None
    assert cache.get(state, adjudicate=True) == ('terminal', -1.0)
    # One shared cache, both modes: a training search must not poison an arena search
    root_state = _materially_won_one_ply_before_cap()
    search(root_state, _uniform, n_sims=30, add_noise=False, cache=cache, adjudicate=True)
    _, root = search(root_state, _uniform, n_sims=30, add_noise=False, cache=cache,
                     return_root=True, adjudicate=False)
    assert np.allclose(root.W, 0.0)


def test_n1_batched_arena_search_does_not_adjudicate():
    from pipeline.inference_server import _try_expand_without_network, _expand_roots
    state = _materially_won_one_ply_before_cap()
    child = state.apply(state.legal_moves()[0])          # at the cap, Black to move
    assert _try_expand_without_network(Node(child), None, adjudicate=True) == -1.0
    assert _try_expand_without_network(Node(child), None, adjudicate=False) == 0.0
    root = Node(child)
    _expand_roots([root], evaluator=None, cache=None, adjudicate=False)
    assert root.is_terminal and root.terminal_value == 0.0


@pytest.fixture
def adjudicated_cap_checks(monkeypatch):
    """Records every terminal check of a capped position made WITH adjudication."""
    import chaturanga.terminal as terminal
    real = terminal.is_terminal
    calls = []

    def spy(state, adjudicate=True):
        if adjudicate and state.ply_count >= get_move_cap():
            calls.append(state.ply_count)
        return real(state, adjudicate=adjudicate)

    monkeypatch.setattr(terminal, 'is_terminal', spy)
    return calls


def _tiny_net():
    import torch
    from net.model import create_chaturanga_net
    torch.manual_seed(0)
    return create_chaturanga_net(n_blocks=1, channels=8)


def test_n1_batched_arena_end_to_end_never_adjudicates(adjudicated_cap_checks):
    from pipeline.inference_server import run_batched_arena
    stats = run_batched_arena(_tiny_net(), _tiny_net(), _materially_won_one_ply_before_cap,
                              n_games=2, n_sims=8, device='cpu', amp=False)
    assert stats['draws'] == 2                       # both games hit the cap: draws
    assert adjudicated_cap_checks == [], "arena search adjudicated a capped position"


def test_n1_eval_mcts_player_never_adjudicates(adjudicated_cap_checks):
    from eval.evaluate import mcts_player
    player = mcts_player(_tiny_net(), 'cpu', n_sims=8, action_size=4096)
    state = _materially_won_one_ply_before_cap()
    assert player(state) in state.legal_moves()
    assert adjudicated_cap_checks == [], "evaluation search adjudicated a capped position"


def test_n1_spy_sees_training_search(adjudicated_cap_checks):
    """Guard for the two tests above: the spy does fire for a self-play search."""
    search(_materially_won_one_ply_before_cap(), _uniform, n_sims=8, add_noise=False)
    assert adjudicated_cap_checks, "spy never fired; the end-to-end tests would prove nothing"


# --- N13: human games score a capped game as a draw ------------------------

def test_n13_web_reports_capped_game_as_draw():
    pytest.importorskip('flask')
    from web import app as web_app
    state = _materially_won_one_ply_before_cap()
    web_app.current_session.state = state.apply(state.legal_moves()[0])
    data = web_app.serialize_state(web_app.current_session)
    assert data['is_terminal'] and data['result'] == 0.0 and data['terminal_reason'] == 'move_cap'
