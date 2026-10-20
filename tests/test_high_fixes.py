"""
Regression tests for the high-priority bugs listed in needtofix.md (H1–H9).
H6 (the two tests that could never fail) is fixed in tests/test_rules.py itself.
"""

import os
import sys
import random
import builtins

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chaturanga.board import (
    State, WHITE, BLACK, PADATI, ASHVA, RATHA, RAJA, sq,
    configure_rules, get_move_cap, get_adjudicate_margin,
)
from chaturanga.encoding import MIRROR_PERM
from chaturanga.moves import encode_action, apply_move
from connect4.game import Connect4State
from net.model import (create_chaturanga_net, create_connect4_net,
                       load_net_from_checkpoint, net_config_from_state_dict)
from net.replay import ReplayBuffer
from net.train import Trainer
from search.arena import play_game
from search.baselines import (alpha_beta_player, greedy_material_player, random_player,
                              _evaluate_position)
from eval.elo import compute_elo


@pytest.fixture(autouse=True)
def restore_rules():
    """configure_rules() is process-global; never leak a changed cap into other tests."""
    cap, margin = get_move_cap(), get_adjudicate_margin()
    yield
    configure_rules(move_cap=cap, adjudicate_margin=margin)


def _random_position(n_plies: int, seed: int) -> State:
    rng = np.random.default_rng(seed)
    state = State()
    for _ in range(n_plies):
        if state.is_terminal()[0]:
            break
        state = state.apply(int(rng.choice(state.legal_moves())))
    return state


# --- H1: the human-play CLI no longer crashes on startup ---------------------

def test_h1_cli_baselines_resolve_and_play_legal_moves():
    from eval.play_cli import make_baseline
    state = State()
    for name in ('random', 'greedy', 'minimax'):
        player = make_baseline(name)
        assert player is not None, name
        assert player(state) in state.legal_moves(), name
    assert make_baseline('mcts') is None


def test_h1_cli_starts_against_a_baseline(monkeypatch):
    from eval.play_cli import play_cli
    monkeypatch.setattr(builtins, 'input', lambda prompt='': 'quit')
    play_cli(human_side=WHITE, opponent='random', device='cpu')   # used to raise ImportError


# --- H2: checkpoints carry their architecture --------------------------------

def test_h2_checkpoint_roundtrip_any_architecture(tmp_path):
    torch.manual_seed(0)
    net = create_chaturanga_net(n_blocks=2, channels=24)
    path = str(tmp_path / 'ckpt.pt')
    Trainer(net, device='cpu').save_checkpoint(path)

    loaded, ckpt = load_net_from_checkpoint(path, device='cpu')
    assert ckpt['net_config'] == net.config
    assert loaded.config['n_blocks'] == 2 and loaded.config['channels'] == 24

    x = torch.randn(3, 14, 8, 8)
    net.eval()
    with torch.no_grad():
        for a, b in zip(net(x), loaded(x)):
            assert torch.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize('factory,blocks,channels', [
    (create_chaturanga_net, 3, 16),
    (create_connect4_net, 2, 8),
])
def test_h2_legacy_checkpoint_architecture_is_inferred(tmp_path, factory, blocks, channels):
    net = factory(n_blocks=blocks, channels=channels)
    path = str(tmp_path / 'legacy.pt')
    torch.save({'model_state_dict': net.state_dict()}, path)   # no 'net_config'
    assert net_config_from_state_dict(net.state_dict()) == net.config
    loaded, _ = load_net_from_checkpoint(path, device='cpu')
    assert loaded.config == net.config


# --- H3: masked policy loss --------------------------------------------------

def _sample_positions(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    out, state = [], State()
    while len(out) < n:
        if state.is_terminal()[0]:
            state = State()
        legal = state.legal_moves()
        visited = rng.choice(legal, size=max(1, len(legal) // 3), replace=False)
        pi = np.zeros(4096, dtype=np.float32)
        pi[visited] = rng.dirichlet(np.ones(len(visited)))
        legal_canonical = np.flatnonzero(state.orient_policy(state.legal_mask()))
        out.append((state.encode(), state.orient_policy(pi), 0.0, legal_canonical))
        state = state.apply(int(rng.choice(legal)))
    return out


def test_h3_buffer_returns_full_legal_masks_mirrored_with_pi():
    samples = _sample_positions(20)
    buf = ReplayBuffer(max_size=20, mirror_perm=MIRROR_PERM)
    for s, p, v, legal in samples:
        buf.add(s, p, v, legal=legal)
    np.random.seed(3)
    states, policies, _, masks = buf.sample(200, return_masks=True)
    for st, pol, mask in zip(states, policies, masks):
        assert np.all(mask[pol > 0]), "π puts mass on a move the mask calls illegal"
        for s, p, _, legal in samples:
            if np.array_equal(st, s):
                assert set(np.flatnonzero(mask)) == set(legal.tolist())
                break
            if np.array_equal(st, s[..., ::-1]):
                assert set(np.flatnonzero(mask)) == set(MIRROR_PERM[legal].tolist())
                break
        else:
            raise AssertionError("sampled state not found in the inserted positions")
        # The mask must include legal moves MCTS never visited, not just π's support
    assert masks.sum() > (policies > 0).sum()


def test_h3_masked_loss_is_cross_entropy_over_legal_moves_only():
    torch.manual_seed(0)
    net = create_chaturanga_net(n_blocks=1, channels=16)
    samples = _sample_positions(8, seed=1)
    states = np.stack([s for s, _, _, _ in samples])
    pis = np.stack([p for _, p, _, _ in samples])
    values = np.zeros((8, 1), dtype=np.float32)
    masks = np.zeros((8, 4096), dtype=bool)
    for i, (_, _, _, legal) in enumerate(samples):
        masks[i, legal] = True

    net.eval()   # use running BN stats so the manual check sees the same logits
    with torch.no_grad():
        logits, _ = net(torch.tensor(states))
    expected = 0.0
    for i in range(8):
        legal_logits = logits[i][torch.tensor(masks[i])]
        log_p = torch.log_softmax(legal_logits, dim=0)
        expected -= float((torch.tensor(pis[i][masks[i]]) * log_p).sum())
    expected /= 8

    trainer = Trainer(net, lr=0.0, device='cpu')    # lr=0: the step must not change weights
    trainer.net.eval()
    trainer.net.train = lambda mode=True: trainer.net   # keep eval-mode BN for the comparison
    stats = trainer.train_step(states, pis, values, legal_masks=masks)
    assert np.isfinite(stats['policy_loss']) and np.isfinite(stats['entropy'])
    assert abs(stats['policy_loss'] - expected) < 1e-4, (stats['policy_loss'], expected)


def test_h3_train_epoch_uses_masks(monkeypatch):
    samples = _sample_positions(40)
    buf = ReplayBuffer(max_size=40)
    for s, p, v, legal in samples:
        buf.add(s, p, v, legal=legal)
    trainer = Trainer(create_chaturanga_net(n_blocks=1, channels=8), device='cpu')
    seen = []
    original = trainer.train_step
    monkeypatch.setattr(trainer, 'train_step',
                        lambda *a, **kw: seen.append(kw.get('legal_masks')) or original(*a, **kw))
    trainer.train_epoch(buf, steps=2, batch_size=16)
    assert len(seen) == 2 and all(m is not None and m.shape == (16, 4096) for m in seen)


# --- H4: no material adjudication in arena / baseline games ------------------

def _capped_position(ply: int) -> State:
    """White up a Ratha and an Ashva at the move cap: adjudication says White wins."""
    board = np.zeros(64, dtype=np.int8)
    board[sq(0, 4)] = RAJA; board[sq(0, 0)] = RATHA; board[sq(0, 1)] = ASHVA
    board[sq(7, 4)] = -RAJA; board[sq(6, 7)] = -PADATI
    return State(board=board, side_to_move=WHITE, ply_count=ply)


def test_h4_adjudicate_switch():
    state = _capped_position(get_move_cap())
    assert state.is_terminal() == (True, 1.0)                    # training: material
    assert state.is_terminal(adjudicate=False) == (True, 0.0)    # evaluation: draw


def test_h4_arena_game_at_cap_is_a_draw():
    result, _, _ = play_game(random_player, random_player, _capped_position(get_move_cap()))
    assert result == 0.0


# --- H5: config values take effect; hard cap invariant -----------------------

def test_h5_configure_rules_moves_cap_and_margin():
    configure_rules(move_cap=200, adjudicate_margin=2.0)
    assert not _capped_position(199).is_terminal()[0]
    assert _capped_position(200).is_terminal()[0]
    assert _capped_position(199).encode()[12, 0, 0] == pytest.approx(199 / 200)

    # margin 2.0: White up exactly 3 points (Ratha + Ashva - Padati... ) still wins
    board = np.zeros(64, dtype=np.int8)
    board[sq(0, 4)] = RAJA; board[sq(0, 1)] = ASHVA
    board[sq(7, 4)] = -RAJA; board[sq(6, 7)] = -PADATI     # diff = 3 - 1 = 2
    state = State(board=board, side_to_move=WHITE, ply_count=200)
    assert state.is_terminal() == (True, 1.0)
    configure_rules(adjudicate_margin=4.0)
    assert state.is_terminal() == (True, 0.0)


def test_h5_applying_a_move_past_the_cap_asserts():
    state = _capped_position(get_move_cap())
    with pytest.raises(AssertionError, match="exceeded cap"):
        apply_move(state, state.legal_moves()[0])


def test_h5_training_loop_applies_yaml_move_cap(tmp_path):
    from pipeline.loop import training_loop
    config = {
        'game': 'chaturanga', 'seed': 2, 'device': 'cpu',
        'net': {'blocks': 1, 'channels': 8, 'action_space': 4096, 'input_planes': 14},
        'mcts': {'sims': 2, 'c_puct': 1.5, 'dirichlet_alpha': 0.3, 'dirichlet_eps': 0.25,
                 'reuse_subtree': False},
        'play': {'temp_moves': 4, 'move_cap': 12, 'adjudicate_margin': 4.0},
        'train': {'lr': 1e-3, 'l2': 1e-4, 'batch_size': 8, 'buffer_size': 1000,
                  'steps_per_iter': 1, 'grad_clip': 1.0, 'amp_inference': False,
                  'mirror_augment': True},
        'arena': {'games': 2, 'win_threshold': 0.55, 'every_n_iters': 99},
        'loop': {'games_per_iter': 3, 'iterations': 1, 'parallel_games': 3},
    }
    training_loop(config, output_dir=str(tmp_path / 'run'))
    assert get_move_cap() == 12
    import json
    with open(tmp_path / 'run' / 'metrics.json') as f:
        metrics = json.load(f)[0]
    assert metrics['avg_plies'] <= 12, metrics


# --- H7: alpha-beta evaluation and search ------------------------------------

def test_h7_evaluation_is_antisymmetric():
    """Same position, other side to move: the score must flip sign (old mobility didn't)."""
    assert _evaluate_position(State()) == 0.0
    for seed in range(10):
        s = _random_position(10 + seed, seed)
        if s.is_terminal()[0]:
            continue
        flipped = State(board=s.board, side_to_move=-s.side_to_move, ply_count=s.ply_count)
        assert _evaluate_position(s) == pytest.approx(-_evaluate_position(flipped))


def test_h7_alpha_beta_takes_the_raja_now_not_later():
    """
    Black's Raja on a8 is boxed in by Rathas on the a- and b-files, so many White
    moves win within 3 plies. With depth-blind scores they all tied at 1000 and
    the first one generated (a1-a2) was played; now the immediate capture wins.
    """
    board = np.zeros(64, dtype=np.int8)
    board[sq(0, 0)] = RATHA; board[sq(0, 1)] = RATHA; board[sq(0, 7)] = RAJA
    board[sq(7, 0)] = -RAJA; board[sq(6, 7)] = -PADATI
    state = State(board=board, side_to_move=WHITE)
    capture = encode_action(sq(0, 0), sq(7, 0))
    for seed in range(5):
        np.random.seed(seed)
        assert alpha_beta_player(state, depth=3) == capture


# --- H8: baselines break ties randomly ---------------------------------------

def test_h8_greedy_varies_its_choice_among_equal_moves():
    state = State()   # no captures available: every move scores 0
    choices = set()
    for seed in range(30):
        np.random.seed(seed)
        choices.add(greedy_material_player(state))
    assert len(choices) > 1


# --- H9: Elo is an order-independent fit -------------------------------------

def _results(seed: int = 0):
    rng = np.random.default_rng(seed)
    strength = {'a': 0.0, 'b': 200.0, 'c': 400.0, 'd': 600.0}
    out = []
    names = list(strength)
    for _ in range(400):
        p1, p2 = rng.choice(names, size=2, replace=False)
        p_win = 1 / (1 + 10 ** ((strength[p2] - strength[p1]) / 400))
        out.append({'player1': p1, 'player2': p2, 'score': float(rng.random() < p_win)})
    return out


def test_h9_elo_does_not_depend_on_game_order():
    results = _results()
    first = compute_elo(results)
    shuffled = list(results)
    random.Random(7).shuffle(shuffled)
    second = compute_elo(shuffled)
    for name in first:
        assert first[name] == pytest.approx(second[name], abs=1e-6)


def test_h9_elo_recovers_strength_order_and_stays_finite():
    ratings = compute_elo(_results())
    assert ratings['a'] < ratings['b'] < ratings['c'] < ratings['d']
    # roughly the true 600-point spread (prior shrinks it a little)
    assert 400 < ratings['d'] - ratings['a'] < 800
    unbeaten = compute_elo([{'player1': 'x', 'player2': 'y', 'score': 1.0}] * 10)
    assert all(np.isfinite(v) for v in unbeaten.values())
    assert unbeaten['x'] > unbeaten['y']
