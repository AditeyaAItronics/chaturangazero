"""
Tests for the MCTS search and neural network.
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connect4.game import Connect4State
from search.mcts import search, select_action_with_temperature
from search.baselines import random_player
from search.arena import play_game, arena
from net.model import create_connect4_net, create_chaturanga_net, DualHeadNet
from net.replay import ReplayBuffer
from net.train import Trainer, create_evaluator


def test_connect4_random_game():
    """A random Connect-4 game should terminate."""
    state = Connect4State()
    while True:
        done, z = state.is_terminal()
        if done:
            break
        moves = state.legal_moves()
        action = np.random.choice(moves)
        state = state.apply(action)
    
    assert state.ply_count <= 42
    print(f"  PASS: Connect-4 random game ended in {state.ply_count} plies, z={z}")


def test_mcts_with_random_eval():
    """MCTS should run without crashing using a random evaluator."""
    state = Connect4State()
    
    def random_eval(s):
        policy = np.ones(7, dtype=np.float32) / 7
        value = np.random.uniform(-1, 1)
        return policy, value
    
    visit_counts = search(
        state=state,
        evaluate=random_eval,
        n_sims=20,
        action_size=7,
        c_puct=1.5,
        add_noise=True,
        dirichlet_alpha=1.0,
    )
    
    assert visit_counts.shape == (7,)
    assert np.sum(visit_counts) > 0
    print(f"  PASS: MCTS ran 20 sims, visits = {visit_counts}")


def test_temperature_selection():
    """Temperature 0 should select the most-visited action."""
    visits = np.array([0, 5, 3, 1, 0, 0, 10], dtype=np.float32)
    
    action, pi = select_action_with_temperature(visits, temperature=0.0)
    assert action == 6, f"Expected action 6 (most visited), got {action}"
    print(f"  PASS: temperature=0 selects argmax, pi={pi}")


def test_connect4_net():
    """Connect-4 network should produce correct output shapes."""
    net = create_connect4_net(n_blocks=2, channels=32)
    x = torch.randn(4, 3, 6, 7)  # batch of 4
    
    policy, value = net(x)
    
    assert policy.shape == (4, 7), f"Policy shape: {policy.shape}"
    assert value.shape == (4, 1), f"Value shape: {value.shape}"
    
    # Value should be in [-1, 1] due to tanh
    assert torch.all(value >= -1) and torch.all(value <= 1)
    print(f"  PASS: Connect-4 net shapes correct, params={net.count_parameters():,}")


def test_chaturanga_net():
    """Chaturanga network should produce correct output shapes."""
    net = create_chaturanga_net(n_blocks=2, channels=32)
    x = torch.randn(4, 14, 8, 8)
    
    policy, value = net(x)
    
    assert policy.shape == (4, 4096), f"Policy shape: {policy.shape}"
    assert value.shape == (4, 1), f"Value shape: {value.shape}"
    print(f"  PASS: Chaturanga net shapes correct, params={net.count_parameters():,}")


def test_replay_buffer():
    """Replay buffer should store and sample correctly."""
    buf = ReplayBuffer(max_size=100)
    
    # Add 50 positions from random Connect-4 games (real encodings: binary or
    # constant planes, which is what the compact buffer stores)
    game_state = Connect4State()
    for i in range(50):
        if game_state.is_terminal()[0]:
            game_state = Connect4State()
        state = game_state.encode()
        game_state = game_state.apply(int(np.random.choice(game_state.legal_moves())))
        policy = np.random.dirichlet(np.ones(7)).astype(np.float32)
        value = np.random.uniform(-1, 1)
        buf.add(state, policy, value)
    
    assert len(buf) == 50
    
    states, policies, values = buf.sample(16)
    assert states.shape == (16, 3, 6, 7)
    assert policies.shape == (16, 7)
    assert values.shape == (16, 1)
    print("  PASS: replay buffer store and sample")


def test_trainer_step():
    """A single training step should reduce loss without crashing."""
    net = create_connect4_net(n_blocks=1, channels=16)
    trainer = Trainer(net, lr=1e-3, device='cpu')
    
    states = np.random.randn(8, 3, 6, 7).astype(np.float32)
    policies = np.random.dirichlet(np.ones(7), size=8).astype(np.float32)
    values = np.random.uniform(-1, 1, (8, 1)).astype(np.float32)
    
    stats = trainer.train_step(states, policies, values)
    
    assert 'policy_loss' in stats
    assert 'value_loss' in stats
    assert stats['total_loss'] > 0
    print(f"  PASS: training step, loss={stats['total_loss']:.4f}")


def test_evaluator():
    """The evaluate callable should work with MCTS."""
    net = create_connect4_net(n_blocks=1, channels=16)
    evaluate = create_evaluator(net, device='cpu', action_size=7)
    
    state = Connect4State()
    policy, value = evaluate(state)
    
    assert policy.shape == (7,)
    assert -1 <= value <= 1
    assert abs(np.sum(policy) - 1.0) < 0.01, f"Policy doesn't sum to 1: {np.sum(policy)}"
    print(f"  PASS: evaluator produces valid output, v={value:.3f}")


def test_mcts_with_network():
    """MCTS driven by a real network should work end-to-end."""
    net = create_connect4_net(n_blocks=1, channels=16)
    evaluate = create_evaluator(net, device='cpu', action_size=7)
    
    state = Connect4State()
    visit_counts = search(
        state=state,
        evaluate=evaluate,
        n_sims=10,
        action_size=7,
        c_puct=1.5,
        add_noise=True,
        dirichlet_alpha=1.0,
    )
    
    assert np.sum(visit_counts) > 0
    action, pi = select_action_with_temperature(visit_counts, 1.0, state.legal_moves())
    assert action in state.legal_moves()
    print(f"  PASS: MCTS + network end-to-end, action={action}")


def test_arena_random_vs_random():
    """Arena should run games between two random players."""
    def p1(state):
        return random_player(state, action_size=7)
    
    def p2(state):
        return random_player(state, action_size=7)
    
    result = arena(p1, p2, Connect4State, n_games=10, action_size=7)
    
    assert result['p1_wins'] + result['p2_wins'] + result['draws'] == 10
    print(f"  PASS: arena 10 games, P1 {result['p1_wins']}-{result['draws']}-{result['p2_wins']} P2")


if __name__ == '__main__':
    print("=" * 50)
    print("MCTS & Network Tests")
    print("=" * 50)
    
    tests = [
        test_connect4_random_game,
        test_mcts_with_random_eval,
        test_temperature_selection,
        test_connect4_net,
        test_chaturanga_net,
        test_replay_buffer,
        test_trainer_step,
        test_evaluator,
        test_mcts_with_network,
        test_arena_random_vs_random,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n{test.__name__}:")
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 50}")
