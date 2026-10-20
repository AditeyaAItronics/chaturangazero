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


# --- N2: an interrupted iteration is redone, not skipped --------------------

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


def test_n2_kill_during_arena_redoes_the_iteration(tmp_path, monkeypatch):
    import json
    import pipeline.loop as loop
    from net.replay import ReplayBuffer
    from pipeline.resume import read_run_state

    out = str(tmp_path / 'run')
    real_arena = loop.run_batched_arena

    class Preempted(Exception):
        pass

    def killed_in_arena(*args, **kwargs):
        raise Preempted()

    # Iteration 1 is a gating iteration (every 2); the "VM dies" during its arena,
    # after checkpoint_0001.pt and the buffer snapshot are already on disk.
    monkeypatch.setattr(loop, 'run_batched_arena', killed_in_arena)
    with pytest.raises(Preempted):
        loop.training_loop(_tiny_connect4_config(iterations=2, arena_every=2), output_dir=out)
    assert os.path.exists(os.path.join(out, 'checkpoint_0001.pt'))
    marker = read_run_state(out)
    assert marker['last_completed_iteration'] == 0
    on_disk = ReplayBuffer(max_size=1000)
    on_disk.load(os.path.join(out, 'buffer.npz'))
    assert on_disk.total_added > marker['buffer_total_added'], "scenario: snapshot is ahead"

    monkeypatch.setattr(loop, 'run_batched_arena', real_arena)
    loop.training_loop(_tiny_connect4_config(iterations=2, arena_every=2), output_dir=out,
                       resume_from='auto')

    with open(os.path.join(out, 'metrics.json')) as f:
        metrics = json.load(f)
    assert [m['iteration'] for m in metrics] == [0, 1]
    assert 'arena' in metrics[1], "the gate of the interrupted iteration never ran"
    marker = read_run_state(out)
    assert marker['last_completed_iteration'] == 1
    final = ReplayBuffer(max_size=1000)
    final.load(os.path.join(out, 'buffer.npz'))
    assert final.total_added == marker['buffer_total_added'] == len(final), \
        "the interrupted iteration's positions were kept and then added again"


def test_n2_latest_checkpoint_trusts_the_marker(tmp_path):
    import torch
    from pipeline.resume import latest_checkpoint, write_run_state
    for i in range(3):
        torch.save({'iteration': i}, tmp_path / f'checkpoint_{i:04d}.pt')
    assert latest_checkpoint(str(tmp_path)).endswith('checkpoint_0002.pt')   # no marker: newest
    write_run_state(str(tmp_path), {'last_completed_iteration': 1})
    assert latest_checkpoint(str(tmp_path)).endswith('checkpoint_0001.pt')   # 0002 incomplete
    write_run_state(str(tmp_path), {'last_completed_iteration': 7})         # marker's file missing
    assert latest_checkpoint(str(tmp_path)).endswith('checkpoint_0002.pt')
    # Marker's checkpoint unreadable: fall back to an EARLIER one, never the
    # newer checkpoint of the cut-short iteration
    write_run_state(str(tmp_path), {'last_completed_iteration': 1})
    (tmp_path / 'checkpoint_0001.pt').write_bytes(b'corrupt')
    assert latest_checkpoint(str(tmp_path)).endswith('checkpoint_0000.pt')
    (tmp_path / 'run_state.json').write_text('{not json')                   # corrupt marker
    assert latest_checkpoint(str(tmp_path)).endswith('checkpoint_0002.pt')


def test_n2_resume_with_sparse_buffer_snapshots(tmp_path, monkeypatch):
    """buffer_snapshot_every=2: the only snapshot is written by the interrupted
    iteration. Resume must keep iteration 0's games (they were never snapshotted
    otherwise) and drop iteration 1's."""
    import pipeline.loop as loop
    from net.replay import ReplayBuffer
    from pipeline.resume import read_run_state

    out = str(tmp_path / 'run')
    config = _tiny_connect4_config(iterations=2, arena_every=2)
    config['cloud'] = {'buffer_snapshot_every': 2}
    real_arena = loop.run_batched_arena

    def killed_in_arena(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(loop, 'run_batched_arena', killed_in_arena)
    with pytest.raises(KeyboardInterrupt):
        loop.training_loop(config, output_dir=out)
    t0 = read_run_state(out)['buffer_total_added']
    monkeypatch.setattr(loop, 'run_batched_arena', real_arena)
    kept = []
    real_drop = ReplayBuffer.drop_newest

    def spy(self, k):
        real_drop(self, k)
        kept.append(self.total_added)

    monkeypatch.setattr(ReplayBuffer, 'drop_newest', spy)
    loop.training_loop(config, output_dir=out, resume_from='auto')
    assert kept == [t0], "resume should cut the buffer back to exactly iteration 0's games"
    final = ReplayBuffer(max_size=1000)
    final.load(os.path.join(out, 'buffer.npz'))
    assert final.total_added == read_run_state(out)['buffer_total_added'] == len(final)


@pytest.mark.parametrize('n_added', [5, 10, 13])
def test_n2_drop_newest_and_total_added_survive_save_load(tmp_path, n_added):
    from net.replay import ReplayBuffer
    buf = ReplayBuffer(max_size=10)
    for i in range(n_added):
        state = np.zeros((3, 6, 7), np.float32)
        state[0, 0, 0] = i % 2
        pi = np.zeros(7, np.float32)
        pi[i % 7] = 1.0
        buf.add(state, pi, float(i))           # value = insertion index, to track order
    buf.save(str(tmp_path / 'b.npz'))
    loaded = ReplayBuffer(max_size=10)
    loaded.load(str(tmp_path / 'b.npz'))
    assert loaded.total_added == n_added
    loaded.drop_newest(3)
    assert loaded.total_added == n_added - 3
    kept = sorted(loaded._values[:len(loaded)].tolist())
    expected = list(range(max(0, n_added - 10), n_added - 3))
    assert kept == expected
    # Adding after a drop continues the ring correctly
    loaded.add(np.zeros((3, 6, 7), np.float32), np.eye(7, dtype=np.float32)[0], 99.0)
    assert len(loaded) == min(len(expected) + 1, 10) and 99.0 in loaded._values


# --- N5: gradient steps follow the data (replay ratio + warm-up) ------------

def test_n5_training_steps_formula():
    from pipeline.loop import training_steps
    # The plan's cloud numbers: ~80k new positions, batch 512, ratio 3 -> 469 steps, not 2000
    assert training_steps(80_000, 1_000_000, 512, 2000, replay_ratio=3.0) == 469
    assert training_steps(10**7, 10**7, 512, 2000, replay_ratio=3.0) == 2000      # cap binds
    assert training_steps(80_000, 1_000_000, 512, 2000) == 2000                   # old behaviour
    assert training_steps(80_000, 40_000, 512, 2000, replay_ratio=3.0, min_buffer=50_000) == 0
    assert training_steps(100, 100, 512, 2000, replay_ratio=3.0) == 0             # < one batch


def test_n5_loop_uses_replay_ratio_and_warmup(tmp_path):
    import json
    import math
    from pipeline.loop import training_loop
    config = _tiny_connect4_config(iterations=6, arena_every=99)
    config['train'].update({'replay_ratio': 2.0, 'min_buffer': 170, 'steps_per_iter': 1000})
    out = str(tmp_path / 'run')
    training_loop(config, output_dir=out)
    with open(os.path.join(out, 'metrics.json')) as f:
        metrics = json.load(f)
    batch = config['train']['batch_size']
    trained_any = False
    for m in metrics:
        if m['buffer_size'] < max(170, batch):   # 170 > one iteration's max (4 x 42)
            assert m['train_steps'] == 0, m
        else:
            trained_any = True
            assert m['train_steps'] == math.ceil(2.0 * m['new_positions'] / batch), m
            assert 2.0 <= m['samples_per_new_position'] < 2.0 + batch / m['new_positions']
    assert trained_any, "the run never got past the warm-up"
    # Every position played is paid for exactly once, warm-up games included:
    # the new_positions of the training iterations add up to the buffer at the
    # last training iteration (the tiny run never fills the 1000-slot buffer)
    trained = [m for m in metrics if m['train_steps'] > 0]
    assert sum(m['new_positions'] for m in trained) == trained[-1]['buffer_size']
    assert metrics[0]['train_steps'] == 0 and metrics[0]['buffer_size'] < 170, \
        "scenario: the first iteration must be inside the warm-up" 


def test_n5_warmup_skips_gate_and_lr_schedule(tmp_path):
    import json
    import torch
    from pipeline.loop import training_loop
    config = _tiny_connect4_config(iterations=6, arena_every=1)
    config['train'].update({'replay_ratio': 2.0, 'min_buffer': 170, 'lr_milestones': [100]})
    out = str(tmp_path / 'run')
    training_loop(config, output_dir=out)
    with open(os.path.join(out, 'metrics.json')) as f:
        metrics = json.load(f)
    trained = 0
    for m in metrics:
        trained += m['train_steps'] > 0
        # a gate runs exactly when the challenger has trained since the previous gate
        assert ('arena' in m) == (m['train_steps'] > 0), m['iteration']
    assert metrics[0]['train_steps'] == 0 and 0 < trained < len(metrics)
    ckpt = torch.load(os.path.join(out, 'checkpoint_0005.pt'), map_location='cpu', weights_only=False)
    assert ckpt['scheduler_state_dict']['last_epoch'] == trained, \
        "lr_milestones must count training iterations, not warm-up ones"


def test_n5_capped_steps_carry_the_remainder_over(tmp_path):
    import json
    from pipeline.loop import training_loop
    config = _tiny_connect4_config(iterations=4, arena_every=99)
    # ratio 2, batch 16, at most 1 step: each iteration pays for only 8 positions
    config['train'].update({'replay_ratio': 2.0, 'steps_per_iter': 1})
    out = str(tmp_path / 'run')
    training_loop(config, output_dir=out)
    with open(os.path.join(out, 'metrics.json')) as f:
        metrics = json.load(f)
    for i, m in enumerate(metrics):
        assert m['train_steps'] == 1
        assert m['new_positions'] == m['buffer_size'] - 8 * i, \
            "positions the capped steps didn't pay for must carry over"


def test_n5_resume_keeps_untrained_warmup_positions(tmp_path, monkeypatch):
    import pipeline.loop as loop
    from pipeline.resume import read_run_state
    config = _tiny_connect4_config(iterations=1, arena_every=99)
    config['train'].update({'replay_ratio': 2.0, 'min_buffer': 10**6})   # never trains
    out = str(tmp_path / 'run')
    loop.training_loop(config, output_dir=out)
    marker = read_run_state(out)
    assert marker['trained_upto'] == 0 < marker['buffer_total_added']
    seen = []
    real = loop.training_steps
    monkeypatch.setattr(loop, 'training_steps', lambda new, *a, **k: seen.append(new) or real(new, *a, **k))
    config['loop']['iterations'] = 2
    loop.training_loop(config, output_dir=out, resume_from='auto')
    assert seen[0] > marker['buffer_total_added'], \
        "after resume, iteration 0's warm-up positions must still count as untrained"


# --- N4: the cloud VM never idles while not training ------------------------

INFRA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'infra')


def _bash():
    import shutil
    bash = shutil.which('bash')
    if bash is None:
        pytest.skip('bash not available')
    return bash


def _run_watchdog(tmp_path, max_age, timeout, keep_fresh=False, old_heartbeat=False):
    import subprocess
    import threading
    import time
    heartbeat = tmp_path / 'metrics.json'
    fired = tmp_path / 'fired'
    if old_heartbeat:                      # e.g. pulled from the bucket, hours old
        heartbeat.write_text('[]')
        os.utime(heartbeat, (time.time() - 10_000, time.time() - 10_000))
    stop = threading.Event()

    def writer():
        while not stop.is_set():
            heartbeat.write_text('[]')
            time.sleep(0.3)

    if keep_fresh:
        threading.Thread(target=writer, daemon=True).start()
    cmd = [_bash(), os.path.join(INFRA, 'watchdog.sh'), heartbeat.as_posix(),
           str(max_age), '1', 'touch', fired.as_posix()]
    try:
        subprocess.run(cmd, timeout=timeout, capture_output=True)
    except subprocess.TimeoutExpired:
        pass
    finally:
        stop.set()
    return fired.exists()


def test_n4_watchdog_fires_on_a_stale_heartbeat(tmp_path):
    assert _run_watchdog(tmp_path, max_age=2, timeout=15)


def test_n4_watchdog_grace_counts_from_its_own_start(tmp_path):
    """An hours-old metrics.json pulled from the bucket is not 'stale' at boot."""
    assert not _run_watchdog(tmp_path, max_age=4, timeout=3, old_heartbeat=True)


def test_n4_watchdog_quiet_while_training_progresses(tmp_path):
    assert not _run_watchdog(tmp_path, max_age=2, timeout=6, keep_fresh=True)


def _run_training_wrapper(tmp_path, exit_code: int, mig: str = 'cz-train-mig'):
    import subprocess
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    calls = tmp_path / 'gcloud_calls'
    stub = bin_dir / 'gcloud'
    stub.write_text(f'#!/usr/bin/env bash\necho "$@" >> "{calls.as_posix()}"\n', newline='\n')
    stub.chmod(0o755)
    env = dict(os.environ, CZ_MIG=mig, CZ_ZONE='us-central1-a',
               CZ_FAILURE_FILE=(tmp_path / 'fast_failures').as_posix())
    env['PATH'] = bin_dir.as_posix() + os.pathsep + env.get('PATH', '')
    script = (f'export PATH="{bin_dir.as_posix()}:$PATH"; '
              f'bash "{os.path.join(INFRA, "run_training.sh")}" bash -c "exit {exit_code}"')
    result = subprocess.run([_bash(), '-c', script], env=env, capture_output=True, text=True,
                            timeout=60)
    return result.returncode, (calls.read_text() if calls.exists() else '')


def test_n4_finished_run_resizes_the_group_to_zero(tmp_path):
    code, calls = _run_training_wrapper(tmp_path, exit_code=0)
    assert code == 0
    assert 'instance-groups managed resize cz-train-mig --size 0 --zone us-central1-a' in calls


@pytest.mark.parametrize('exit_code', [1, 143])
def test_n4_crash_or_preemption_does_not_resize(tmp_path, exit_code):
    code, calls = _run_training_wrapper(tmp_path, exit_code=exit_code)
    assert code == exit_code and calls == '', "only a finished run may stop the group"


def test_n8_repeated_fast_crashes_stop_the_group(tmp_path):
    """A crash that repeats on every start (bad config) must not cycle VMs forever."""
    for attempt in range(1, 6):
        code, calls = _run_training_wrapper(tmp_path, exit_code=1)
        assert code == 1
        if attempt < 5:
            assert calls == '', f"resized after only {attempt} failures"
    assert 'instance-groups managed resize cz-train-mig --size 0' in calls


def test_n8_preemption_resets_the_fast_failure_count(tmp_path):
    for _ in range(4):
        _run_training_wrapper(tmp_path, exit_code=1)
    _run_training_wrapper(tmp_path, exit_code=143)            # a preemption in between
    code, calls = _run_training_wrapper(tmp_path, exit_code=1)
    assert calls == '', "failures separated by a preemption are not 'in a row'"


def test_n4_no_mig_no_resize(tmp_path):
    code, calls = _run_training_wrapper(tmp_path, exit_code=0, mig='')
    assert code == 0 and calls == ''


def test_n4_startup_supervises_training():
    with open(os.path.join(INFRA, 'startup.sh'), encoding='utf-8') as f:
        script = f.read()
    assert 'nohup' not in script, "training must not be a fire-and-forget background job"
    for needle in ('Restart=on-failure', 'SuccessExitStatus=143', 'cz-watchdog.service',
                   'watchdog.sh', "trap 'on_error $LINENO' ERR", 'nvidia-smi',
                   'run_training.sh', 'MAX_STARTUP_FAILURES', 'CZ_MIG=$MIG'):
        assert needle in script, needle
    # 143 is the training loop's SIGTERM exit code
    with open(os.path.join(os.path.dirname(INFRA), 'pipeline', 'loop.py'), encoding='utf-8') as f:
        assert 'sys.exit(143)' in f.read()
    with open(os.path.join(INFRA, 'create_vm.sh'), encoding='utf-8') as f:
        vm = f.read()
    assert 'cz-heartbeat-minutes' in vm and 'cz-mig=' in vm and 'compute-rw' in vm


# --- N6: virtual-loss collisions are retried, never dropped -----------------

@pytest.mark.parametrize('leaves', [4, 8])
def test_n6_every_move_gets_its_full_simulation_budget(leaves):
    import torch
    from connect4.game import Connect4State
    from net.model import create_connect4_net
    from pipeline.inference_server import run_batched_self_play
    torch.manual_seed(0)
    np.random.seed(0)
    stats = run_batched_self_play(create_connect4_net(1, 8), Connect4State, total_games=4,
                                  num_parallel_games=4, n_sims=16, action_size=7,
                                  device='cpu', amp=False, reuse_subtree=False,
                                  leaves_per_step=leaves, virtual_loss=0.0, eval_cache_size=0)   # no VL: extra descents collide (worst case)
    assert stats['virtual_loss_collisions'] > 0, "scenario: collisions must actually happen"
    assert stats['simulations'] == 16 * stats['moves'], \
        f"{16 * stats['moves'] - stats['simulations']} simulations lost to collisions"
    assert stats['simulations_dropped'] == 0


def test_n6_search_accounting_reaches_metrics(tmp_path):
    import json
    from pipeline.loop import training_loop
    config = _tiny_connect4_config(iterations=1, arena_every=99)
    config['mcts'].update({'leaves_per_step': 4, 'virtual_loss': 0.0})
    out = str(tmp_path / 'run')
    training_loop(config, output_dir=out)
    with open(os.path.join(out, 'metrics.json')) as f:
        m = json.load(f)[0]
    assert m['simulations_per_move'] == config['mcts']['sims']
    assert m['virtual_loss_collisions'] > 0


# --- N7: self-play worker processes live for the whole run ------------------

def _connect4_selfplay(pool=None, games=4, workers=2, parallel=4):
    from net.model import create_connect4_net
    from pipeline.workers import run_parallel_self_play
    return run_parallel_self_play(create_connect4_net(1, 8), 'connect4', total_games=games,
                                  workers=workers, device='cpu', pool=pool, n_sims=4,
                                  action_size=7, num_parallel_games=parallel, amp=False)


def test_n7_pool_reuses_worker_processes():
    from pipeline.workers import SelfPlayPool
    with SelfPlayPool(2) as pool:
        first = _connect4_selfplay(pool)
        second = _connect4_selfplay(pool)
    assert first['games'] == second['games'] == 4
    assert first['worker_pids'] == second['worker_pids'] and len(first['worker_pids']) == 2, \
        "the second iteration started new worker processes"
    assert pool._pool is None, "the pool must be shut down on exit"


def test_n7_warns_when_parallel_games_cannot_take_effect(capsys):
    stats = _connect4_selfplay(games=4, workers=2, parallel=8)    # 2 games per worker < 8
    assert 'games per worker < parallel_games=8' in capsys.readouterr().out
    assert stats['concurrent_games_per_worker'] == 2


def test_n7_training_loop_starts_the_pool_once(tmp_path, monkeypatch):
    import pipeline.workers as workers
    created = []
    real_init = workers.SelfPlayPool.__init__

    def counting_init(self, n):
        created.append(n)
        real_init(self, n)

    monkeypatch.setattr(workers.SelfPlayPool, '__init__', counting_init)
    from pipeline.loop import training_loop
    config = _tiny_connect4_config(iterations=2, arena_every=99)
    config['loop'].update({'workers': 2, 'parallel_games': 2})
    training_loop(config, output_dir=str(tmp_path / 'run'))
    assert created == [2], f"expected one pool for the run, got {created}"


# --- N11 / N12: arena search budget and gate statistics ---------------------

def test_n12_score_interval():
    from search.arena import score_interval
    # 50% over 60 games: half-width ~1.96 x 6.45 points
    lo, hi = score_interval([1] * 30 + [-1] * 30)
    assert abs((hi - lo) / 2 / 1.96 - 0.5 / np.sqrt(60)) < 0.003
    # The plan's old gate: 55% over 60 games is well inside the interval around 50%
    assert lo < 0.5 < hi and hi > 0.6
    # 200 games: SE ~3.5 points
    lo, hi = score_interval([1] * 100 + [-1] * 100)
    assert abs((hi - lo) / 2 / 1.96 - 0.5 / np.sqrt(200)) < 0.001
    # Never collapses at the extremes; always inside [0, 1]
    lo, hi = score_interval([1] * 5)
    assert hi > 1 - 1e-9 and lo < 0.6
    lo, hi = score_interval([-1] * 5)
    assert lo < 1e-9 and hi > 0.4


@pytest.mark.parametrize('arena_sims', [None, 3])
def test_n11_arena_uses_the_configured_search_budget(tmp_path, monkeypatch, arena_sims):
    import json
    import pipeline.loop as loop
    seen = []
    real = loop.run_batched_arena

    def spy(*args, **kwargs):
        seen.append(kwargs['n_sims'])
        return real(*args, **kwargs)

    monkeypatch.setattr(loop, 'run_batched_arena', spy)
    config = _tiny_connect4_config(iterations=1, arena_every=1)
    if arena_sims is not None:
        config['arena']['sims'] = arena_sims
    out = str(tmp_path / 'run')
    loop.training_loop(config, output_dir=out)
    expected = config['mcts']['sims'] if arena_sims is None else arena_sims
    assert seen == [expected], "the arena must not silently halve the search budget"
    with open(os.path.join(out, 'metrics.json')) as f:
        arena = json.load(f)[0]['arena']
    assert arena['sims'] == expected and len(arena['score_ci95']) == 2
    assert arena['score_ci95'][0] <= arena['score'] <= arena['score_ci95'][1]


# --- N3: the web app refuses moves after the game or out of turn ------------

@pytest.fixture
def web_client():
    pytest.importorskip('flask')
    from web import app as web_app
    s = web_app.current_session
    saved = (s.game, s.p1_type, s.p2_type, s.state, list(s.history), list(s.move_log))
    yield web_app, web_app.app.test_client()
    s.game, s.p1_type, s.p2_type, s.state, s.history, s.move_log = saved


def _raja_captured_position() -> State:
    """Black to move, Black's Raja already captured: the game is over."""
    board = np.zeros(64, dtype=np.int8)
    board[sq(0, 0)] = RAJA
    board[sq(7, 7)] = RATHA
    board[sq(6, 0)] = -PADATI        # Black still has a pseudo-legal move
    return State(board=board, side_to_move=BLACK, ply_count=20)


def test_n3_no_moves_after_game_over(web_client):
    web_app, client = web_client
    web_app.current_session.game = 'chaturanga'
    web_app.current_session.p1_type = web_app.current_session.p2_type = 'human'
    web_app.current_session.state = _raja_captured_position()
    move = web_app.current_session.state.legal_moves()[0]
    r = client.post('/api/make_move', json={'action': move})
    assert r.status_code == 400 and 'terminal' in r.get_json()['error']


def test_n3_no_move_at_the_cap_crashes(web_client):
    web_app, client = web_client
    web_app.current_session.game = 'chaturanga'
    web_app.current_session.p1_type = web_app.current_session.p2_type = 'human'
    state = _materially_won_one_ply_before_cap()
    web_app.current_session.state = state.apply(state.legal_moves()[0])     # at the cap
    r = client.post('/api/make_move', json={'action': web_app.current_session.state.legal_moves()[0]})
    assert r.status_code == 400, "a move at the cap used to trip apply_move's assertion (HTTP 500)"


def test_n3_human_cannot_move_for_the_ai(web_client):
    web_app, client = web_client
    web_app.current_session.game = 'chaturanga'
    web_app.current_session.p1_type, web_app.current_session.p2_type = 'random', 'human'
    web_app.current_session.state = State()            # White (the AI) to move
    r = client.post('/api/make_move', json={'action': State().legal_moves()[0]})
    assert r.status_code == 400
    web_app.current_session.p1_type = 'human'
    r = client.post('/api/make_move', json={'action': State().legal_moves()[0]})
    assert r.status_code == 200


# --- N8: a config's rules: block must match the engine ----------------------

@pytest.mark.parametrize('name', ['cloud.yaml', 'local.yaml'])
def test_n8_shipped_configs_match_the_engine(name):
    from chaturanga.board import check_ruleset
    from pipeline.loop import load_config
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    check_ruleset(load_config(os.path.join(root, 'configs', name))['rules'])


@pytest.mark.parametrize('change', [{'stalemate': 'loss'}, {'bare_king': 'draw'},
                                    {'repetition_draw': True}, {'en_passant': True}])
def test_n8_a_rule_edit_in_yaml_fails_loudly(tmp_path, change):
    from chaturanga.board import RULESET
    from pipeline.loop import load_config, training_loop
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = load_config(os.path.join(root, 'configs', 'local.yaml'))
    config['rules'] = {**RULESET, **change}
    with pytest.raises(ValueError, match='does not match the engine'):
        training_loop(config, output_dir=str(tmp_path / 'run'), device='cpu')
    assert not (tmp_path / 'run' / 'config.yaml').exists(), "rejected before writing anything"


# --- N9: the ply plane doesn't change meaning when the cap is tightened -----

def test_n9_ply_plane_is_independent_of_the_cap():
    state = State(ply_count=150)
    at_300 = state.encode()[12, 0, 0]
    configure_rules(move_cap=200)          # the §2.2 response to the shuffling attractor
    at_200 = State(ply_count=150).encode()[12, 0, 0]
    assert at_300 == at_200 == pytest.approx(0.5), \
        "tightening the cap re-scaled the network input (old code: 0.5 -> 0.75)"


# --- N10: missed-Raja-capture counter separates sampling from search --------

def test_n10_counter_split():
    from pipeline.inference_server import RAJA_CAPTURE_COUNTERS, _count_raja_capture
    stats = dict.fromkeys(RAJA_CAPTURE_COUNTERS, 0)
    visits = np.zeros(4096)
    visits[100] = 10                         # search's favourite: the Raja capture
    # sampled (temperature 1) move that missed: not search's fault
    _count_raja_capture(stats, {100}, action=7, visits=visits, temperature=1.0)
    # greedy move that captured
    _count_raja_capture(stats, {100}, action=100, visits=visits, temperature=0.0)
    assert stats == {'raja_capture_chances': 2, 'raja_captures_missed': 1,
                     'raja_capture_chances_greedy': 1, 'raja_captures_missed_greedy': 0,
                     'raja_captures_missed_by_search': 0}
    visits[100], visits[7] = 1, 10           # now search itself prefers another move
    _count_raja_capture(stats, {100}, action=7, visits=visits, temperature=0.0)
    assert stats['raja_captures_missed_greedy'] == 1 and stats['raja_captures_missed_by_search'] == 1


@pytest.mark.parametrize('temp_threshold, all_greedy', [(0, True), (10**6, False)])
def test_n10_selfplay_reports_greedy_counts(temp_threshold, all_greedy):
    import torch
    from net.model import create_chaturanga_net
    from pipeline.inference_server import run_batched_self_play
    torch.manual_seed(0)
    np.random.seed(1)
    stats = run_batched_self_play(create_chaturanga_net(1, 8), State, total_games=2,
                                  num_parallel_games=2, n_sims=2, device='cpu', amp=False,
                                  temp_threshold=temp_threshold, eval_cache_size=0)
    assert stats['raja_capture_chances'] > 0, "scenario: random nets leave the Raja hanging"
    if all_greedy:
        assert stats['raja_capture_chances_greedy'] == stats['raja_capture_chances']
        assert stats['raja_capture_miss_rate_greedy'] == stats['raja_capture_miss_rate']
    else:
        assert stats['raja_capture_chances_greedy'] == 0
