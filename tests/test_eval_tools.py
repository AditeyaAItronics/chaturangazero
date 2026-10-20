"""
Smoke tests for the Phase 6 evaluation toolkit (eval/evaluate.py), the
profiling harness (pipeline/profile.py) and the cloud sync/resume pieces
(pipeline/gcs.py, pipeline/resume.py) — needtofix I3, I4, I5.
"""

import json
import os

import numpy as np
import pytest
import torch

from tests.conftest import ROOT


@pytest.fixture(scope='module')
def tiny_run(tmp_path_factory):
    """A real (tiny) Connect-4 training run with 3 checkpoints."""
    from pipeline.loop import training_loop
    out = tmp_path_factory.mktemp('run')
    config = {
        'game': 'connect4', 'seed': 3, 'device': 'cpu',
        'net': {'blocks': 1, 'channels': 8, 'action_space': 7, 'input_planes': 3},
        'mcts': {'sims': 4, 'c_puct': 1.5, 'dirichlet_alpha': 1.0, 'dirichlet_eps': 0.25},
        'play': {'temp_moves': 4, 'move_cap': 42},
        'train': {'lr': 1e-3, 'l2': 1e-4, 'batch_size': 16, 'buffer_size': 2000,
                  'steps_per_iter': 2, 'grad_clip': 1.0, 'amp_inference': False,
                  'mirror_augment': True},
        'arena': {'games': 2, 'win_threshold': 0.55, 'every_n_iters': 2},
        'loop': {'games_per_iter': 4, 'iterations': 3, 'parallel_games': 4},
    }
    training_loop(config, output_dir=str(out))
    return out


def _run(*argv):
    from eval.evaluate import main
    main(['--device', 'cpu', *argv])


def test_ladder_writes_an_elo_curve(tiny_run):
    _run('ladder', str(tiny_run), '--games', '2', '--sims', '4', '--neighbors', '2', '--opening-plies', '1')
    elo = json.load(open(tiny_run / 'elo.json'))
    assert [it for it, _ in elo['curve']] == [0, 1, 2]
    assert (tiny_run / 'elo_curve.png').exists()


def test_baselines_sims_match_and_game(tiny_run, tmp_path):
    ckpt = str(tiny_run / 'checkpoint_0002.pt')
    _run('baselines', ckpt, '--games', '2', '--sims', '4', '--out', str(tmp_path / 'b.json'))
    assert set(json.load(open(tmp_path / 'b.json'))['results']) == {'random', 'greedy'}

    _run('sims', ckpt, '--games', '2', '--levels', '2', '4', '8', '--opening-plies', '1')
    curve = json.load(open(tiny_run / 'sims_strength.json'))['curve']
    assert [lvl for lvl, _ in curve] == [2, 4, 8]

    _run('match', ckpt, str(tiny_run / 'checkpoint_0000.pt'), '--games', '2', '--sims', '4',
         '--out', str(tmp_path / 'm.json'))
    assert json.load(open(tmp_path / 'm.json'))['p1_wins'] >= 0

    _run('game', ckpt, '--sims', '4', '--opponent', 'random', '--out', str(tmp_path / 'g.md'))
    text = open(tmp_path / 'g.md', encoding='utf-8').read()
    assert '**Result:**' in text and '| 1 | agent |' in text


def test_sample_game_for_chaturanga(tmp_path):
    from net.model import create_chaturanga_net
    from net.train import Trainer
    ckpt = str(tmp_path / 'c.pt')
    Trainer(create_chaturanga_net(1, 8), device='cpu').save_checkpoint(ckpt)
    _run('game', ckpt, '--sims', '2', '--opponent', 'random', '--out', str(tmp_path / 'g.md'))
    text = open(tmp_path / 'g.md', encoding='utf-8').read()
    assert '**Result:**' in text and 'Final position' in text


def test_plots(tiny_run):
    _run('plots', str(tiny_run))
    for name in ('loss.png', 'value_head.png', 'cap_and_draws.png', 'terminal_reasons.png'):
        assert (tiny_run / 'plots' / name).exists(), name


def test_profile_harness_runs(tmp_path):
    from pipeline.profile import profile_once
    from pipeline.loop import load_config
    from net.model import create_connect4_net
    config = load_config(os.path.join(ROOT, 'configs', 'connect4.yaml'))
    row = profile_once(config, create_connect4_net(1, 8), games=4, device=torch.device('cpu'),
                       settings={'workers': 1, 'parallel_games': 4, 'sims': 4})
    assert row['games'] == 4 and row['games_per_hour'] > 0 and 0 < row['network_time_share'] <= 1


def test_local_sync_and_auto_resume(tmp_path):
    from pipeline import gcs
    from pipeline.resume import latest_checkpoint
    run, bucket = tmp_path / 'run', tmp_path / 'bucket'
    run.mkdir()
    for i in range(3):
        torch.save({'iteration': i}, run / f'checkpoint_{i:04d}.pt')
    (run / 'checkpoint_0003.pt').write_bytes(b'truncated')     # a write cut short by a kill
    (run / 'buffer.tmp.npz').write_bytes(b'half written')      # temp files never sync
    gcs.push(str(run), str(bucket))
    assert not (bucket / 'buffer.tmp.npz').exists()
    restored = tmp_path / 'fresh_vm'
    gcs.pull(str(bucket), str(restored))
    assert latest_checkpoint(str(restored)).endswith('checkpoint_0002.pt')
