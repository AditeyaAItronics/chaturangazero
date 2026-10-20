"""
Self-play profiling harness (Phase 4 gate, needtofix I4).

Measures end-to-end self-play throughput for a config on THIS machine —
games/hour, simulations/s, how much time the network takes, mean batch size,
evaluation-cache hit rate — and optionally sweeps the settings that matter
(worker processes, concurrent games, leaves per step). With --price-per-hour
it also reports games per dollar, for the plan's machine-shape comparison
(§2.4: n1-standard-16 vs -32, T4 vs L4): run the same command on each VM.

No training happens: the network is untrained (or loaded from --checkpoint)
and the generated positions are discarded.

Usage:
    python -m pipeline.profile --config configs/local.yaml --games 64
    python -m pipeline.profile --config configs/cloud.yaml --games 256 \\
        --sweep workers=1,4,8 parallel_games=64,128 leaves_per_step=1,4 \\
        --price-per-hour 0.46 --out profile_n1-32.md
"""

import argparse
import copy
import itertools
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np   # noqa: E402
import torch         # noqa: E402

from utils.device import get_device, describe_device, use_amp, add_device_argument   # noqa: E402


def _build(config, checkpoint, device):
    from net.model import create_chaturanga_net, create_connect4_net, load_net_from_checkpoint
    if checkpoint:
        net, _ = load_net_from_checkpoint(checkpoint, device=device)
        return net
    factory = create_connect4_net if config.get('game') == 'connect4' else create_chaturanga_net
    return factory(n_blocks=config['net']['blocks'], channels=config['net']['channels']).to(device)


def profile_once(config: dict, net, games: int, device, settings: dict) -> dict:
    """One measurement with `settings` overriding loop/mcts keys."""
    from pipeline.inference_server import BatchedEvaluator, run_batched_self_play
    from pipeline.workers import run_parallel_self_play
    from net.replay import ReplayBuffer

    game = config.get('game', 'chaturanga')
    if game == 'connect4':
        from connect4.game import Connect4State as state_cls
    else:
        from chaturanga.board import State as state_cls, configure_rules
        configure_rules(move_cap=config['play'].get('move_cap'),
                        adjudicate_margin=config['play'].get('adjudicate_margin'))
    mcts = config['mcts']
    workers = settings.get('workers', config['loop'].get('workers', 1))
    kwargs = dict(
        num_parallel_games=settings.get('parallel_games', config['loop'].get('parallel_games', 32)),
        n_sims=settings.get('sims', mcts['sims']),
        action_size=config['net']['action_space'],
        c_puct=mcts['c_puct'],
        dirichlet_alpha=mcts['dirichlet_alpha'],
        dirichlet_eps=mcts['dirichlet_eps'],
        temp_threshold=config['play']['temp_moves'],
        amp=use_amp(device, config['train'].get('amp_inference', True)),
        reuse_subtree=mcts.get('reuse_subtree', True),
        leaves_per_step=settings.get('leaves_per_step', mcts.get('leaves_per_step', 1)),
        virtual_loss=mcts.get('virtual_loss', 3),
        eval_cache_size=mcts.get('eval_cache_size', 50_000),
    )

    nn = {'seconds': 0.0, 'calls': 0, 'positions': 0}
    original = BatchedEvaluator.evaluate_batch

    def timed(self, states):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t = time.perf_counter()
        out = original(self, states)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        nn['seconds'] += time.perf_counter() - t
        nn['calls'] += 1
        nn['positions'] += len(states)
        return out

    buffer = ReplayBuffer(max_size=games * (config['play'].get('move_cap', 300) + 1))
    start = time.perf_counter()
    if workers > 1:
        stats = run_parallel_self_play(net, game=game, total_games=games, workers=workers,
                                       buffer=buffer, seed=1, device=device, **kwargs)
    else:
        BatchedEvaluator.evaluate_batch = timed
        try:
            stats = run_batched_self_play(net, state_cls, total_games=games, device=device,
                                          buffer=buffer, **kwargs)
        finally:
            BatchedEvaluator.evaluate_batch = original
    wall = time.perf_counter() - start

    plies = sum(stats['game_lengths'])
    row = {
        **{k: settings.get(k, v) for k, v in [('workers', workers),
                                               ('parallel_games', kwargs['num_parallel_games']),
                                               ('leaves_per_step', kwargs['leaves_per_step']),
                                               ('sims', kwargs['n_sims'])]},
        'games': stats['games'],
        'mean_plies': plies / max(stats['games'], 1),
        'games_per_hour': 3600 * stats['games'] / wall,
        'sims_per_second': plies * kwargs['n_sims'] / wall,
        'eval_cache_hit_rate': stats.get('eval_cache_hit_rate', 0.0),
        'seconds': wall,
    }
    if workers <= 1 and nn['calls']:
        row['network_time_share'] = nn['seconds'] / wall
        row['mean_batch'] = nn['positions'] / nn['calls']
    return row


def _parse_sweep(items):
    grid = {}
    for item in items or []:
        key, values = item.split('=')
        grid[key] = [int(v) for v in values.split(',')]
    return grid


def _markdown(rows, price):
    cols = ['workers', 'parallel_games', 'leaves_per_step', 'sims', 'games', 'mean_plies',
            'games_per_hour', 'sims_per_second', 'network_time_share', 'mean_batch',
            'eval_cache_hit_rate']
    if price:
        cols.append('games_per_dollar')
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        if price:
            r['games_per_dollar'] = r['games_per_hour'] / price
        cells = []
        for c in cols:
            v = r.get(c, '')
            if isinstance(v, float):
                v = f"{v:.1%}" if c in ('network_time_share', 'eval_cache_hit_rate') else f"{v:,.1f}"
            cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main():
    from pipeline.loop import load_config
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='configs/local.yaml')
    parser.add_argument('--checkpoint', default=None, help='profile a trained net (default: untrained)')
    parser.add_argument('--games', type=int, default=64)
    parser.add_argument('--sweep', nargs='*', default=None,
                        help='e.g. workers=1,4 parallel_games=32,64 leaves_per_step=1,4 sims=40')
    parser.add_argument('--price-per-hour', type=float, default=None, help='VM $/h, for games per dollar')
    parser.add_argument('--out', default=None, help='also write the table to this markdown file')
    add_device_argument(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(args.device or config.get('device'))
    net = _build(config, args.checkpoint, device)
    print(f"Device: {describe_device(device)} | config {args.config} | "
          f"net {config['net']['blocks']}x{config['net']['channels']} | {args.games} games per setting")

    grid = _parse_sweep(args.sweep)
    keys = list(grid)
    rows = []
    for combo in itertools.product(*[grid[k] for k in keys]) if keys else [()]:
        settings = dict(zip(keys, combo))
        np.random.seed(0)
        row = profile_once(copy.deepcopy(config), net, args.games, device, settings)
        rows.append(row)
        print(f"  {settings or 'config defaults'}: {row['games_per_hour']:,.0f} games/h, "
              f"{row['sims_per_second']:,.0f} sims/s", flush=True)

    table = _markdown(rows, args.price_per_hour)
    print()
    print(table)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(f"Device: {describe_device(device)}, config {args.config}, {args.games} games/setting\n\n")
            f.write(table + '\n')


if __name__ == '__main__':
    main()
