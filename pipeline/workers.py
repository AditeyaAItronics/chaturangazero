"""
Multiprocessing self-play workers (§5 Phase 4, needtofix M3).

The Python MCTS tree work runs on one CPU core per process, so on its own
the batched server cannot use more than one core no matter how fast the GPU
is. This module splits an iteration's games across `workers` processes. Each
process runs its own parallel-games batched server (its own model copy on the
same device) and sends back a compact ReplayBuffer, which the main process
merges into the training buffer.

Processes use the 'spawn' start method, which works the same on Windows,
Linux and macOS and is required for CUDA in child processes.
"""

import multiprocessing as mp
import random
import time
from typing import Optional

import numpy as np
import torch


def _split(total: int, parts: int) -> list[int]:
    """Split `total` games into `parts` near-equal positive shares."""
    parts = max(1, min(parts, total))
    base, extra = divmod(total, parts)
    return [base + (1 if k < extra else 0) for k in range(parts)]


def _worker(task: dict):
    """Run one worker's share of self-play. Top-level so 'spawn' can pickle it."""
    from net.model import DualHeadNet
    from net.replay import ReplayBuffer
    from pipeline.inference_server import run_batched_self_play
    from utils.device import get_device

    device = get_device(task['device'])
    if device.type == 'cpu':
        torch.set_num_threads(1)   # one core per worker; the pool provides the parallelism

    seed = task['seed']
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)

    if task['game'] == 'connect4':
        from connect4.game import Connect4State as state_cls
        max_plies = 42
    else:
        from chaturanga.board import State as state_cls, configure_rules
        configure_rules(**task['rules'])
        max_plies = task['rules']['move_cap']

    net = DualHeadNet(**task['net_config'])
    net.load_state_dict(task['state_dict'])

    buffer = ReplayBuffer(max_size=task['games'] * (max_plies + 1))
    stats = run_batched_self_play(net=net, state_cls=state_cls, total_games=task['games'],
                                  device=device, buffer=buffer, **task['kwargs'])
    return stats, buffer


_SUMMED = ('games', 'p1_wins', 'p2_wins', 'draws', 'samples_generated',
           'raja_capture_chances', 'raja_captures_missed', 'virtual_loss_collisions',
           'eval_cache_hits', 'eval_cache_misses')


def merge_stats(parts: list[dict], wall_seconds: float) -> dict:
    """Combine per-worker self-play statistics into one dict."""
    out = {k: sum(p.get(k, 0) for p in parts) for k in _SUMMED}
    out['game_lengths'] = [n for p in parts for n in p['game_lengths']]
    reasons = {}
    for p in parts:
        for r, n in p['terminal_reasons'].items():
            reasons[r] = reasons.get(r, 0) + n
    out['terminal_reasons'] = reasons
    out['elapsed_seconds'] = wall_seconds
    out['games_per_second'] = out['games'] / max(wall_seconds, 1e-5)
    out['mean_length'] = float(np.mean(out['game_lengths'])) if out['game_lengths'] else 0.0
    out['draw_rate'] = out['draws'] / max(out['games'], 1)
    out['raja_capture_miss_rate'] = (out['raja_captures_missed'] / out['raja_capture_chances']
                                     if out['raja_capture_chances'] else 0.0)
    lookups = out['eval_cache_hits'] + out['eval_cache_misses']
    out['eval_cache_hit_rate'] = out['eval_cache_hits'] / lookups if lookups else 0.0
    out['workers'] = len(parts)
    return out


def run_parallel_self_play(
    net: torch.nn.Module,
    game: str,
    total_games: int,
    workers: int,
    buffer=None,
    seed: int = 0,
    device=None,
    rules: Optional[dict] = None,
    **kwargs,
) -> dict:
    """
    Self-play `total_games` games across `workers` processes.

    Args:
        net: the network to play with (its weights are copied to every worker).
        game: 'chaturanga' or 'connect4'.
        buffer: ReplayBuffer that receives every worker's positions.
        seed: base seed; worker k uses seed + k, so runs are reproducible.
        device: device for the workers ('cuda' shares the one GPU between them).
        rules: {'move_cap': ..., 'adjudicate_margin': ...} for Chaturanga workers.
        **kwargs: forwarded to run_batched_self_play (n_sims, c_puct,
            num_parallel_games, leaves_per_step, ...).

    Returns:
        merged statistics, same keys as run_batched_self_play.
    """
    from chaturanga.board import get_move_cap, get_adjudicate_margin

    if rules is None:
        rules = {'move_cap': get_move_cap(), 'adjudicate_margin': get_adjudicate_margin()}
    state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    shares = _split(total_games, workers)
    tasks = [{
        'game': game, 'games': n, 'seed': seed + k, 'device': str(device) if device else None,
        'rules': rules, 'net_config': net.config, 'state_dict': state_dict, 'kwargs': kwargs,
    } for k, n in enumerate(shares)]

    start = time.time()
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=len(tasks)) as pool:
        results = pool.map(_worker, tasks)
    wall = time.time() - start

    if buffer is not None:
        for _, worker_buffer in results:
            buffer.merge(worker_buffer)
    return merge_stats([stats for stats, _ in results], wall)
