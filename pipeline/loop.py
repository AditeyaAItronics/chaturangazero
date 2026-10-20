"""
Main training loop — orchestrates the AlphaZero self-play → train cycle.

1 iteration = N self-play games → M gradient steps → optional arena gate
"""

import os
import math
import time
import json
import random
import signal
import subprocess
import sys
import yaml
import numpy as np
import torch
import copy
from typing import Optional
from pathlib import Path

from net.model import DualHeadNet, create_chaturanga_net, create_connect4_net
from net.replay import ReplayBuffer
from net.train import Trainer, create_evaluator
from search.arena import gate_model
from search.baselines import random_player
from search.mcts import search, select_action_with_temperature
from pipeline.selfplay import run_self_play_batch
from pipeline.inference_server import run_batched_self_play, run_batched_arena
from utils.device import ENV_VAR, get_device, use_amp, describe_device, add_device_argument


def git_revision() -> dict:
    """Git SHA of the code being run, and whether it has uncommitted changes."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                                    cwd=root, capture_output=True, text=True).stdout.strip())
        return {'git_sha': sha, 'git_dirty': dirty}
    except (OSError, subprocess.CalledProcessError):
        return {'git_sha': None, 'git_dirty': None}


def summarize_sequential_games(games: list[dict], elapsed: float) -> dict:
    """Same statistics dict as run_batched_self_play, for the sequential path."""
    reasons = {}
    for g in games:
        reasons[g['terminal_reason']] = reasons.get(g['terminal_reason'], 0) + 1
    n = max(len(games), 1)
    draws = sum(1 for g in games if g['result'] == 0)
    chances = sum(g.get('raja_capture_chances', 0) for g in games)
    missed = sum(g.get('raja_captures_missed', 0) for g in games)
    last = games[-1] if games else {}
    return {
        'games': len(games),
        'p1_wins': sum(1 for g in games if g['result'] > 0),
        'p2_wins': sum(1 for g in games if g['result'] < 0),
        'draws': draws,
        'draw_rate': draws / n,
        'game_lengths': [g['ply_count'] for g in games],
        'mean_length': sum(g['ply_count'] for g in games) / n,
        'terminal_reasons': reasons,
        'elapsed_seconds': elapsed,
        'games_per_second': len(games) / max(elapsed, 1e-5),
        'raja_capture_chances': chances,
        'raja_captures_missed': missed,
        'raja_capture_miss_rate': missed / chances if chances else 0.0,
        'eval_cache_hit_rate': last.get('eval_cache_hit_rate', 0.0),
    }


def training_steps(new_positions: int, buffer_len: int, batch_size: int, steps_cap: int,
                   replay_ratio: Optional[float] = None, min_buffer: int = 0) -> int:
    """
    Gradient steps for one iteration (needtofixagain N5).

    Fixed `steps_cap` steps (the old behaviour, kept when replay_ratio is None)
    can train each new position many times: 2000 x 512 samples over ~80k new
    positions is a replay ratio of ~13, and the very first iteration would
    fit a random network's games ~13 epochs over. With replay_ratio set, the
    step count follows the data:

        steps = min(steps_cap, ceil(replay_ratio * new_positions / batch_size))

    and no training happens until the buffer holds at least `min_buffer`
    positions (a warm-up, so early iterations don't overfit a tiny buffer).
    """
    if buffer_len < max(int(min_buffer), batch_size):
        return 0
    if replay_ratio is None:
        return int(steps_cap)
    return int(min(steps_cap, math.ceil(replay_ratio * new_positions / batch_size)))


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _write_json_atomic(path: str, data):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _install_preemption_handler(output_dir: str, sync_to: Optional[str]):
    """
    Spot VMs get SIGTERM ~30 s before they disappear (§2.4). Every completed
    iteration is already synced, so the handler only needs to push anything
    not yet uploaded and exit; on restart `--resume auto` continues from the
    last completed iteration (the iteration in progress is redone).
    """
    def on_sigterm(signum, frame):
        print(f"\nSIGTERM received: syncing {output_dir} and exiting for resume.", flush=True)
        try:
            from pipeline import gcs
            gcs.push(output_dir, sync_to)
        finally:
            sys.exit(143)

    try:
        signal.signal(signal.SIGTERM, on_sigterm)
    except (ValueError, OSError):   # not in the main thread / unsupported platform
        pass


def training_loop(config: dict, output_dir: str = 'runs', resume_from: str = None,
                  device=None, sync_to: Optional[str] = None):
    """
    Main AlphaZero training loop.
    
    Args:
        config: configuration dict (from YAML)
        output_dir: directory for checkpoints and logs
        resume_from: path to checkpoint to resume from, or 'auto' for the newest
            valid checkpoint in output_dir (a fresh run if there is none)
        sync_to: 'gs://bucket/run' (or a local folder) — pull the run directory
            from there at start, push after every iteration (Phase 4b)
        device: 'auto', 'cpu', 'cuda', 'cuda:N'; None falls back to CHATURANGA_DEVICE,
                then config['device'], then auto-detect
    """
    # Setup
    game_type = config.get('game', 'chaturanga')
    seed = config.get('seed', 42)
    random.seed(seed)          # replay sampling / any stdlib randomness
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Precedence: --device flag > CHATURANGA_DEVICE > config 'device:' > auto
    device = get_device(device or os.environ.get(ENV_VAR) or config.get('device'))
    print(f"Device: {describe_device(device)}")
    
    # Create output directory, restoring it from the bucket on a fresh VM
    os.makedirs(output_dir, exist_ok=True)
    from pipeline import gcs
    from pipeline.resume import latest_checkpoint, read_run_state, write_run_state
    if sync_to:
        print(f"Restoring run directory from {sync_to} ...", flush=True)
        gcs.pull(sync_to, output_dir)
    if resume_from == 'auto':
        resume_from = latest_checkpoint(output_dir)
        print(f"Auto-resume: {resume_from or 'no checkpoint found, starting fresh'}", flush=True)
    _install_preemption_handler(output_dir, sync_to)
    
    # Save config, with the exact code revision it ran on (Part 6: "every run
    # records its exact config and git SHA")
    revision = git_revision()
    print(f"Code revision: {revision['git_sha']}{' (+ uncommitted changes)' if revision['git_dirty'] else ''}")
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump({**config, 'run_info': revision}, f, default_flow_style=False)
    
    # Game setup
    if game_type == 'connect4':
        from connect4.game import Connect4State
        
        net = create_connect4_net(
            n_blocks=config['net']['blocks'],
            channels=config['net']['channels'],
        )
        action_size = config['net']['action_space']
        create_state = Connect4State
        mirror_perm = np.arange(action_size)[::-1].copy()  # column c -> 6 - c
        board_size = (6, 7)
    else:
        from chaturanga.board import State
        from chaturanga.encoding import MIRROR_PERM
        
        net = create_chaturanga_net(
            n_blocks=config['net']['blocks'],
            channels=config['net']['channels'],
        )
        action_size = config['net']['action_space']
        create_state = State
        mirror_perm = MIRROR_PERM
        board_size = (8, 8)
        
        # §2.2: the YAML move cap / adjudication margin must actually take effect
        from chaturanga.board import configure_rules, get_move_cap, get_adjudicate_margin
        configure_rules(move_cap=config['play'].get('move_cap'),
                        adjudicate_margin=config['play'].get('adjudicate_margin'))
        print(f"Rules: move_cap={get_move_cap()}, adjudicate_margin={get_adjudicate_margin()}")
    
    print(f"Network: {net.count_parameters():,} parameters")
    
    # Create trainer
    trainer = Trainer(
        net=net,
        lr=config['train']['lr'],
        l2=config['train']['l2'],
        lr_milestones=config['train'].get('lr_milestones'),
        lr_gamma=config['train'].get('lr_gamma', 0.1),
        grad_clip=config['train'].get('grad_clip', 1.0),
        device=device,
        optimizer=config['train'].get('optimizer', 'adamw'),
        weight_decay=config['train'].get('weight_decay', 1e-2),
    )
    
    # Create replay buffer — mirror augmentation is applied at sample time
    mirror_augment = config['train'].get('mirror_augment', True)
    buffer = ReplayBuffer(max_size=config['train']['buffer_size'],
                          mirror_perm=mirror_perm if mirror_augment else None)
    
    # Resume from checkpoint if specified
    start_iteration = 0
    best_net_path = None
    
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from {resume_from}")
        checkpoint = trainer.load_checkpoint(resume_from)
        start_iteration = checkpoint.get('iteration', 0) + 1
        # Fresh random streams for the resumed part, so self-play noise and
        # sampling don't replay the sequences used from iteration 0
        random.seed(seed + 104729 * start_iteration)
        np.random.seed((seed + 104729 * start_iteration) % (2 ** 32))
        torch.manual_seed(seed + 104729 * start_iteration)
        
        # Load buffer if available
        buffer_path = os.path.join(output_dir, 'buffer.npz')
        if os.path.exists(buffer_path):
            buffer.load(buffer_path)
            print(f"Loaded buffer with {len(buffer)} positions")
            # N2: a snapshot written by an iteration that was then cut short is
            # ahead of the last completed iteration. That iteration is redone,
            # so drop its positions instead of adding them twice.
            run_state = read_run_state(output_dir)
            if (run_state is not None
                    and run_state.get('last_completed_iteration') == start_iteration - 1
                    and 'buffer_total_added' in run_state):
                excess = buffer.total_added - int(run_state['buffer_total_added'])
                if excess > 0:
                    buffer.drop_newest(excess)
                    print(f"Dropped {excess} positions from the interrupted iteration "
                          f"{start_iteration}; buffer now {len(buffer)}")
    
    # Training config
    n_iterations = config['loop']['iterations']
    games_per_iter = config['loop']['games_per_iter']
    steps_per_iter = config['train']['steps_per_iter']     # upper bound when replay_ratio is set
    replay_ratio = config['train'].get('replay_ratio')       # None: always steps_per_iter (old behaviour)
    min_buffer = config['train'].get('min_buffer', 0)
    batch_size = config['train']['batch_size']
    n_sims = config['mcts']['sims']
    c_puct = config['mcts']['c_puct']
    temp_moves = config['play']['temp_moves']
    move_cap = config['play'].get('move_cap', 300)
    
    # Arena config
    arena_games = config['arena']['games']
    arena_every = config['arena']['every_n_iters']
    win_threshold = config['arena']['win_threshold']
    
    opening_plies = config['arena'].get('opening_plies', 4)
    
    # Training metrics log — on resume, keep the history up to the resume point
    metrics_log = []
    metrics_path = os.path.join(output_dir, 'metrics.json')
    if start_iteration > 0 and os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics_log = [m for m in json.load(f) if m.get('iteration', -1) < start_iteration]
        print(f"Loaded {len(metrics_log)} previous metrics entries")
    
    print(f"\n{'='*60}")
    print(f"Starting training: {n_iterations} iterations")
    print(f"  Games/iter: {games_per_iter}, Sims/move: {n_sims}")
    if replay_ratio is None:
        print(f"  Steps/iter: {steps_per_iter}, Batch: {batch_size}")
    else:
        print(f"  Steps/iter: replay ratio {replay_ratio} (at most {steps_per_iter}), "
              f"Batch: {batch_size}, training starts at {min_buffer:,} positions")
    print(f"  Buffer: {config['train']['buffer_size']:,}")
    print(f"{'='*60}\n")
    
    # Incumbent ("best") network: generates self-play data and defends in the arena.
    # `net` is the challenger that the trainer updates; it only replaces the
    # incumbent after winning the arena gate.
    best_net_path = os.path.join(output_dir, 'best.pt')
    best_net = copy.deepcopy(net).to(device)
    if start_iteration > 0 and os.path.exists(best_net_path):
        # Resuming: keep the existing incumbent instead of overwriting it
        best_checkpoint = torch.load(best_net_path, map_location=device, weights_only=False)
        best_net.load_state_dict(best_checkpoint['model_state_dict'])
        print(f"Loaded incumbent from {best_net_path}")
    else:
        trainer.save_checkpoint(best_net_path, iteration=0)
    best_net.eval()
    
    # Positions already paid for by training (N5), restored from the commit
    # marker on resume so warm-up games played before a restart still count
    trained_upto = buffer.total_added
    run_state = read_run_state(output_dir) if start_iteration > 0 else None
    if run_state is not None and run_state.get('last_completed_iteration') == start_iteration - 1:
        trained_upto = min(buffer.total_added, int(run_state.get('trained_upto', trained_upto)))
    trained_since_gate = False
    for iteration in range(start_iteration, n_iterations):
        iter_start = time.time()
        print(f"\n{'='*40}\n>>> Iteration {iteration+1}/{n_iterations} <<<\n{'='*40}", flush=True)
        
        # 1. Self-play
        parallel_games = config['loop'].get('parallel_games', 1)
        workers = config['loop'].get('workers', 1)
        amp = use_amp(device, config['train'].get('amp_inference', True))
        mcts_cfg = config['mcts']
        batched_kwargs = dict(
            num_parallel_games=parallel_games,
            n_sims=n_sims,
            action_size=action_size,
            c_puct=c_puct,
            dirichlet_alpha=mcts_cfg['dirichlet_alpha'],
            dirichlet_eps=mcts_cfg['dirichlet_eps'],
            temp_threshold=temp_moves,
            amp=amp,
            reuse_subtree=mcts_cfg.get('reuse_subtree', True),
            leaves_per_step=mcts_cfg.get('leaves_per_step', 1),
            virtual_loss=mcts_cfg.get('virtual_loss', 3),
            eval_cache_size=mcts_cfg.get('eval_cache_size', 50_000),
        )

        if parallel_games > 1 and workers > 1:
            print(f"Parallel self-play: {games_per_iter} games over {workers} worker processes, "
                  f"{parallel_games} concurrent each, {n_sims} sims/move (AMP={amp})...", flush=True)
            from pipeline.workers import run_parallel_self_play
            sp_stats = run_parallel_self_play(
                net=best_net, game=game_type, total_games=games_per_iter, workers=workers,
                buffer=buffer, seed=seed + 7919 * (iteration + 1), device=device,
                **batched_kwargs,
            )
        elif parallel_games > 1:
            print(f"Batched self-play: {games_per_iter} games, {parallel_games} concurrent, {n_sims} sims/move (AMP={amp})...", flush=True)
            sp_stats = run_batched_self_play(
                net=best_net,
                state_cls=create_state,
                total_games=games_per_iter,
                device=device,
                buffer=buffer,
                **batched_kwargs,
            )
        else:
            print(f"Sequential self-play: {games_per_iter} games, {n_sims} sims/move...", flush=True)
            evaluate = create_evaluator(best_net, device=device, action_size=action_size)
            sp_start = time.time()

            games = run_self_play_batch(
                n_games=games_per_iter,
                create_state=create_state,
                evaluate=evaluate,
                action_size=action_size,
                n_sims=n_sims,
                c_puct=c_puct,
                temp_moves=temp_moves,
                move_cap=move_cap,
                dirichlet_alpha=mcts_cfg['dirichlet_alpha'],
                dirichlet_eps=mcts_cfg['dirichlet_eps'],
                verbose=True,
                reuse_subtree=mcts_cfg.get('reuse_subtree', True),
                eval_cache_size=mcts_cfg.get('eval_cache_size', 50_000),
            )

            # Add games to buffer
            for game in games:
                buffer.add_game(
                    states=game['states'],
                    policies=game['policies'],
                    result=game['result'],
                    legal=game['legal'],
                )
            sp_stats = summarize_sequential_games(games, time.time() - sp_start)

        avg_plies = sp_stats['mean_length']
        terminal_stats = sp_stats['terminal_reasons']
        if game_type != 'connect4':
            # Always report every terminal rule, so e.g. a zero stalemate count is visible
            for reason in ('raja_captured', 'king_vs_king', 'bare_king', 'stalemate', 'move_cap'):
                terminal_stats.setdefault(reason, 0)
        cap_rate = terminal_stats.get('move_cap', 0) / max(sp_stats['games'], 1)
        selfplay_diagnostics = {
            'games_per_hour': 3600 * sp_stats['games_per_second'],
            'draw_rate': sp_stats.get('draw_rate', 0.0),
            'raja_capture_chances': sp_stats.get('raja_capture_chances', 0),
            'raja_captures_missed': sp_stats.get('raja_captures_missed', 0),
            'raja_capture_miss_rate': sp_stats.get('raja_capture_miss_rate', 0.0),
            'eval_cache_hit_rate': sp_stats.get('eval_cache_hit_rate', 0.0),
            # Batched-server search accounting (Phase 4 tuning of leaves_per_step / virtual_loss)
            'virtual_loss_collisions': sp_stats.get('virtual_loss_collisions', 0),
            'simulations_per_move': (sp_stats['simulations'] / sp_stats['moves']
                                     if sp_stats.get('moves') else None),
        }
        print(f"  Self-play finished: {sp_stats['p1_wins']}W-{sp_stats['draws']}D-{sp_stats['p2_wins']}L "
              f"in {sp_stats['elapsed_seconds']:.1f}s ({selfplay_diagnostics['games_per_hour']:.0f} games/h) | "
              f"draw rate {selfplay_diagnostics['draw_rate']:.1%} | "
              f"missed Raja captures {selfplay_diagnostics['raja_captures_missed']}/"
              f"{selfplay_diagnostics['raja_capture_chances']} | "
              f"cache hit rate {selfplay_diagnostics['eval_cache_hit_rate']:.1%}", flush=True)

        print(f"  Buffer: {len(buffer):,} positions | Avg game length: {avg_plies:.1f} plies | Cap rate: {cap_rate:.1%}", flush=True)
        print(f"  Terminal reasons: {terminal_stats}", flush=True)

        # 2. Training — step count from the replay ratio (N5). "New" means not
        # yet trained on, so games from warm-up iterations count once training starts.
        new_positions = buffer.total_added - trained_upto
        steps = training_steps(new_positions, len(buffer), batch_size, steps_per_iter,
                               replay_ratio, min_buffer)
        train_stats = {}
        if steps > 0:
            print(f"Training: {steps} steps (batch_size={batch_size}, "
                  f"{steps * batch_size / max(new_positions, 1):.1f} samples per new position)...",
                  flush=True)
            train_stats = trainer.train_epoch(
                buffer=buffer,
                steps=steps,
                batch_size=batch_size,
            )
            trained_since_gate = True
            if replay_ratio is None:
                trained_upto = buffer.total_added
            else:
                # Advance only by what these steps paid for: if steps_per_iter
                # capped them, the rest carries over to the next iteration
                paid = int(steps * batch_size / replay_ratio)
                trained_upto = min(buffer.total_added, trained_upto + paid)
        else:
            print(f"Training skipped: buffer {len(buffer):,} < warm-up {max(min_buffer, batch_size):,} positions",
                  flush=True)
        training_diagnostics = {
            'new_positions': new_positions,
            'train_steps': steps,
            'samples_per_new_position': steps * batch_size / new_positions if new_positions else 0.0,
        }
        
        if train_stats:
            print(f"  Loss summary: Policy={train_stats['mean_policy_loss']:.4f} | Value={train_stats['mean_value_loss']:.4f} | Entropy={train_stats['mean_entropy']:.4f} | LR={trainer.optimizer.param_groups[0]['lr']:.2e}", flush=True)
            print(f"  Value head: calibration error={train_stats['value_calibration_error']:.3f} | "
                  f"sign accuracy={train_stats['value_sign_accuracy']:.1%} | mean |v|={train_stats['value_mean_abs']:.3f}", flush=True)
        
        # Step the LR scheduler only when training happened, so lr_milestones
        # count training iterations (warm-up iterations don't advance them)
        if train_stats:
            trainer.step_scheduler()
        
        # 3. Save checkpoint
        ckpt_path = os.path.join(output_dir, f'checkpoint_{iteration:04d}.pt')
        trainer.save_checkpoint(ckpt_path, iteration=iteration, extra={
            **revision,
            'terminal_stats': terminal_stats,
            'avg_plies': avg_plies,
            'cap_rate': cap_rate,
        })
        print(f"  Saved checkpoint to {ckpt_path}", flush=True)
        
        # Save buffer snapshot (default: every iteration — the compact buffer makes
        # this cheap, and a resume then loses no self-play data). Written to a
        # temp file and renamed, so a preemption mid-write can't corrupt it.
        if (iteration + 1) % config.get('cloud', {}).get('buffer_snapshot_every', 1) == 0:
            buf_path = os.path.join(output_dir, 'buffer.npz')
            tmp_path = os.path.join(output_dir, 'buffer.tmp.npz')
            buffer.save(tmp_path)
            os.replace(tmp_path, buf_path)
            print(f"  Snapshot saved to {buf_path}", flush=True)
        
        # 4. Arena gating
        arena_record = None
        gate_due = (iteration + 1) % arena_every == 0 and best_net_path
        if gate_due and not trained_since_gate:
            print("Arena skipped: the challenger hasn't trained since the last gate", flush=True)
        elif gate_due:
            trained_since_gate = False
            print(f"Arena Gating: {arena_games} games vs best incumbent...", flush=True)
            
            # All arena games run concurrently with batched evaluation (S6);
            # same rules as gate_model(): shared random openings, argmax, no adjudication
            arena_start = time.time()
            arena_stats = run_batched_arena(
                net_a=net, net_b=best_net, create_state=create_state,
                n_games=arena_games, n_sims=max(1, n_sims // 2),
                action_size=action_size, c_puct=c_puct, device=device,
                amp=use_amp(device, config['train'].get('amp_inference', True)),
                opening_plies=opening_plies,
                eval_cache_size=config['mcts'].get('eval_cache_size', 50_000),
            )
            promoted = arena_stats['p1_win_rate'] >= win_threshold
            print(f"  Arena took {time.time() - arena_start:.1f}s", flush=True)
            
            status_str = "PROMOTED (New best!)" if promoted else "REJECTED (Incumbent retained)"
            print(f"  Arena result: Challenger {arena_stats['p1_wins']}W-{arena_stats['draws']}D-{arena_stats['p2_wins']}L (Score: {arena_stats['p1_win_rate']:.1%}) -> {status_str}", flush=True)
            
            if promoted:
                best_net.load_state_dict(net.state_dict())
                best_net.eval()
                # Written before the commit marker: if the run dies between here and
                # the marker, the redone iteration keeps this promoted incumbent. It
                # did win its gate, so that is kept on purpose (N2); its checkpoint
                # records the iteration it came from.
                trainer.save_checkpoint(best_net_path, iteration=iteration)
            arena_record = {
                'games': arena_games,
                'wins': arena_stats['p1_wins'], 'draws': arena_stats['draws'],
                'losses': arena_stats['p2_wins'], 'score': arena_stats['p1_win_rate'],
                'promoted': bool(promoted),
            }
        
        # Log metrics
        iter_elapsed = time.time() - iter_start
        metrics = {
            'iteration': iteration,
            'avg_plies': avg_plies,
            'cap_rate': cap_rate,
            'terminal_stats': terminal_stats,
            'buffer_size': len(buffer),
            'buffer_age_histogram': buffer.age_histogram(),
            'elapsed_seconds': iter_elapsed,
            **selfplay_diagnostics,
            **training_diagnostics,
        }
        if train_stats:
            metrics.update(train_stats)
        if arena_record is not None:
            metrics['arena'] = arena_record
        metrics_log.append(metrics)
        
        # Save metrics, then the commit marker LAST (N2): only an iteration
        # whose checkpoint, buffer, arena gate and metrics are all written
        # counts as completed, so a kill anywhere before this line makes
        # resume redo the whole iteration instead of skipping its gate.
        # Then sync the run directory, so a lost VM costs at most this iteration.
        _write_json_atomic(metrics_path, metrics_log)
        write_run_state(output_dir, {
            'last_completed_iteration': iteration,
            'buffer_total_added': buffer.total_added,
            'trained_upto': trained_upto,
            'checkpoint': os.path.basename(ckpt_path),
        })
        if sync_to:
            gcs.push(output_dir, sync_to)
                
        print(f"  Iteration completed in {iter_elapsed:.1f}s. Metrics logged to {metrics_path}", flush=True)
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ChaturangaZero Training')
    parser.add_argument('--config', type=str, default='configs/local.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--output', type=str, default='runs/default',
                       help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help="Checkpoint to resume from, or 'auto' for the newest valid one")
    parser.add_argument('--sync-to', type=str, default=None,
                       help='gs://bucket/run (or a local folder) to sync the run directory with')
    add_device_argument(parser)

    args = parser.parse_args()
    config = load_config(args.config)
    training_loop(config, output_dir=args.output, resume_from=args.resume,
                  device=args.device, sync_to=args.sync_to)
