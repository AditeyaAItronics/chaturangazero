"""
Preemption soak test (Phase 4b gate, needtofix I3).

Plan §5 Phase 4b: "deliberately kill the VM mid-iteration three times and
confirm clean resume each time." This reproduces that on one machine:

  * a small Connect-4 training run syncs its run directory to a "bucket"
    (a local folder here; pass --bucket gs://... to use Cloud Storage),
  * the training process is killed at a random moment, and its working
    directory is DELETED, as if the VM vanished,
  * it is restarted with `--resume auto --sync-to <bucket>`, exactly as
    infra/startup.sh does on a fresh VM,
  * after the kills, the run is allowed to finish, then the bucket is checked:
    every iteration present exactly once and in order, checkpoints, best.pt and
    the buffer snapshot all load.

Usage:
    python tools/soak_test.py                 # 3 hard kills (works on Windows)
    python tools/soak_test.py --signal term   # SIGTERM, like spot preemption (Linux/macOS)
"""

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = """
game: connect4
seed: 7
device: cpu
net: {{blocks: 1, channels: 8, action_space: 7, input_planes: 3}}
mcts: {{sims: 24, c_puct: 1.5, dirichlet_alpha: 1.0, dirichlet_eps: 0.25}}
play: {{temp_moves: 4, move_cap: 42}}
train: {{lr: 1.0e-3, l2: 1.0e-4, batch_size: 32, buffer_size: 20000, steps_per_iter: 30,
         grad_clip: 1.0, amp_inference: false, mirror_augment: true}}
arena: {{games: 2, win_threshold: 0.55, every_n_iters: 2, opening_plies: 2}}
loop: {{games_per_iter: 24, iterations: {iterations}, parallel_games: 12, workers: 1}}
"""


def _start(config_path, work, bucket, log):
    cmd = [sys.executable, '-m', 'pipeline.loop', '--config', config_path, '--output', work,
           '--resume', 'auto', '--sync-to', bucket, '--device', 'cpu']
    return subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)


def _completed_iterations(bucket):
    path = os.path.join(bucket, 'metrics.json')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [m['iteration'] for m in json.load(f)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kills', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=8)
    parser.add_argument('--signal', choices=['kill', 'term'], default='kill')
    parser.add_argument('--bucket', default=None, help='default: a temporary local folder')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    tmp = tempfile.mkdtemp(prefix='cz_soak_')
    bucket = args.bucket or os.path.join(tmp, 'bucket')
    work = os.path.join(tmp, 'vm', 'run')
    config_path = os.path.join(tmp, 'soak.yaml')
    with open(config_path, 'w') as f:
        f.write(CONFIG.format(iterations=args.iterations))
    log_path = os.path.join(tmp, 'train.log')
    print(f"Soak test in {tmp} (bucket: {bucket})")

    kills = 0
    with open(log_path, 'w') as log:
        while True:
            proc = _start(config_path, work, bucket, log)
            if kills >= args.kills:
                proc.wait(timeout=1800)
                break
            # Wait for one completed iteration (timing how long the next one takes
            # to start), then kill at a random moment inside the following iteration
            before = len(_completed_iterations(bucket))
            deadline = time.time() + 600
            while len(_completed_iterations(bucket)) == before and proc.poll() is None:
                if time.time() > deadline:
                    proc.kill()
                    sys.exit("FAIL: no iteration completed within 10 minutes")
                time.sleep(0.05)
            t0 = time.time()
            while len(_completed_iterations(bucket)) == before + 1 and proc.poll() is None:
                time.sleep(0.05)
            iteration_seconds = time.time() - t0
            # we are now just after a completed iteration: kill partway into the next one
            time.sleep(rng.uniform(0.2, 0.7) * iteration_seconds)
            if proc.poll() is not None:
                break                                   # finished before we could kill it
            if args.signal == 'term' and hasattr(signal, 'SIGTERM') and os.name != 'nt':
                proc.send_signal(signal.SIGTERM)
            else:
                proc.kill()
            proc.wait(timeout=120)
            kills += 1
            done = _completed_iterations(bucket)
            print(f"  kill {kills}: process stopped after iteration {done[-1] if done else None}; "
                  f"deleting the VM's working directory")
            shutil.rmtree(os.path.dirname(work), ignore_errors=True)

    # ---- verify the bucket ----
    import torch
    sys.path.insert(0, ROOT)
    from net.replay import ReplayBuffer

    iterations = _completed_iterations(bucket)
    expected = list(range(args.iterations))
    ok = iterations == expected
    print(f"Iterations in metrics.json: {iterations}  ({'OK' if ok else 'expected ' + str(expected)})")
    for name in [f'checkpoint_{args.iterations - 1:04d}.pt', 'best.pt']:
        ckpt = torch.load(os.path.join(bucket, name), map_location='cpu', weights_only=False)
        print(f"{name}: loads, iteration {ckpt.get('iteration')}")
    buf = ReplayBuffer(max_size=20000)
    buf.load(os.path.join(bucket, 'buffer.npz'))
    print(f"buffer.npz: loads, {len(buf)} positions")
    with open(log_path) as f:
        resumes = sum('Auto-resume: ' in line and 'starting fresh' not in line for line in f)
    print(f"Kills: {kills}, automatic resumes: {resumes}")
    passed = ok and kills == args.kills and resumes == args.kills and len(buf) > 0
    print("SOAK TEST PASSED" if passed else f"SOAK TEST FAILED (log: {log_path})")
    if passed and not args.bucket:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
