"""
Main Chaturanga training entry point.

Usage:
    python train_chaturanga.py --config configs/local.yaml --output runs/local
    python train_chaturanga.py --config configs/cloud.yaml --output runs/cloud
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.loop import training_loop, load_config
from utils.device import add_device_argument


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ChaturangaZero Training')
    parser.add_argument('--config', type=str, default='configs/local.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--output', type=str, default='runs/chaturanga',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                       help="Checkpoint to resume from, or 'auto' for the newest valid one")
    parser.add_argument('--sync-to', type=str, default=None,
                       help='gs://bucket/run (or a local folder) to sync the run directory with')
    parser.add_argument('--iterations', type=int, default=None,
                       help='Override number of iterations')
    parser.add_argument('--games', type=int, default=None,
                       help='Override games per iteration')
    parser.add_argument('--no-mirror', action='store_true',
                       help='Disable mirror augmentation (the mirror ablation run)')
    parser.add_argument('--reset-lr', action='store_true',
                       help="On resume, restart the LR at the config's train.lr instead of "
                            "the checkpoint's saved LR (sets train.reset_lr_on_resume)")
    add_device_argument(parser)

    args = parser.parse_args()
    
    print("ChaturangaZero — Self-play AlphaZero for Chaturanga")
    print("=" * 50)
    
    config = load_config(args.config)
    if args.iterations is not None:
        config['loop']['iterations'] = args.iterations
    if args.games is not None:
        config['loop']['games_per_iter'] = args.games
    if args.no_mirror:
        config['train']['mirror_augment'] = False
    if args.reset_lr:
        config['train']['reset_lr_on_resume'] = True
    
    # Print key config
    print(f"Config: {args.config}")
    print(f"Network: {config['net']['blocks']} blocks, {config['net']['channels']} channels")
    print(f"MCTS: {config['mcts']['sims']} sims/move")
    print(f"Loop: {config['loop']['iterations']} iterations, "
          f"{config['loop']['games_per_iter']} games/iter")
    print(f"Output: {args.output}")
    print("=" * 50)
    
    training_loop(config, output_dir=args.output, resume_from=args.resume,
                  device=args.device, sync_to=args.sync_to)
