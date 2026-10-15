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


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ChaturangaZero Training')
    parser.add_argument('--config', type=str, default='configs/local.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--output', type=str, default='runs/chaturanga',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--iterations', type=int, default=None,
                       help='Override number of iterations')
    parser.add_argument('--games', type=int, default=None,
                       help='Override games per iteration')
    
    args = parser.parse_args()
    
    print("ChaturangaZero — Self-play AlphaZero for Chaturanga")
    print("=" * 50)
    
    config = load_config(args.config)
    if args.iterations is not None:
        config['loop']['iterations'] = args.iterations
    if args.games is not None:
        config['loop']['games_per_iter'] = args.games
    
    # Print key config
    print(f"Config: {args.config}")
    print(f"Network: {config['net']['blocks']} blocks, {config['net']['channels']} channels")
    print(f"MCTS: {config['mcts']['sims']} sims/move")
    print(f"Loop: {config['loop']['iterations']} iterations, "
          f"{config['loop']['games_per_iter']} games/iter")
    print(f"Output: {args.output}")
    print("=" * 50)
    
    training_loop(config, output_dir=args.output, resume_from=args.resume)
