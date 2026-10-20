"""
Human-play CLI — play Chaturanga against the trained agent.

Terminal ASCII board, no GUI needed.
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chaturanga.board import State, WHITE, BLACK, PIECE_NAMES, sq, rank_of, file_of
from chaturanga.moves import generate_moves, apply_move, encode_action, decode_action
from chaturanga.terminal import is_terminal
from utils.device import get_device, describe_device, add_device_argument


def parse_move(move_str: str) -> int:
    """
    Parse human move input like 'e2e4' into an action.
    """
    move_str = move_str.strip().lower()
    if len(move_str) != 4:
        raise ValueError(f"Move must be 4 characters like 'e2e4', got '{move_str}'")
    
    from_file = ord(move_str[0]) - ord('a')
    from_rank = int(move_str[1]) - 1
    to_file = ord(move_str[2]) - ord('a')
    to_rank = int(move_str[3]) - 1
    
    if not (0 <= from_file < 8 and 0 <= from_rank < 8 and
            0 <= to_file < 8 and 0 <= to_rank < 8):
        raise ValueError(f"Invalid square in '{move_str}'")
    
    from_sq = sq(from_rank, from_file)
    to_sq = sq(to_rank, to_file)
    return encode_action(from_sq, to_sq)


def action_to_str(action: int) -> str:
    """Convert action to human-readable 'e2e4' format."""
    from_sq, to_sq = decode_action(action)
    from_file = chr(ord('a') + file_of(from_sq))
    from_rank = str(rank_of(from_sq) + 1)
    to_file = chr(ord('a') + file_of(to_sq))
    to_rank = str(rank_of(to_sq) + 1)
    return f"{from_file}{from_rank}{to_file}{to_rank}"


BASELINE_NAMES = {
    'minimax': '3-ply Minimax/Alpha-Beta',
    'greedy': 'Greedy Material',
    'random': 'Random',
}


def make_baseline(opponent: str):
    """Return a callable state -> action for a baseline opponent, or None for 'mcts'."""
    from search.baselines import random_player, greedy_material_player, alpha_beta_player
    players = {
        'minimax': lambda state: alpha_beta_player(state, depth=3),
        'greedy': greedy_material_player,
        'random': random_player,
    }
    return players.get(opponent)


def play_cli(checkpoint_path: str = None, human_side: int = WHITE,
             n_sims: int = 100, opponent: str = 'mcts', device=None):
    """
    Play against the agent or baseline in the terminal.
    """
    from search.mcts import search, select_action_with_temperature
    from net.model import create_chaturanga_net, load_net_from_checkpoint
    from net.train import create_evaluator
    
    device = get_device(device)
    
    baseline_player = make_baseline(opponent)
    if baseline_player is not None:
        print(f"Playing against {BASELINE_NAMES[opponent]} baseline!")
    else:
        # Neural MCTS — architecture comes from the checkpoint (local 5x64 or cloud 8x128)
        if checkpoint_path:
            net, _ = load_net_from_checkpoint(checkpoint_path, device=device)
            cfg = net.config
            print(f"Loaded checkpoint: {checkpoint_path} "
                  f"({cfg['n_blocks']} blocks x {cfg['channels']} channels)")
        else:
            net = create_chaturanga_net(n_blocks=5, channels=64)
            net.to(device)
            print("No checkpoint provided — using MCTS with untrained network")
        
        evaluate = create_evaluator(net, device=device, action_size=4096)
    
    state = State()
    
    print("\n" + "=" * 40)
    print("  ChaturangaZero — Human vs AI")
    print("  Enter moves as 'e2e4', 'quit' to exit")
    print("  No check rule — Raja can go anywhere!")
    print("=" * 40)
    
    while True:
        print(f"\n{state.display()}")
        
        done, z = is_terminal(state)
        if done:
            if state.side_to_move == human_side:
                result = z
            else:
                result = -z
            
            if result > 0:
                print("\n🎉 You win!")
            elif result < 0:
                print("\n💀 AI wins!")
            else:
                print("\n🤝 Draw!")
            break
        
        if state.side_to_move == human_side:
            # Human's turn
            legal = generate_moves(state)
            legal_strs = [action_to_str(a) for a in legal]
            
            while True:
                try:
                    move_input = input(f"\nYour move ({len(legal)} legal): ").strip()
                    if move_input.lower() in ('quit', 'q', 'exit'):
                        print("Goodbye!")
                        return
                    if move_input.lower() == 'moves':
                        print("Legal moves:", ' '.join(sorted(legal_strs)))
                        continue
                    
                    action = parse_move(move_input)
                    if action not in legal:
                        print(f"Illegal move! Type 'moves' to see legal moves.")
                        continue
                    break
                except ValueError as e:
                    print(f"Error: {e}")
            
            state = apply_move(state, action)
        else:
            # AI's turn
            print("\nAI is thinking...")
            if baseline_player is not None:
                action = baseline_player(state)
            else:
                visit_counts = search(
                    state=state,
                    evaluate=evaluate,
                    n_sims=n_sims,
                    action_size=4096,
                    c_puct=1.5,
                    add_noise=False,
                )
                action = int(np.argmax(visit_counts))
            print(f"AI plays: {action_to_str(action)}")
            state = apply_move(state, action)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Play Chaturanga against the AI')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--side', type=str, default='white',
                       choices=['white', 'black'],
                       help='Which side to play')
    parser.add_argument('--sims', type=int, default=100,
                       help='MCTS simulations per move')
    parser.add_argument('--opponent', type=str, default='mcts',
                       choices=['mcts', 'minimax', 'greedy', 'random'],
                       help='Opponent type: neural mcts, 3-ply minimax, greedy, or random')
    add_device_argument(parser)
    
    args = parser.parse_args()
    human_side = WHITE if args.side == 'white' else BLACK
    play_cli(args.checkpoint, human_side, args.sims, args.opponent, device=args.device)
