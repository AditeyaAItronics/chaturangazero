"""
Interactive Connect-4 CLI — play against the AlphaZero agent in the terminal.

Usage:
    python eval/play_connect4.py
    python eval/play_connect4.py --model runs/connect4/phase0_passed.pt --sims 50
    python eval/play_connect4.py --ai-first
"""

import sys
import os
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connect4.game import Connect4State, ROWS, COLS
from net.model import create_connect4_net, load_net_from_checkpoint
from net.train import create_evaluator
from search.mcts import search, select_action_with_temperature
from utils.device import get_device, describe_device, add_device_argument


def print_board(state: Connect4State):
    """Render the Connect-4 grid with ASCII characters."""
    symbols = {0: ' . ', 1: ' X ', -1: ' O '}
    print("\n" + "=" * 29)
    print("  0   1   2   3   4   5   6   (Columns)")
    print("+" + "---+" * 7)
    for r in range(ROWS - 1, -1, -1):
        row_str = "|"
        for c in range(COLS):
            val = state.board[r, c]
            row_str += symbols[int(val)] + "|"
        print(row_str)
        if r > 0:
            print("+" + "---+" * 7)
    print("+" + "---+" * 7)
    side_str = "Human (X)" if state.side_to_move == 1 else "AI (O)"
    print(f"Turn: {side_str}")
    print("=" * 29)


def main():
    parser = argparse.ArgumentParser(description="Play Connect-4 against AlphaZero")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model checkpoint")
    parser.add_argument("--sims", type=int, default=50, help="MCTS simulations per move")
    parser.add_argument("--ai-first", action="store_true", help="AI plays first (X)")
    add_device_argument(parser)
    args = parser.parse_args()

    human_side = -1 if args.ai_first else 1
    device = get_device(args.device)
    print(f"Device: {describe_device(device)}")

    # Resolve model path
    model_path = args.model
    if model_path is None:
        candidate_paths = [
            'runs/connect4/phase0_passed.pt',
            'runs/connect4/final.pt',
            'connect4_model.pt',
        ]
        for cp in candidate_paths:
            if os.path.exists(cp):
                model_path = cp
                break

    if model_path and os.path.exists(model_path):
        print(f"Loading trained weights from {model_path}...")
        net, _ = load_net_from_checkpoint(model_path, device=device)
        print("Model loaded successfully!")
    else:
        net = create_connect4_net(n_blocks=3, channels=64).to(device)
        print("No checkpoint found. Playing against an untrained baseline network.")

    net.eval()
    evaluate = create_evaluator(net, device=device, action_size=7)
    state = Connect4State()

    print("\n" + "#" * 40)
    print("      CONNECT-4 vs ALPHAZERO")
    print(f"  You are: {'X (First)' if human_side == 1 else 'O (Second)'}")
    print(f"  AI is:   {'O (Second)' if human_side == 1 else 'X (First)'}")
    print("  Type column number (0-6) or 'q' to quit.")
    print("#" * 40)

    while True:
        done, z = state.is_terminal()
        if done:
            reward = z
            break

        print_board(state)

        if state.side_to_move == human_side:
            # Human turn
            legal = state.legal_moves()
            while True:
                try:
                    user_input = input(f"\nYour move {legal}: ").strip().lower()
                    if user_input in ('q', 'quit', 'exit'):
                        print("Game aborted by user.")
                        return
                    col = int(user_input)
                    if col in legal:
                        state = state.apply(col)
                        break
                    else:
                        print(f"Column {col} is full or invalid! Choose from {legal}.")
                except ValueError:
                    print("Please enter a valid integer column (0-6).")
        else:
            # AI turn
            print(f"\nAI is thinking ({args.sims} MCTS simulations)...")
            visit_dist = search(
                state,
                evaluate,
                n_sims=args.sims,
                action_size=7,
                add_noise=False
            )
            ai_move, _ = select_action_with_temperature(visit_dist, temperature=0.0)
            print(f"AI chose column: {ai_move}")
            state = state.apply(ai_move)

    print_board(state)
    print("\n" + "*" * 40)
    if reward == 0.0:
        print("               IT'S A DRAW!")
    elif (reward == 1.0 and state.side_to_move == human_side) or (reward == -1.0 and state.side_to_move != human_side):
        print("          CONGRATULATIONS, YOU WON!")
    else:
        print("              ALPHAZERO WON!")
    print("*" * 40 + "\n")


if __name__ == '__main__':
    main()
