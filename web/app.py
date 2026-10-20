"""
Flask API backend for the ChaturangaZero & Connect-4 interactive web frontend.

Supports:
- Both Chaturanga (8x8 Ashtapada) and Connect-4 (6x7)
- Player vs AI, AI vs AI, Human vs Human, AI vs Baseline
- Models: AlphaZero (best/latest checkpoints), Alpha-Beta (3-ply), Greedy Material, Random
- Move validation, legal move highlights, evaluation score bar, MCTS candidate move stats
"""

import os
import sys
import glob
import time
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from chaturanga.board import (
    State as ChaturangaState, WHITE, BLACK,
    PADATI, ASHVA, GAJA, RATHA, MANTRI, RAJA,
    rank_of, file_of, sq, PIECE_NAMES, PIECE_VALUES
)
from chaturanga.moves import generate_moves, apply_move, decode_action, encode_action
from chaturanga.terminal import is_terminal as chaturanga_is_terminal
from connect4.game import Connect4State, ROWS as C4_ROWS, COLS as C4_COLS
from net.model import create_chaturanga_net, create_connect4_net
from search.mcts import search
from search.baselines import random_player, greedy_material_player, alpha_beta_player
from utils.device import get_device, describe_device, add_device_argument

app = Flask(__name__, static_folder='static')
device = get_device()  # CHATURANGA_DEVICE or auto-detect; --device overrides in __main__

# Model cache: path -> (model, eval_fn)
_MODEL_CACHE = {}


def get_model(game: str, model_id: str):
    """Retrieve or load model and create evaluator."""
    if game == 'chaturanga':
        checkpoint_path = None
        if model_id == 'az_best':
            checkpoint_path = os.path.join(PROJECT_ROOT, 'runs', 'chaturanga', 'best.pt')
        elif model_id == 'az_latest':
            # Find latest checkpoint_*.pt
            ckpts = sorted(glob.glob(os.path.join(PROJECT_ROOT, 'runs', 'chaturanga', 'checkpoint_*.pt')))
            if ckpts:
                checkpoint_path = ckpts[-1]
            else:
                checkpoint_path = os.path.join(PROJECT_ROOT, 'runs', 'chaturanga', 'best.pt')

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return None, None

        mtime = os.path.getmtime(checkpoint_path)
        cache_key = (checkpoint_path, mtime)
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        try:
            # 5 blocks, 64 channels per local.yaml
            net = create_chaturanga_net(n_blocks=5, channels=64).to(device)
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            net.load_state_dict(ckpt['model_state_dict'])
            net.eval()

            def evaluate(state):
                enc = state.encode()
                x = torch.from_numpy(enc).unsqueeze(0).to(device)
                with torch.inference_mode():
                    logits, v = net(x)
                    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                    val = float(v.item())
                # Canonical (network) frame -> absolute (MCTS) frame
                return state.orient_policy(probs), val

            _MODEL_CACHE[cache_key] = (net, evaluate)
            return net, evaluate
        except Exception as e:
            print(f"Error loading Chaturanga model {checkpoint_path}: {e}")
            return None, None

    elif game == 'connect4':
        checkpoint_path = None
        if model_id in ('az_final', 'az_latest'):
            checkpoint_path = os.path.join(PROJECT_ROOT, 'runs', 'connect4', 'final.pt')
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(PROJECT_ROOT, 'runs', 'connect4', 'phase0_passed.pt')
        elif model_id == 'az_phase0':
            checkpoint_path = os.path.join(PROJECT_ROOT, 'runs', 'connect4', 'phase0_passed.pt')

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return None, None

        mtime = os.path.getmtime(checkpoint_path)
        cache_key = (checkpoint_path, mtime)
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        try:
            # 3 blocks, 64 channels for Connect-4
            net = create_connect4_net(n_blocks=3, channels=64).to(device)
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            net.load_state_dict(ckpt['model_state_dict'])
            net.eval()

            def evaluate(state):
                enc = state.encode()
                x = torch.from_numpy(enc).unsqueeze(0).to(device)
                with torch.inference_mode():
                    logits, v = net(x)
                    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                    val = float(v.item())
                return probs, val

            _MODEL_CACHE[cache_key] = (net, evaluate)
            return net, evaluate
        except Exception as e:
            print(f"Error loading Connect-4 model {checkpoint_path}: {e}")
            return None, None

    return None, None


# Game Session State
class GameSession:
    def __init__(self, game: str = 'chaturanga', p1_type: str = 'human', p2_type: str = 'az_latest'):
        self.game = game
        self.p1_type = p1_type
        self.p2_type = p2_type
        self.history = []  # list of previous states for undo
        self.move_log = [] # list of move notations
        self.reset()

    def reset(self):
        if self.game == 'chaturanga':
            self.state = ChaturangaState()
        else:
            self.state = Connect4State()
        self.history = []
        self.move_log = []

    def current_player_type(self) -> str:
        # P1 is side 1 (White / Red), P2 is side -1 (Black / Yellow)
        return self.p1_type if self.state.side_to_move == 1 else self.p2_type


current_session = GameSession('chaturanga', 'human', 'az_latest')


def serialize_state(session: GameSession) -> dict:
    state = session.state
    game = session.game
    done, z = state.is_terminal()

    # Determine legal moves with full coordinates
    legal_moves_data = []
    if not done:
        for a in state.legal_moves():
            if game == 'chaturanga':
                from_s, to_s = decode_action(a)
                legal_moves_data.append({
                    'action': int(a),
                    'from_sq': int(from_s),
                    'to_sq': int(to_s),
                    'from_rank': int(rank_of(from_s)),
                    'from_file': int(file_of(from_s)),
                    'to_rank': int(rank_of(to_s)),
                    'to_file': int(file_of(to_s)),
                })
            else:
                legal_moves_data.append({
                    'action': int(a),
                    'column': int(a),
                })

    # Terminal reason string
    terminal_reason = None
    if done:
        if game == 'chaturanga':
            from pipeline.selfplay import _classify_terminal
            terminal_reason = _classify_terminal(state, z)
        else:
            terminal_reason = 'four_in_a_row' if z != 0 else 'board_full'

    # Board serialization
    if game == 'chaturanga':
        board_grid = []
        for r in range(8):
            row = []
            for f in range(8):
                sq_idx = sq(r, f)
                piece = int(state.piece_at(sq_idx))
                row.append(piece)
            board_grid.append(row)

        # Captured pieces calculation
        white_pieces = {PADATI: 0, ASHVA: 0, GAJA: 0, RATHA: 0, MANTRI: 0, RAJA: 0}
        black_pieces = {PADATI: 0, ASHVA: 0, GAJA: 0, RATHA: 0, MANTRI: 0, RAJA: 0}
        for s in range(64):
            p = state.piece_at(s)
            if p > 0:
                white_pieces[p] += 1
            elif p < 0:
                black_pieces[-p] += 1

        initial_counts = {PADATI: 8, ASHVA: 2, GAJA: 2, RATHA: 2, MANTRI: 1, RAJA: 1}
        white_captured = []  # Black pieces captured by White
        for p, count in initial_counts.items():
            diff = count - black_pieces[p]
            if diff > 0:
                white_captured.extend([p] * diff)

        black_captured = []  # White pieces captured by Black
        for p, count in initial_counts.items():
            diff = count - white_pieces[p]
            if diff > 0:
                black_captured.extend([p] * diff)

        data = {
            'game': 'chaturanga',
            'board': board_grid,
            'side_to_move': int(state.side_to_move),
            'ply_count': int(state.ply_count),
            'legal_moves': legal_moves_data,
            'is_terminal': bool(done),
            'result': float(z),
            'terminal_reason': terminal_reason,
            'p1_type': session.p1_type,
            'p2_type': session.p2_type,
            'current_player_type': session.current_player_type(),
            'white_material': float(state.material_score(WHITE)),
            'black_material': float(state.material_score(BLACK)),
            'white_captured': white_captured,
            'black_captured': black_captured,
            'move_history': session.move_log,
        }
    else:
        # Connect-4
        board_grid = state.board.tolist()
        data = {
            'game': 'connect4',
            'board': board_grid,
            'side_to_move': int(state.side_to_move),
            'ply_count': int(state.ply_count),
            'legal_moves': legal_moves_data,
            'is_terminal': bool(done),
            'result': float(z),
            'terminal_reason': terminal_reason,
            'p1_type': session.p1_type,
            'p2_type': session.p2_type,
            'current_player_type': session.current_player_type(),
            'move_history': session.move_log,
        }

    return data


# API Routes
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)


@app.route('/api/models', methods=['GET'])
def list_models():
    # Detect available checkpoints
    chaturanga_ckpts = sorted(glob.glob(os.path.join(PROJECT_ROOT, 'runs', 'chaturanga', 'checkpoint_*.pt')))
    latest_chaturanga_iter = len(chaturanga_ckpts) - 1 if chaturanga_ckpts else None

    models_data = {
        'chaturanga': [
            {'id': 'human', 'name': 'Human Player', 'type': 'human'},
            {'id': 'az_latest', 'name': f'AlphaZero Latest (Iter {latest_chaturanga_iter if latest_chaturanga_iter is not None else "N/A"})', 'type': 'az'},
            {'id': 'az_best', 'name': 'AlphaZero Incumbent (best.pt)', 'type': 'az'},
            {'id': 'baseline_alphabeta', 'name': 'Baseline: Alpha-Beta (3-ply search)', 'type': 'baseline'},
            {'id': 'baseline_greedy', 'name': 'Baseline: Greedy Material', 'type': 'baseline'},
            {'id': 'random', 'name': 'Random Moves', 'type': 'random'},
        ],
        'connect4': [
            {'id': 'human', 'name': 'Human Player', 'type': 'human'},
            {'id': 'az_final', 'name': 'AlphaZero Connect-4 (Trained)', 'type': 'az'},
            {'id': 'az_phase0', 'name': 'AlphaZero Connect-4 (Phase 0 Pass)', 'type': 'az'},
            {'id': 'random', 'name': 'Random Moves', 'type': 'random'},
        ]
    }
    return jsonify(models_data)


@app.route('/api/state', methods=['GET'])
def get_state():
    return jsonify(serialize_state(current_session))


@app.route('/api/new_game', methods=['POST'])
def new_game():
    data = request.json or {}
    game = data.get('game', 'chaturanga')
    p1 = data.get('p1_type', 'human')
    p2 = data.get('p2_type', 'az_latest')

    current_session.game = game
    current_session.p1_type = p1
    current_session.p2_type = p2
    current_session.reset()

    return jsonify(serialize_state(current_session))


@app.route('/api/make_move', methods=['POST'])
def make_move():
    data = request.json or {}
    action = data.get('action')
    if action is None:
        return jsonify({'error': 'action required'}), 400

    action = int(action)
    state = current_session.state
    legal = state.legal_moves()
    if action not in legal:
        return jsonify({'error': f'illegal action {action}'}), 400

    # Save to history for undo
    current_session.history.append(state.copy() if hasattr(state, 'copy') else state)

    # Move notation
    if current_session.game == 'chaturanga':
        f_sq, t_sq = decode_action(action)
        piece = state.piece_at(f_sq)
        name = PIECE_NAMES.get(piece, '').upper()
        cols = 'abcdefgh'
        from_str = f"{cols[file_of(f_sq)]}{rank_of(f_sq)+1}"
        to_str = f"{cols[file_of(t_sq)]}{rank_of(t_sq)+1}"
        captured = state.piece_at(t_sq)
        op = 'x' if captured != 0 else '-'
        notation = f"{name}{from_str}{op}{to_str}"
    else:
        notation = f"Col {action + 1}"

    current_session.move_log.append(notation)
    current_session.state = state.apply(action)

    return jsonify(serialize_state(current_session))


@app.route('/api/ai_move', methods=['POST'])
def ai_move():
    state = current_session.state
    done, _ = state.is_terminal()
    if done:
        return jsonify({'error': 'game is already terminal'}), 400

    player_type = current_session.current_player_type()
    if player_type == 'human':
        return jsonify({'error': 'current player is human'}), 400

    action_size = 4096 if current_session.game == 'chaturanga' else 7
    chosen_action = None
    ai_stats = {}

    if player_type in ('az_latest', 'az_best', 'az_final', 'az_phase0'):
        _, evaluate = get_model(current_session.game, player_type)
        if evaluate is not None:
            # Run MCTS search
            n_sims = 200 if current_session.game == 'chaturanga' else 200
            c_puct = 1.5
            vc = search(
                state=state,
                evaluate=evaluate,
                n_sims=n_sims,
                action_size=action_size,
                c_puct=c_puct,
                add_noise=False,
            )
            legal = state.legal_moves()
            legal_vc = {int(a): int(vc[a]) for a in legal}
            # Pick argmax
            chosen_action = max(legal_vc.keys(), key=lambda a: legal_vc[a])

            # Evaluate current position value
            _, root_val = evaluate(state)
            total_visits = sum(legal_vc.values())

            # Top 3 candidate moves
            sorted_candidates = sorted(legal_vc.items(), key=lambda kv: kv[1], reverse=True)[:3]
            top_moves = []
            for a, count in sorted_candidates:
                pct = (count / total_visits * 100) if total_visits > 0 else 0
                if current_session.game == 'chaturanga':
                    f_s, t_s = decode_action(a)
                    cols = 'abcdefgh'
                    notate = f"{cols[file_of(f_s)]}{rank_of(f_s)+1}->{cols[file_of(t_s)]}{rank_of(t_s)+1}"
                else:
                    notate = f"Column {a+1}"
                top_moves.append({'action': a, 'name': notate, 'visits': count, 'pct': round(pct, 1)})

            ai_stats = {
                'value': round(root_val, 3),
                'top_moves': top_moves,
                'sims': n_sims,
            }
        else:
            chosen_action = random_player(state, action_size)
    elif player_type == 'baseline_alphabeta':
        chosen_action = alpha_beta_player(state, depth=3, action_size=action_size)
    elif player_type == 'baseline_greedy':
        chosen_action = greedy_material_player(state, action_size=action_size)
    elif player_type == 'random':
        chosen_action = random_player(state, action_size)
    else:
        chosen_action = random_player(state, action_size)

    if chosen_action is None or chosen_action not in state.legal_moves():
        chosen_action = state.legal_moves()[0]

    # Save to history for undo
    current_session.history.append(state.copy() if hasattr(state, 'copy') else state)

    # Move notation
    if current_session.game == 'chaturanga':
        f_sq, t_sq = decode_action(chosen_action)
        piece = state.piece_at(f_sq)
        name = PIECE_NAMES.get(piece, '').upper()
        cols = 'abcdefgh'
        from_str = f"{cols[file_of(f_sq)]}{rank_of(f_sq)+1}"
        to_str = f"{cols[file_of(t_sq)]}{rank_of(t_sq)+1}"
        captured = state.piece_at(t_sq)
        op = 'x' if captured != 0 else '-'
        notation = f"{name}{from_str}{op}{to_str}"
    else:
        notation = f"Col {chosen_action + 1}"

    current_session.move_log.append(notation)
    current_session.state = state.apply(chosen_action)

    resp = serialize_state(current_session)
    resp['ai_stats'] = ai_stats
    resp['played_action'] = int(chosen_action)
    return jsonify(resp)


@app.route('/api/undo', methods=['POST'])
def undo_move():
    if not current_session.history:
        return jsonify({'error': 'no moves to undo'}), 400

    # Pop last state
    current_session.state = current_session.history.pop()
    if current_session.move_log:
        current_session.move_log.pop()

    # If the user is playing against an AI, pop again so it returns to human's turn
    if current_session.p1_type == 'human' and current_session.p2_type != 'human' and current_session.state.side_to_move == -1 and current_session.history:
        current_session.state = current_session.history.pop()
        if current_session.move_log:
            current_session.move_log.pop()
    elif current_session.p2_type == 'human' and current_session.p1_type != 'human' and current_session.state.side_to_move == 1 and current_session.history:
        current_session.state = current_session.history.pop()
        if current_session.move_log:
            current_session.move_log.pop()

    return jsonify(serialize_state(current_session))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Chaturanga & Connect4 Web Server')
    parser.add_argument('--port', type=int, default=5000, help='Port to serve on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host address')
    add_device_argument(parser)
    args = parser.parse_args()
    if args.device is not None:
        device = get_device(args.device)

    print(f"Starting Game Server on http://{args.host}:{args.port}")
    print(f"Inference device: {describe_device(device)}")
    app.run(host=args.host, port=args.port, debug=False)
