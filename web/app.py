"""
Flask API backend for the ChaturangaZero & Connect-4 interactive web frontend.

Supports:
- Both Chaturanga (8x8 Ashtapada) and Connect-4 (6x7)
- Player vs AI, AI vs AI, Human vs Human, AI vs Baseline
- Models: AlphaZero (best/latest checkpoints), Alpha-Beta (3-ply), Greedy Material, Random
- Move validation, legal move highlights, evaluation score bar, MCTS candidate move stats
"""

import json
import os
import re
import sys
import time
import threading
from functools import wraps
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
from net.model import create_chaturanga_net, create_connect4_net, load_net_from_checkpoint
from search.mcts import search
from search.baselines import random_player, greedy_material_player, alpha_beta_player
from utils.device import get_device, describe_device, add_device_argument

app = Flask(__name__, static_folder='static')
device = get_device()  # CHATURANGA_DEVICE or auto-detect; --device overrides in __main__

# Model cache: path -> (model, eval_fn)
_MODEL_CACHE = {}


def _latest_chaturanga_checkpoint():
    """(iteration, path) of the newest checkpoint_*.pt, or (None, None)."""
    from pipeline.resume import list_checkpoints
    ckpts = list_checkpoints(os.path.join(PROJECT_ROOT, 'runs', 'chaturanga'))
    return ckpts[-1] if ckpts else (None, None)


def _checkpoint_path(game: str, model_id: str):
    """The checkpoint file an AlphaZero player id refers to, or None if it doesn't exist."""
    path = None
    if game == 'chaturanga':
        if model_id == 'az_best':
            path = os.path.join(PROJECT_ROOT, 'runs', 'chaturanga', 'best.pt')
        elif model_id == 'az_latest':
            path = (_latest_chaturanga_checkpoint()[1]
                    or os.path.join(PROJECT_ROOT, 'runs', 'chaturanga', 'best.pt'))
        elif model_id.startswith('ckpt_') and model_id[5:].isdigit():
            # A specific checkpoint_NNNN.pt (digits only, so no path tricks)
            path = os.path.join(PROJECT_ROOT, 'runs', 'chaturanga',
                                f'checkpoint_{int(model_id[5:]):04d}.pt')
    elif game == 'connect4':
        if model_id in ('az_final', 'az_latest'):
            path = os.path.join(PROJECT_ROOT, 'runs', 'connect4', 'final.pt')
            if not os.path.exists(path):
                path = os.path.join(PROJECT_ROOT, 'runs', 'connect4', 'phase0_passed.pt')
        elif model_id == 'az_phase0':
            path = os.path.join(PROJECT_ROOT, 'runs', 'connect4', 'phase0_passed.pt')
    return path if path and os.path.exists(path) else None


PLAY_SIMS = 200          # default MCTS simulations per AI move in this app
MIN_SIMS, MAX_SIMS = 10, 800   # range of the strength slider
_INFO_CACHE = {}
_UI_PIECE_NAMES = {1: 'Padati', 2: 'Ashva', 3: 'Gaja', 4: 'Ratha', 5: 'Mantri', 6: 'Raja'}


def _iteration_of(path: str, ckpt: dict):
    """1-based iteration a checkpoint was saved after: its stored 'iteration', else
    the number in a checkpoint_NNNN.pt filename (N15: never a count of files)."""
    if isinstance(ckpt, dict) and 'iteration' in ckpt:
        return int(ckpt['iteration']) + 1
    match = re.search(r'checkpoint_(\d+)\.pt$', os.path.basename(path))
    return int(match.group(1)) + 1 if match else None


def _move_name(state, action: int, game: str) -> str:
    """'Ashva g8 → f6' / 'Raja e2 × e3' for Chaturanga, 'Column 4' for Connect-4."""
    if game != 'chaturanga':
        return f"Column {action + 1}"
    f_s, t_s = decode_action(action)
    cols = 'abcdefgh'
    capture = '×' if state.piece_at(t_s) != 0 else '→'
    return (f"{_UI_PIECE_NAMES.get(abs(int(state.piece_at(f_s))), '')} "
            f"{cols[file_of(f_s)]}{rank_of(f_s)+1} {capture} "
            f"{cols[file_of(t_s)]}{rank_of(t_s)+1}")


ANALYSIS_SIMS = 200      # simulations per position when analysing a game
# Expected-score points a move may cost before it gets a label (0-100 scale)
BLUNDER, MISTAKE, INACCURACY = 20.0, 10.0, 5.0
BRILLIANT_MAX_PRIOR = 0.05   # the network gave the move under 5% on instinct...
BRILLIANT_MIN_GAIN = 10.0    # ...and its own favourite was this many points worse
ONLY_MOVE_GAP = 10.0         # "!": every reasonable alternative is this much worse
DECIDED = 0.9                # |value| above this: the game is decided, no praise


def _search_summary(state, evaluate, sims: int) -> dict:
    """Search one position; values are from the side to move's view (-1..1)."""
    done, z = state.is_terminal(adjudicate=False)
    if done:
        return {'terminal': True, 'value': float(z)}
    _, root = search(state=state, evaluate=evaluate, n_sims=sims, action_size=4096,
                     c_puct=1.5, add_noise=False, adjudicate=False, return_root=True)
    acts = [int(a) for a in root.legal_actions]
    visits = {a: int(n) for a, n in zip(acts, root.N)}
    q = {a: float(w / n) for a, w, n in zip(acts, root.W, root.N) if n > 0}
    prior = {a: float(pr) for a, pr in zip(acts, root.P)}
    best = max(acts, key=lambda a: visits[a])
    return {'terminal': False, 'value': q.get(best, 0.0), 'best': best,
            'q': q, 'visits': visits, 'prior': prior}


def _raja_captures(state, raja_code: int) -> list:
    """Legal moves of the side to move that capture the piece `raja_code`."""
    return [a for a in state.legal_moves()
            if state.piece_at(decode_action(a)[1]) == raja_code]


def classify_move(before, after, action: int, before_info: dict, after_info: dict) -> dict:
    """
    Label one move from searches of the positions before and after it.
    Exact checks first (no check rule, so these decide games): a missed Raja
    capture and a Raja left capturable are always blunders. Otherwise the
    label comes from how many expected-score points the move cost the mover.
    """
    mover = before.side_to_move
    v_best = before_info['value']                 # mover's value with best play
    v_after = -after_info['value']                # mover's value after this move
    loss = max(0.0, v_best - v_after) * 50.0      # expected-score points (0-100)
    played_best = action == before_info.get('best')
    wins_now = _raja_captures(before, -6 * mover)
    result = {'loss': round(loss, 1), 'label': None, 'reason': None,
              'white_before': round((v_best * mover + 1) / 2, 4),
              'white_after': round((v_after * mover + 1) / 2, 4)}
    if not played_best and before_info.get('best') is not None:
        result['better'] = _move_name(before, before_info['best'], 'chaturanga')

    if wins_now and action not in wins_now:
        result.update(label='??', reason='Missed a Raja capture that wins on the spot')
        return result
    if not after_info['terminal'] and _raja_captures(after, 6 * mover):
        result.update(label='??', reason='Leaves the Raja capturable: the opponent can win next move')
        return result
    if action in wins_now:
        return result                              # the winning capture itself

    if not played_best:
        # Search noise can't make the best move look bad, so only non-best moves lose
        for cut, label, word in ((BLUNDER, '??', 'Blunder'), (MISTAKE, '?', 'Mistake'),
                                 (INACCURACY, '?!', 'Inaccuracy')):
            if loss >= cut:
                result.update(label=label, reason=f'{word}: costs {loss:.0f} points of expected score')
                return result
        return result

    q, prior = before_info.get('q') or {}, before_info.get('prior') or {}
    if abs(v_best) > DECIDED or not prior:
        return result                              # decided game, or nothing to compare
    instinct = max(prior, key=prior.get)
    if (prior.get(action, 0.0) < BRILLIANT_MAX_PRIOR and instinct != action
            and instinct in q and (v_best - q[instinct]) * 50.0 >= BRILLIANT_MIN_GAIN):
        result.update(label='!!', reason=(
            f"Brilliant: the network gave it only {prior[action] * 100:.0f}% on instinct; "
            f"its favourite ({_move_name(before, instinct, 'chaturanga')}) was "
            f"{(v_best - q[instinct]) * 50:.0f} points worse"))
        return result
    total = sum(before_info['visits'].values()) or 1
    rivals = [q[a] for a, n in before_info['visits'].items()
              if a != action and a in q and n >= 0.05 * total]
    if rivals and (v_best - max(rivals)) * 50.0 >= ONLY_MOVE_GAP:
        result.update(label='!', reason=(
            f"Good: the only strong move; the next best was "
            f"{(v_best - max(rivals)) * 50:.0f} points worse"))
    return result


def _checkpoint_info(path: str) -> dict:
    """Iteration, architecture and training size of a checkpoint (cached by mtime)."""
    key = (path, os.path.getmtime(path))
    if key not in _INFO_CACHE:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        config = ckpt.get('net_config') or {}
        _INFO_CACHE[key] = {
            'iteration': _iteration_of(path, ckpt),
            'blocks': config.get('n_blocks'),
            'channels': config.get('channels'),
            'parameters': int(sum(v.numel() for k, v in state_dict.items()
                                  if hasattr(v, 'numel') and 'running_' not in k
                                  and 'num_batches' not in k)),
            'train_steps': ckpt.get('total_steps'),
        }
    return _INFO_CACHE[key]


def _run_summary(run_dir: str) -> dict:
    """Training totals from the run's metrics.json, if it is there."""
    path = os.path.join(run_dir, 'metrics.json')
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            metrics = json.load(f)
    except (OSError, ValueError):
        return {}
    arenas = [m['arena'] for m in metrics if m.get('arena')]
    return {
        'iterations_trained': len(metrics),
        'selfplay_games': int(sum(sum(m.get('terminal_stats', {}).values()) for m in metrics)),
        'arena_matches': len(arenas),
        'promotions': sum(1 for a in arenas if a.get('promoted')),
        'hours': round(sum(m.get('elapsed_seconds', 0) for m in metrics) / 3600, 1),
    }


def get_model(game: str, model_id: str):
    """Retrieve or load model and create evaluator."""
    if game == 'chaturanga':
        checkpoint_path = _checkpoint_path(game, model_id)
        if checkpoint_path is None:
            return None, None

        mtime = os.path.getmtime(checkpoint_path)
        cache_key = (checkpoint_path, mtime)
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        try:
            # Architecture comes from the checkpoint (local 5x64 or cloud 8x128)
            net, _ = load_net_from_checkpoint(checkpoint_path, device=device)

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
        checkpoint_path = _checkpoint_path(game, model_id)
        if checkpoint_path is None:
            return None, None

        mtime = os.path.getmtime(checkpoint_path)
        cache_key = (checkpoint_path, mtime)
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        try:
            net, _ = load_net_from_checkpoint(checkpoint_path, device=device)

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
    def __init__(self, game: str = 'chaturanga', p1_type: str = 'human', p2_type: str = 'az_best'):
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
        self.actions = []   # actions played, parallel to move_log
        self.evals = []     # White's expected score (0-1) after each move, None if unknown
        self.analysis = {}  # (ply, model, sims) -> search summary of that position

    def current_player_type(self) -> str:
        # P1 is side 1 (White / Red), P2 is side -1 (Black / Yellow)
        return self.p1_type if self.state.side_to_move == 1 else self.p2_type


current_session = GameSession('chaturanga', 'human', 'az_best')

# Flask serves requests on several threads; every route that reads or changes
# the shared game session holds this lock, so e.g. a double-clicked "AI move"
# cannot apply two moves to the same position (needtofix L4).
_SESSION_LOCK = threading.Lock()


def _with_session_lock(route):
    @wraps(route)
    def locked(*args, **kwargs):
        with _SESSION_LOCK:
            return route(*args, **kwargs)
    return locked


def serialize_state(session: GameSession) -> dict:
    state = session.state
    game = session.game
    # Material adjudication is a training-only device (§2.2): a capped game is a draw
    done, z = state.is_terminal(adjudicate=False)

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
            'evals': session.evals,
            'last_move': ({'from_sq': int(decode_action(session.actions[-1])[0]),
                           'to_sq': int(decode_action(session.actions[-1])[1])}
                          if session.actions else None),
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
    models_data = {
        'chaturanga': [
            {'id': 'human', 'name': 'Human Player', 'type': 'human'},
            {'id': 'az_best', 'name': 'AlphaZero · best model', 'type': 'az'},
            {'id': 'az_latest', 'name': 'AlphaZero · latest checkpoint', 'type': 'az'},
            {'id': 'baseline_alphabeta', 'name': 'Baseline · alpha-beta search (3 ply)', 'type': 'baseline'},
            {'id': 'baseline_greedy', 'name': 'Baseline · greedy material', 'type': 'baseline'},
            {'id': 'random', 'name': 'Baseline · random moves', 'type': 'random'},
        ],
        'connect4': [
            {'id': 'human', 'name': 'Human Player', 'type': 'human'},
            {'id': 'az_final', 'name': 'AlphaZero Connect-4 (Trained)', 'type': 'az'},
            {'id': 'az_phase0', 'name': 'AlphaZero Connect-4 (Phase 0 Pass)', 'type': 'az'},
            {'id': 'random', 'name': 'Random Moves', 'type': 'random'},
        ]
    }
    # Every other checkpoint in the run is an opponent too ("old vs new" games);
    # the newest is already listed as "latest checkpoint"
    from pipeline.resume import list_checkpoints
    ckpts = list_checkpoints(os.path.join(PROJECT_ROOT, 'runs', 'chaturanga'))
    older = [{'id': f'ckpt_{it:04d}', 'name': 'AlphaZero · iteration', 'type': 'az'}
             for it, _ in reversed(ckpts[:-1])]
    chat = models_data['chaturanga']
    insert_at = next(i for i, m in enumerate(chat) if m['id'] == 'az_latest') + 1
    chat[insert_at:insert_at] = older

    # Say so when an AlphaZero player has no checkpoint: it then plays random
    # moves, which must not pass for the trained agent (N14)
    for game, entries in models_data.items():
        for entry in entries:
            if entry['type'] == 'az':
                path = _checkpoint_path(game, entry['id'])
                entry['available'] = path is not None
                if not entry['available']:
                    entry['name'] += ' (no checkpoint: plays random moves)'
                elif game == 'chaturanga':
                    iteration = _checkpoint_info(path)['iteration']
                    if entry['id'].startswith('ckpt_'):
                        entry['name'] += f' {iteration}'
                    elif iteration is not None:
                        entry['name'] += f' (iteration {iteration})'
    return jsonify(models_data)


@app.route('/api/model_info', methods=['GET'])
def model_info():
    """What the model card shows for an AlphaZero player id (default: the best model)."""
    model_id = request.args.get('id', 'az_best')
    path = _checkpoint_path('chaturanga', model_id)
    if path is None:
        return jsonify({'available': False, 'id': model_id})
    return jsonify({
        'available': True,
        'id': model_id,
        'play_sims': PLAY_SIMS,
        **_checkpoint_info(path),
        **_run_summary(os.path.dirname(path)),
    })


@app.route('/api/positions', methods=['GET'])
@_with_session_lock
def game_positions():
    """Board after every ply of the current Chaturanga game, for step-by-step review."""
    if current_session.game != 'chaturanga':
        return jsonify({'positions': []})
    states = list(current_session.history) + [current_session.state]
    positions = []
    for ply, st in enumerate(states):
        board = [[int(st.piece_at(sq(r, f))) for f in range(8)] for r in range(8)]
        last = None
        if ply > 0 and ply - 1 < len(current_session.actions):
            f_s, t_s = decode_action(current_session.actions[ply - 1])
            last = {'from_sq': int(f_s), 'to_sq': int(t_s)}
        positions.append({'ply': ply, 'board': board, 'last_move': last,
                          'side_to_move': int(st.side_to_move)})
    return jsonify({'positions': positions})


@app.route('/api/analyze_move', methods=['POST'])
@_with_session_lock
def analyze_move():
    """Label move `ply` (0-based) of the current Chaturanga game, judged by `model`."""
    if current_session.game != 'chaturanga':
        return jsonify({'error': 'analysis is for Chaturanga games'}), 400
    body = request.get_json(silent=True) or {}
    model = body.get('model', 'az_best')
    try:
        ply = int(body.get('ply', -1))
        sims = max(MIN_SIMS, min(MAX_SIMS, int(body.get('sims', ANALYSIS_SIMS))))
    except (TypeError, ValueError):
        return jsonify({'error': 'ply and sims must be integers'}), 400
    if not 0 <= ply < len(current_session.actions):
        return jsonify({'error': f'no move {ply}'}), 400
    _, evaluate = get_model('chaturanga', model)
    if evaluate is None:
        return jsonify({'error': f'no checkpoint for {model}'}), 400

    states = list(current_session.history) + [current_session.state]

    def summary(k):
        key = (k, model, sims)
        if key not in current_session.analysis:
            current_session.analysis[key] = _search_summary(states[k], evaluate, sims)
        return current_session.analysis[key]

    action = current_session.actions[ply]
    result = classify_move(states[ply], states[ply + 1], action, summary(ply), summary(ply + 1))
    result.update(ply=ply + 1, move=_move_name(states[ply], action, 'chaturanga'),
                  side='White' if states[ply].side_to_move == 1 else 'Black')
    return jsonify(result)


@app.route('/api/state', methods=['GET'])
@_with_session_lock
def get_state():
    return jsonify(serialize_state(current_session))


@app.route('/api/new_game', methods=['POST'])
@_with_session_lock
def new_game():
    data = request.json or {}
    game = data.get('game', 'chaturanga')
    p1 = data.get('p1_type', 'human')
    p2 = data.get('p2_type', 'az_best')

    current_session.game = game
    current_session.p1_type = p1
    current_session.p2_type = p2
    current_session.reset()
    current_session.analysis = {}

    return jsonify(serialize_state(current_session))


@app.route('/api/make_move', methods=['POST'])
@_with_session_lock
def make_move():
    data = request.json or {}
    action = data.get('action')
    if action is None:
        return jsonify({'error': 'action required'}), 400

    action = int(action)
    state = current_session.state
    # A finished game has no next move (pseudo-legal moves still exist after a
    # Raja capture, and applying one at the move cap trips apply_move's
    # assertion), and the human may only move for a human side (needtofixagain N3)
    if state.is_terminal(adjudicate=False)[0]:
        return jsonify({'error': 'game is already terminal'}), 400
    if current_session.current_player_type() != 'human':
        return jsonify({'error': 'it is not a human player\'s turn'}), 400
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
    current_session.actions.append(int(action))
    current_session.evals.append(None)      # a human move: no search ran
    current_session.state = state.apply(action)

    return jsonify(serialize_state(current_session))


@app.route('/api/ai_move', methods=['POST'])
@_with_session_lock
def ai_move():
    state = current_session.state
    done, _ = state.is_terminal(adjudicate=False)
    if done:
        return jsonify({'error': 'game is already terminal'}), 400

    player_type = current_session.current_player_type()
    if player_type == 'human':
        return jsonify({'error': 'current player is human'}), 400

    action_size = 4096 if current_session.game == 'chaturanga' else 7
    chosen_action = None
    ai_stats = {}
    body = request.get_json(silent=True) or {}
    try:
        requested_sims = int(body.get('sims', PLAY_SIMS))
    except (TypeError, ValueError):
        requested_sims = PLAY_SIMS

    if player_type in ('az_latest', 'az_best', 'az_final', 'az_phase0') or player_type.startswith('ckpt_'):
        _, evaluate = get_model(current_session.game, player_type)
        if evaluate is not None:
            # Run MCTS search
            n_sims = max(MIN_SIMS, min(MAX_SIMS, requested_sims))
            c_puct = 1.5
            vc, root = search(
                state=state,
                evaluate=evaluate,
                n_sims=n_sims,
                action_size=action_size,
                c_puct=c_puct,
                add_noise=False,
                adjudicate=False,   # a capped human game is a draw (§2.2)
                return_root=True,
            )
            legal = state.legal_moves()
            legal_vc = {int(a): int(vc[a]) for a in legal}
            # Pick argmax
            chosen_action = max(legal_vc.keys(), key=lambda a: legal_vc[a])

            # Position value from the search (the chosen move's mean backed-up value,
            # side to move's view), turned into White's / Red's view (+1 = first
            # player winning) so the evaluation bar always reads the same way.
            edge = root.legal_actions.index(chosen_action)
            side_value = float(root.W[edge] / root.N[edge]) if root.N[edge] > 0 else 0.0
            root_val = side_value * state.side_to_move
            total_visits = sum(legal_vc.values())

            # The network's own move probabilities at the root (its "instinct",
            # before any search; no exploration noise in this app)
            prior = {int(a): float(p) for a, p in zip(root.legal_actions, root.P)}
            instinct = max(prior, key=prior.get)

            def describe(a):
                count = legal_vc.get(a, 0)
                entry = {'action': a, 'name': _move_name(state, a, current_session.game),
                         'visits': count,
                         'pct': round(count / total_visits * 100, 1) if total_visits else 0.0,
                         'prior': round(prior.get(a, 0.0) * 100, 1)}
                if current_session.game == 'chaturanga':
                    entry['piece'] = int(state.piece_at(decode_action(a)[0]))
                return entry

            # Top 3 by search visits, plus the network's favourite if search dropped it
            ranked = [a for a, _ in sorted(legal_vc.items(), key=lambda kv: kv[1], reverse=True)[:3]]
            top_moves = [describe(a) for a in ranked]
            ai_stats = {
                'value': round(root_val, 3),       # White's / Red's view
                'top_moves': top_moves,
                'instinct': describe(instinct),
                'search_agrees': instinct == chosen_action,
                'sims': n_sims,
                'total_visits': total_visits,
            }
        else:
            # No checkpoint for this player (N14): play randomly, but say so
            chosen_action = random_player(state, action_size)
            ai_stats = {'note': f'no checkpoint for {player_type}; played a random move'}
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
    current_session.actions.append(int(chosen_action))
    # Momentum: the search's value (White's view, -1..1) as White's expected score
    current_session.evals.append(
        round((ai_stats['value'] + 1) / 2, 4) if 'value' in ai_stats else None)
    current_session.state = state.apply(chosen_action)

    resp = serialize_state(current_session)
    resp['ai_stats'] = ai_stats
    resp['played_action'] = int(chosen_action)
    return jsonify(resp)


@app.route('/api/undo', methods=['POST'])
@_with_session_lock
def undo_move():
    if not current_session.history:
        return jsonify({'error': 'no moves to undo'}), 400

    # Pop last state
    current_session.analysis = {}       # positions after the undone move are gone
    current_session.state = current_session.history.pop()
    if current_session.move_log:
        current_session.move_log.pop()
    if current_session.actions:
        current_session.actions.pop()
    if current_session.evals:
        current_session.evals.pop()

    # If the user is playing against an AI, pop again so it returns to human's turn
    if current_session.p1_type == 'human' and current_session.p2_type != 'human' and current_session.state.side_to_move == -1 and current_session.history:
        current_session.state = current_session.history.pop()
        if current_session.move_log:
            current_session.move_log.pop()
        if current_session.actions:
            current_session.actions.pop()
        if current_session.evals:
            current_session.evals.pop()
    elif current_session.p2_type == 'human' and current_session.p1_type != 'human' and current_session.state.side_to_move == 1 and current_session.history:
        current_session.state = current_session.history.pop()
        if current_session.move_log:
            current_session.move_log.pop()
        if current_session.actions:
            current_session.actions.pop()
        if current_session.evals:
            current_session.evals.pop()

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
