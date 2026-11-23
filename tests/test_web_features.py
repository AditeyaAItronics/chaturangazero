"""Web app features: checkpoint opponents, search strength, network instinct
vs search, and the positions used to review a finished game."""

import pytest

from chaturanga.moves import decode_action


@pytest.fixture
def web(tmp_path, monkeypatch):
    """Test client over a run directory holding three tiny checkpoints."""
    pytest.importorskip('flask')
    from web import app as web_app
    from net.model import create_chaturanga_net
    from net.train import Trainer

    run = tmp_path / 'runs' / 'chaturanga'
    run.mkdir(parents=True)
    trainer = Trainer(create_chaturanga_net(n_blocks=1, channels=8), device='cpu')
    for iteration in (3, 10):
        trainer.save_checkpoint(str(run / f'checkpoint_{iteration:04d}.pt'), iteration=iteration)
    trainer.save_checkpoint(str(run / 'best.pt'), iteration=7)
    monkeypatch.setattr(web_app, 'PROJECT_ROOT', str(tmp_path))

    s = web_app.current_session
    saved = (s.game, s.p1_type, s.p2_type, s.state, list(s.history),
             list(s.move_log), list(s.actions))
    yield web_app, web_app.app.test_client()
    (s.game, s.p1_type, s.p2_type, s.state, s.history,
     s.move_log, s.actions) = saved


def _new_game(client, p1, p2):
    return client.post('/api/new_game', json={'game': 'chaturanga', 'p1_type': p1, 'p2_type': p2})


def test_older_checkpoints_are_opponents(web):
    web_app, client = web
    names = {m['id']: m['name'] for m in client.get('/api/models').get_json()['chaturanga']}
    assert names['az_best'] == 'AlphaZero · best model (iteration 8)'
    assert names['az_latest'] == 'AlphaZero · latest checkpoint (iteration 11)'
    assert names['ckpt_0003'] == 'AlphaZero · iteration 4'
    assert 'ckpt_0010' not in names          # already listed as "latest"


def test_checkpoint_ids_cannot_escape_the_run_directory(web):
    web_app, _ = web
    assert web_app._checkpoint_path('chaturanga', 'ckpt_0003') is not None
    for bad in ('ckpt_../../x', 'ckpt_', 'ckpt_3.pt', 'ckpt_-1', 'ckpt_0004'):
        assert web_app._checkpoint_path('chaturanga', bad) is None, bad


@pytest.mark.parametrize('asked, used', [(5, 10), (40, 40), (5000, 800), ('lots', 200)])
def test_search_strength_is_clamped(web, asked, used):
    _, client = web
    _new_game(client, 'ckpt_0003', 'human')
    stats = client.post('/api/ai_move', json={'sims': asked}).get_json()['ai_stats']
    assert stats['sims'] == used
    assert stats['total_visits'] >= used - 1


def test_ai_move_reports_network_instinct(web):
    _, client = web
    _new_game(client, 'az_best', 'human')
    resp = client.post('/api/ai_move', json={'sims': 20}).get_json()
    stats = resp['ai_stats']
    played = stats['top_moves'][0]
    assert played['action'] == resp['played_action']
    assert all(0 <= m['prior'] <= 100 for m in stats['top_moves'])
    assert 0 < stats['instinct']['prior'] <= 100
    assert stats['search_agrees'] == (stats['instinct']['action'] == played['action'])


def test_ai_move_without_a_body_still_works(web):
    _, client = web
    _new_game(client, 'ckpt_0003', 'human')
    resp = client.post('/api/ai_move', headers={'Content-Type': 'application/json'})
    assert resp.status_code == 200


def test_positions_follow_the_game_and_undo(web):
    _, client = web
    state = _new_game(client, 'human', 'human').get_json()
    played = []
    for _ in range(2):
        move = state['legal_moves'][0]
        played.append(move['action'])
        state = client.post('/api/make_move', json={'action': move['action']}).get_json()
    assert state['last_move'] == dict(zip(('from_sq', 'to_sq'), decode_action(played[-1])))

    positions = client.get('/api/positions').get_json()['positions']
    assert [p['ply'] for p in positions] == [0, 1, 2]
    assert positions[0]['last_move'] is None
    for ply, action in enumerate(played, start=1):
        f_s, t_s = decode_action(action)
        assert positions[ply]['last_move'] == {'from_sq': f_s, 'to_sq': t_s}
        board = positions[ply]['board']
        assert board[t_s // 8][t_s % 8] != 0 and board[f_s // 8][f_s % 8] == 0

    client.post('/api/undo')
    positions = client.get('/api/positions').get_json()['positions']
    assert len(positions) == 2
    assert positions[-1]['last_move'] == dict(zip(('from_sq', 'to_sq'), decode_action(played[0])))


def test_momentum_evals_follow_moves_and_undo(web):
    _, client = web
    state = _new_game(client, 'human', 'ckpt_0003').get_json()
    assert state['evals'] == []
    move = state['legal_moves'][0]['action']
    client.post('/api/make_move', json={'action': move})
    state = client.post('/api/ai_move', json={'sims': 10}).get_json()
    human_eval, ai_eval = state['evals']
    assert human_eval is None                    # no search ran for the human move
    assert 0.0 <= ai_eval <= 1.0
    # White's expected score = (value from White's view + 1) / 2
    assert ai_eval == pytest.approx((state['ai_stats']['value'] + 1) / 2, abs=1e-3)
    state = client.post('/api/undo').get_json()  # undoes the AI reply and the human move
    assert state['evals'] == [] and state['move_history'] == []


# --- Move analysis: blunders and brilliant moves -----------------------------

import numpy as np
from chaturanga.board import State, WHITE, BLACK, RAJA, RATHA, PADATI, sq
from chaturanga.moves import encode_action


def _position(pieces, side=WHITE):
    board = np.zeros(64, dtype=np.int8)
    for (rank, file), piece in pieces.items():
        board[sq(rank, file)] = piece
    return State(board=board, side_to_move=side, ply_count=10)


def _info(value, best=None, q=None, visits=None, prior=None, terminal=False):
    return {'terminal': terminal, 'value': value, 'best': best, 'q': q or {},
            'visits': visits or {}, 'prior': prior or {}}


@pytest.fixture
def classify():
    pytest.importorskip('flask')
    from web import app as web_app
    return web_app.classify_move


def test_missing_a_raja_capture_is_a_blunder(classify):
    # White's Ratha on h1 can take Black's Raja on h8; White plays Raja a1-a2 instead
    before = _position({(0, 0): RAJA, (0, 7): RATHA, (7, 7): -RAJA, (6, 0): -PADATI})
    capture, other = encode_action(sq(0, 7), sq(7, 7)), encode_action(sq(0, 0), sq(1, 0))
    r = classify(before, before.apply(other), other, _info(1.0, best=capture), _info(0.0))
    assert r['label'] == '??' and 'Missed a Raja capture' in r['reason']
    # Playing the capture itself is never labelled a mistake
    r = classify(before, before.apply(capture), capture, _info(1.0, best=capture),
                 _info(-1.0, terminal=True))
    assert r['label'] is None


def test_leaving_the_raja_capturable_is_a_blunder(classify):
    # White's Raja steps from d1 onto the e-file, where Black's Ratha on e8 takes it
    before = _position({(0, 3): RAJA, (7, 4): -RATHA, (7, 0): -RAJA, (1, 7): PADATI})
    move = encode_action(sq(0, 3), sq(0, 4))
    r = classify(before, before.apply(move), move, _info(0.0, best=move), _info(0.0))
    assert r['label'] == '??' and 'Raja capturable' in r['reason']


@pytest.mark.parametrize('after_value, label', [
    (0.1, '??'),    # mover 0.5 -> -0.1: loses 30 points
    (-0.2, '?'),    # 0.5 -> 0.2: 15 points
    (-0.35, '?!'),  # 0.5 -> 0.35: 7.5 points
    (-0.45, None),  # 2.5 points: noise, no label
])
def test_loss_thresholds(classify, after_value, label):
    start = State()
    move, best = start.legal_moves()[0], start.legal_moves()[1]
    r = classify(start, start.apply(move), move, _info(0.5, best=best), _info(after_value))
    assert r['label'] == label
    if label:
        assert r['better']                       # names the search's preferred move


def test_the_search_best_move_is_never_called_a_mistake(classify):
    start = State()
    move = start.legal_moves()[0]
    # Even if the next position's search comes out much lower (search noise)
    r = classify(start, start.apply(move), move, _info(0.5, best=move, q={move: 0.5}), _info(0.4))
    assert r['label'] is None


def test_brilliant_and_only_moves(classify):
    start = State()
    move, instinct, other = start.legal_moves()[:3]
    q = {move: 0.3, instinct: -0.1, other: 0.25}
    visits = {move: 120, instinct: 60, other: 20}
    # Search's best, network gave it 2%, its favourite is 20 points worse: brilliant
    r = classify(start, start.apply(move), move,
                 _info(0.3, best=move, q=q, visits=visits, prior={move: 0.02, instinct: 0.6, other: 0.2}),
                 _info(-0.3))
    assert r['label'] == '!!'
    # Network liked it too, but every serious alternative is >= 10 points worse: only move
    q2 = {move: 0.3, instinct: 0.05, other: 0.05}
    r = classify(start, start.apply(move), move,
                 _info(0.3, best=move, q=q2, visits=visits, prior={move: 0.5, instinct: 0.3, other: 0.2}),
                 _info(-0.3))
    assert r['label'] == '!'
    # A decided game gets no praise
    r = classify(start, start.apply(move), move,
                 _info(0.95, best=move, q=q2, visits=visits, prior={move: 0.02, instinct: 0.6}),
                 _info(-0.95))
    assert r['label'] is None


def test_analyze_move_endpoint(web):
    web_app, client = web
    state = _new_game(client, 'human', 'human').get_json()
    for _ in range(3):
        state = client.post('/api/make_move', json={'action': state['legal_moves'][0]['action']}).get_json()
    results = [client.post('/api/analyze_move', json={'ply': k, 'sims': 10}).get_json() for k in range(3)]
    assert [r['ply'] for r in results] == [1, 2, 3]
    assert [r['side'] for r in results] == ['White', 'Black', 'White']
    for r in results:
        assert set(r) >= {'label', 'reason', 'loss', 'white_before', 'white_after', 'move'}
        assert 0 <= r['white_after'] <= 1
    # Consecutive moves share a position: one search, cached
    assert results[0]['white_after'] == results[1]['white_before']
    assert len(web_app.current_session.analysis) == 4
    assert client.post('/api/analyze_move', json={'ply': 3}).status_code == 400
    client.post('/api/undo')
    assert web_app.current_session.analysis == {}
