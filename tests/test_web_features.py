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
