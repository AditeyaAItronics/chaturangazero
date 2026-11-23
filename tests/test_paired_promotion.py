import pytest
from search.arena import paired_p_value


def test_no_evidence_with_draws_or_balanced_pairs():
    assert paired_p_value([0, 0] * 20) == 1.0
    assert paired_p_value([1, -1] * 20) == 1.0


def test_exact_tail_and_direction():
    assert paired_p_value([1, 1] * 5) == 1 / 32
    assert paired_p_value([-1, -1] * 5) == 1.0
    # Two pairs of different magnitudes: sums under H0 are -3,-1,1,3.
    assert paired_p_value([1, 1, -1, 0]) == 0.5


def test_invalid_pairing():
    for outcomes in ([], [1], [1, 0.5]):
        with pytest.raises(ValueError):
            paired_p_value(outcomes)


# --- The gate inside the training loop ---------------------------------------

def _gated_config(alpha, games=2, opening_plies=2):
    from tests.test_critical_fixes import _tiny_connect4_config
    config = _tiny_connect4_config(iterations=1, arena_every=1)
    config['arena'].update(games=games, opening_plies=opening_plies,
                           win_threshold=0.0, promotion_alpha=alpha)
    return config


def test_loop_records_p_value_and_needs_it_to_promote(tmp_path):
    import json
    from pipeline.loop import training_loop

    # win_threshold 0 alone would always promote; with one pair the smallest
    # possible p is 0.5, so alpha 0.05 must block promotion
    out = tmp_path / 'run'
    training_loop(_gated_config(alpha=0.05), output_dir=str(out))
    arena = json.loads((out / 'metrics.json').read_text())[-1]['arena']
    assert arena['promotion_alpha'] == 0.05
    assert arena['p_value'] >= 0.5
    assert arena['promoted'] is False


def test_loop_without_alpha_keeps_score_only_gate(tmp_path):
    import json
    from pipeline.loop import training_loop

    out = tmp_path / 'run'
    training_loop(_gated_config(alpha=None), output_dir=str(out))
    arena = json.loads((out / 'metrics.json').read_text())[-1]['arena']
    assert arena['p_value'] is None
    assert arena['promoted'] is True      # score >= 0 always passes


@pytest.mark.parametrize('kwargs', [dict(alpha=1.5), dict(alpha=0.05, games=3),
                                    dict(alpha=0.05, opening_plies=0)])
def test_loop_rejects_bad_gate_config_before_playing(tmp_path, kwargs):
    from pipeline.loop import training_loop

    with pytest.raises(ValueError):
        training_loop(_gated_config(**kwargs), output_dir=str(tmp_path / 'run'))
    assert not list((tmp_path / 'run').glob('checkpoint_*.pt'))
