"""Resuming with reset_lr_on_resume restarts the LR from the config (the saved
optimizer/scheduler state used to override it silently)."""

from net.model import create_chaturanga_net
from net.train import Trainer


def _trained_checkpoint(tmp_path, steps=200):
    """A checkpoint whose scheduler has decayed 2e-3 -> 2e-5 via milestones [80, 160]."""
    t = Trainer(create_chaturanga_net(n_blocks=1, channels=8), lr=2e-3,
                lr_milestones=[80, 160], device='cpu')
    for _ in range(steps):
        t.optimizer.step()
        t.step_scheduler()
    path = str(tmp_path / 'ckpt.pt')
    t.save_checkpoint(path, iteration=steps - 1)
    return path


def _resumed(path, **config):
    t = Trainer(create_chaturanga_net(n_blocks=1, channels=8), device='cpu', **config)
    t.load_checkpoint(path)
    return t


def lr_of(t):
    return [g['lr'] for g in t.optimizer.param_groups]


def test_plain_resume_keeps_checkpoint_lr(tmp_path):
    # Unchanged behaviour without the option: the checkpoint wins over the config
    path = _trained_checkpoint(tmp_path)
    t = _resumed(path, lr=2e-4, lr_milestones=[300])
    assert lr_of(t) == [2e-5, 2e-5]


def test_reset_lr_without_future_milestones_is_constant(tmp_path):
    path = _trained_checkpoint(tmp_path)
    t = _resumed(path, lr=2e-4, lr_milestones=[80, 160])
    t.reset_lr(2e-4, lr_milestones=[80, 160])
    assert lr_of(t) == [2e-4, 2e-4]
    for _ in range(50):
        t.optimizer.step()
        t.step_scheduler()
    assert lr_of(t) == [2e-4, 2e-4]


def test_reset_lr_applies_later_milestones_on_absolute_steps(tmp_path):
    path = _trained_checkpoint(tmp_path, steps=200)     # scheduler at step 200
    t = _resumed(path, lr=2e-4)
    t.reset_lr(2e-4, lr_milestones=[80, 205], lr_gamma=0.1)
    assert lr_of(t) == [2e-4, 2e-4]
    seen = []
    for _ in range(10):
        t.optimizer.step()
        t.step_scheduler()
        seen.append(t.optimizer.param_groups[0]['lr'])
    # Steps 201..204 stay at 2e-4; step 205 decays once; 80 never fires again
    assert seen[:4] == [2e-4] * 4
    assert all(abs(x - 2e-5) < 1e-12 for x in seen[4:])


def test_reset_run_resumed_again_does_not_reapply_old_milestones(tmp_path):
    # A reset run interrupted and resumed with the same config (old milestones
    # [80, 160], with or without the flag) must stay at the reset LR
    path = _trained_checkpoint(tmp_path)
    t = _resumed(path, lr=2e-4, lr_milestones=[80, 160])
    t.reset_lr(2e-4, lr_milestones=[80, 160])
    t.optimizer.step()
    t.step_scheduler()
    again = str(tmp_path / 'again.pt')
    t.save_checkpoint(again, iteration=200)
    for reset in (False, True):
        t2 = _resumed(again, lr=2e-4, lr_milestones=[80, 160])
        if reset:
            t2.reset_lr(2e-4, lr_milestones=[80, 160])
        for _ in range(200):
            t2.optimizer.step()
            t2.step_scheduler()
        assert lr_of(t2) == [2e-4, 2e-4], f"reset={reset}"


def test_reset_lr_survives_save_and_load(tmp_path):
    path = _trained_checkpoint(tmp_path)
    t = _resumed(path, lr=2e-4)
    t.reset_lr(2e-4, lr_milestones=[500])
    t.optimizer.step()
    t.step_scheduler()
    again = str(tmp_path / 'again.pt')
    t.save_checkpoint(again, iteration=200)
    t2 = _resumed(again, lr=2e-3, lr_milestones=[500])
    assert lr_of(t2) == [2e-4, 2e-4]
    assert t2.scheduler.last_epoch == t.scheduler.last_epoch
