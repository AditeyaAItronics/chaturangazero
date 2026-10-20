"""
Unattended resume (Phase 4b, needtofix I3).

`--resume auto` picks the newest checkpoint in the run directory that
actually loads. A file cut short by a hard kill is skipped instead of
crashing the restart. Checkpoints, metrics and buffer snapshots are all
written atomically (temp file + rename), so in practice a partial file
should never exist, but resume does not rely on that.

Commit marker (needtofixagain N2). An iteration writes several files in turn:
checkpoint, buffer snapshot, arena result / best.pt, metrics. A kill (or a
SIGTERM sync) part-way through leaves a checkpoint for an iteration whose
arena gate and metrics never happened. Resuming from that checkpoint would
skip the gate for good and lose the iteration's metrics row. So the loop
writes `run_state.json` LAST, once the iteration is complete, and resume
trusts that marker over the newest checkpoint file: an incomplete iteration
is simply run again.
"""

import glob
import json
import os
import re
from typing import Optional

import torch

_CKPT = re.compile(r'checkpoint_(\d+)\.pt$')
RUN_STATE = 'run_state.json'


def checkpoint_path(output_dir: str, iteration: int) -> str:
    return os.path.join(output_dir, f'checkpoint_{iteration:04d}.pt')


def list_checkpoints(output_dir: str) -> list[tuple[int, str]]:
    """(iteration, path) for every checkpoint file, oldest first."""
    found = []
    for path in glob.glob(os.path.join(output_dir, 'checkpoint_*.pt')):
        m = _CKPT.search(os.path.basename(path))
        if m:
            found.append((int(m.group(1)), path))
    return sorted(found)


def read_run_state(output_dir: str) -> Optional[dict]:
    """The commit marker of the last completed iteration, or None (fresh or pre-marker run)."""
    path = os.path.join(output_dir, RUN_STATE)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            state = json.load(f)
    except (OSError, ValueError) as e:
        print(f"WARNING: unreadable {path} ({e}); resuming without the commit marker")
        return None
    if isinstance(state, dict) and 'last_completed_iteration' in state:
        return state
    print(f"WARNING: {path} has no last_completed_iteration; resuming without the commit marker")
    return None


def write_run_state(output_dir: str, state: dict) -> None:
    """Atomically record a completed iteration. Call only after everything else is written."""
    path = os.path.join(output_dir, RUN_STATE)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def _loads(path: str) -> bool:
    try:
        torch.load(path, map_location='cpu', weights_only=False)
        return True
    except Exception as e:      # truncated / corrupt
        print(f"Skipping unreadable checkpoint {path}: {e}")
        return False


def latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Path of the checkpoint to resume from, or None for a fresh run.

    With a commit marker: the last COMPLETED iteration's checkpoint; newer
    checkpoints belong to an iteration that was cut short and is redone.
    Without one (runs from before the marker existed): the newest checkpoint
    that loads.
    """
    state = read_run_state(output_dir)
    newest_allowed = None
    if state is not None:
        newest_allowed = int(state['last_completed_iteration'])
        path = checkpoint_path(output_dir, newest_allowed)
        if os.path.exists(path) and _loads(path):
            return path
        print(f"Run state names {path}, which is missing or unreadable; "
              f"falling back to the newest earlier checkpoint that loads")
    for iteration, path in reversed(list_checkpoints(output_dir)):
        # Never resume past the marker: newer checkpoints are from a cut-short iteration
        if newest_allowed is not None and iteration > newest_allowed:
            continue
        if _loads(path):
            return path
    return None
