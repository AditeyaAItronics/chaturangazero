"""
Unattended resume (Phase 4b, needtofix I3).

`--resume auto` picks the newest checkpoint in the run directory that
actually loads. A file cut short by a hard kill is skipped instead of
crashing the restart. Checkpoints, metrics and buffer snapshots are all
written atomically (temp file + rename), so in practice a partial file
should never exist, but resume does not rely on that.
"""

import glob
import os
import re
from typing import Optional

import torch

_CKPT = re.compile(r'checkpoint_(\d+)\.pt$')


def list_checkpoints(output_dir: str) -> list[tuple[int, str]]:
    """(iteration, path) for every checkpoint file, oldest first."""
    found = []
    for path in glob.glob(os.path.join(output_dir, 'checkpoint_*.pt')):
        m = _CKPT.search(os.path.basename(path))
        if m:
            found.append((int(m.group(1)), path))
    return sorted(found)


def latest_checkpoint(output_dir: str) -> Optional[str]:
    """Path of the newest checkpoint that loads, or None for a fresh run."""
    for _, path in reversed(list_checkpoints(output_dir)):
        try:
            torch.load(path, map_location='cpu', weights_only=False)
            return path
        except Exception as e:      # truncated / corrupt: fall back to an older one
            print(f"Skipping unreadable checkpoint {path}: {e}")
    return None
