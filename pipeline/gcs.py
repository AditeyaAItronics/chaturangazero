"""
Run-directory sync to Cloud Storage (Phase 4b, needtofix I3).

Design for the VM disappearing without warning (spot preemption, §2.4):
every checkpoint, buffer snapshot and metrics file is pushed to the bucket
after each iteration, and a fresh VM pulls the run directory back before
resuming.

    remote = 'gs://bucket/run-name'  -> uses the `gcloud storage rsync` CLI
                                        (preinstalled on GCP VM images)
    remote = any local path          -> plain file copy; used for local testing
                                        and by tools/soak_test.py

Temporary files (*.tmp, *.tmp.npz) are never synced, so a half-written file
can't overwrite a good one in the bucket.
"""

import os
import shutil
import subprocess

_SKIP_SUFFIXES = ('.tmp', '.tmp.npz', '.partial')


def _is_gcs(path: str) -> bool:
    return str(path).startswith('gs://')


def _gcloud_rsync(src: str, dst: str):
    cmd = ['gcloud', 'storage', 'rsync', '--recursive',
           '--exclude', r'.*\.tmp(\.npz)?$|.*\.partial$', src, dst]
    subprocess.run(cmd, check=True)


def _local_mirror(src: str, dst: str):
    """Copy new/changed files from src to dst (no deletions), skipping temp files."""
    if not os.path.isdir(src):
        return
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        out_dir = os.path.join(dst, rel) if rel != '.' else dst
        os.makedirs(out_dir, exist_ok=True)
        for name in files:
            if name.endswith(_SKIP_SUFFIXES):
                continue
            a, b = os.path.join(root, name), os.path.join(out_dir, name)
            sa = os.stat(a)
            if os.path.exists(b):
                sb = os.stat(b)
                if sb.st_size == sa.st_size and sb.st_mtime >= sa.st_mtime:
                    continue
            tmp = b + '.partial'
            shutil.copy2(a, tmp)
            os.replace(tmp, b)


def push(local_dir: str, remote: str):
    """Upload the run directory (new/changed files) to `remote`."""
    if not remote:
        return
    if _is_gcs(remote):
        _gcloud_rsync(local_dir, remote)
    else:
        _local_mirror(local_dir, remote)


def pull(remote: str, local_dir: str):
    """Download a run directory from `remote` (if it exists) into `local_dir`."""
    if not remote:
        return
    os.makedirs(local_dir, exist_ok=True)
    if _is_gcs(remote):
        probe = subprocess.run(['gcloud', 'storage', 'ls', remote], capture_output=True)
        if probe.returncode == 0:
            _gcloud_rsync(remote, local_dir)
    elif os.path.isdir(remote):
        _local_mirror(remote, local_dir)
