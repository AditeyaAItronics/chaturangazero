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


def _gcloud_rsync(src: str, dst: str, delete: bool = False, skip_dirs: tuple = ()):
    exclude = r'.*\.tmp(\.npz)?$|.*\.partial$' + ''.join(f'|{d}/.*' for d in skip_dirs)
    cmd = ['gcloud', 'storage', 'rsync', '--recursive', '--exclude', exclude]
    if delete:
        cmd.append('--delete-unmatched-destination-objects')
    subprocess.run(cmd + [src, dst], check=True)


# Folders whose deletions are mirrored to the remote: buffer shards that
# prune_shards() removed must not pile up in the bucket (needtofixagain N20).
# Everything else (checkpoints, metrics) is only ever added or replaced.
MIRROR_DELETIONS = ('buffer',)


def _has_files(path: str) -> bool:
    return os.path.isdir(path) and any(
        not name.endswith(_SKIP_SUFFIXES) for name in os.listdir(path))


def _delete_unmatched(src: str, dst: str):
    """Remove files in dst that src no longer has (one folder, not recursive)."""
    if not os.path.isdir(dst):
        return
    keep = set(os.listdir(src))
    for name in os.listdir(dst):
        path = os.path.join(dst, name)
        if name not in keep and os.path.isfile(path):
            os.remove(path)


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
    mirrored = tuple(sub for sub in MIRROR_DELETIONS if _has_files(os.path.join(local_dir, sub)))
    if _is_gcs(remote):
        # Folders synced with deletions below are skipped here, so they upload once
        _gcloud_rsync(local_dir, remote, skip_dirs=mirrored)
    else:
        _local_mirror(local_dir, remote)
    for sub in MIRROR_DELETIONS:
        src = os.path.join(local_dir, sub)
        if not _has_files(src):        # never wipe the remote copy from an empty folder
            continue
        if _is_gcs(remote):
            _gcloud_rsync(src, remote.rstrip('/') + '/' + sub, delete=True)
        else:
            _delete_unmatched(src, os.path.join(remote, sub))


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
