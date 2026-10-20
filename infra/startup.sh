#!/usr/bin/env bash
# ChaturangaZero VM startup script (Phase 4b, needtofix I3).
#
# Runs on every boot of the training VM — including the fresh VM that the
# managed instance group creates after a spot preemption. It pulls the code,
# restores the run directory from Cloud Storage, and resumes training
# unattended from the newest valid checkpoint.
#
# Settings come from instance metadata (set by infra/create_vm.sh):
#   cz-repo     git URL             cz-branch   branch to check out
#   cz-bucket   gs://bucket          cz-run      run name (folder in the bucket)
#   cz-config   config path in the repo, e.g. configs/cloud.yaml
set -euo pipefail

meta() {
  curl -fs -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" || true
}

REPO="$(meta cz-repo)"
BRANCH="$(meta cz-branch)"; BRANCH="${BRANCH:-main}"
BUCKET="$(meta cz-bucket)"
RUN="$(meta cz-run)"; RUN="${RUN:-main}"
CONFIG="$(meta cz-config)"; CONFIG="${CONFIG:-configs/cloud.yaml}"

WORK=/opt/chaturangazero
RUN_DIR=/opt/runs/"$RUN"
LOG=/var/log/chaturangazero.log
mkdir -p "$WORK" /opt/runs

echo "[$(date -Is)] startup: repo=$REPO branch=$BRANCH bucket=$BUCKET run=$RUN config=$CONFIG" >> "$LOG"

# Code
if [ -d "$WORK/.git" ]; then
  git -C "$WORK" fetch --depth 1 origin "$BRANCH" && git -C "$WORK" checkout -f FETCH_HEAD
else
  git clone --depth 1 --branch "$BRANCH" "$REPO" "$WORK"
fi

# Python environment (Deep Learning VM images ship with CUDA PyTorch)
cd "$WORK"
python3 -m pip install --quiet -r requirements.txt

# Train: restores $RUN_DIR from the bucket, resumes, syncs after every iteration.
# On SIGTERM (preemption) the loop syncs and exits; the MIG recreates the VM.
nohup python3 train_chaturanga.py \
  --config "$CONFIG" \
  --output "$RUN_DIR" \
  --resume auto \
  --sync-to "$BUCKET/$RUN" \
  --device cuda >> "$LOG" 2>&1 &

# Ship the log to the bucket every few minutes, so it survives the VM
( while true; do sleep 300; gcloud storage cp "$LOG" "$BUCKET/$RUN/logs/$(hostname).log" >/dev/null 2>&1 || true; done ) &
