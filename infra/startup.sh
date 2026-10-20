#!/usr/bin/env bash
# ChaturangaZero VM startup script (Phase 4b, needtofix I3).
#
# Runs on every boot of the training VM — including the fresh VM that the
# managed instance group creates after a spot preemption. It pulls the code,
# restores the run directory from Cloud Storage, and resumes training
# unattended from the last completed iteration.
#
# Supervision (needtofixagain N4): a VM that is up but not training still
# bills. So:
#   * training runs as a systemd service that restarts on failure (CUDA OOM,
#     a bucket hiccup, the GPU driver not ready yet on first boot);
#   * a watchdog (infra/watchdog.sh) shuts the VM down if metrics.json stops
#     changing for cz-heartbeat-minutes; the MIG then recreates it;
#   * if this script itself fails (clone, pip), the VM shuts down after a
#     short delay instead of idling, so the MIG retries on a fresh VM; after
#     MAX_STARTUP_FAILURES failures in a row the group is resized to 0;
#   * when the run is finished, infra/run_training.sh resizes the group to 0,
#     so a completed run doesn't keep a VM (and the watchdog) cycling.
#
# A preemption gives systemd 25 s to stop training (TimeoutStopSec): the
# SIGTERM handler's final sync may be cut short, which only loses the
# iteration in progress — every completed iteration is already in the bucket.
#
# Settings come from instance metadata (set by infra/create_vm.sh):
#   cz-repo     git URL             cz-branch   branch to check out
#   cz-bucket   gs://bucket          cz-run      run name (folder in the bucket)
#   cz-config   config path in the repo, e.g. configs/cloud.yaml
#   cz-heartbeat-minutes   watchdog limit (default 240: > one iteration + startup)
#   cz-mig      the managed instance group's name (to resize it to 0 when done)
set -euo pipefail

LOG=/var/log/chaturangazero.log
MAX_STARTUP_FAILURES=5

meta() {
  curl -fs -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" || true
}
ZONE="$(curl -fs -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/zone" || true)"; ZONE="${ZONE##*/}"
MIG="$(meta cz-mig)"

on_error() {
  trap - ERR
  local counter="${BUCKET:-}/${RUN:-main}/startup_failures" failures=0
  if [ -n "${BUCKET:-}" ]; then
    failures="$(gcloud storage cat "$counter" 2>/dev/null || echo 0)"
    failures=$(( ${failures//[^0-9]/} + 1 ))
    echo "$failures" | gcloud storage cp - "$counter" >/dev/null 2>&1 || true
  fi
  echo "[$(date -Is)] startup FAILED at line $1 (failure $failures in a row)" >> "$LOG"
  if [ "$failures" -ge "$MAX_STARTUP_FAILURES" ] && [ -n "$MIG" ]; then
    echo "[$(date -Is)] giving up: resizing $MIG to 0" >> "$LOG"
    gcloud compute instance-groups managed resize "$MIG" --size 0 --zone "$ZONE" --quiet >> "$LOG" 2>&1 || true
  fi
  shutdown -h +10 || true      # the MIG retries on a fresh VM (unless resized to 0)
}
trap 'on_error $LINENO' ERR

REPO="$(meta cz-repo)"
BRANCH="$(meta cz-branch)"; BRANCH="${BRANCH:-main}"
BUCKET="$(meta cz-bucket)"
RUN="$(meta cz-run)"; RUN="${RUN:-main}"
CONFIG="$(meta cz-config)"; CONFIG="${CONFIG:-configs/cloud.yaml}"
HEARTBEAT_MIN="$(meta cz-heartbeat-minutes)"; HEARTBEAT_MIN="${HEARTBEAT_MIN:-240}"

WORK=/opt/chaturangazero
RUN_DIR=/opt/runs/"$RUN"
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
PYTHON="$(command -v python3)"
"$PYTHON" -m pip install --quiet -r requirements.txt

# The NVIDIA driver (install-nvidia-driver=True) may still be installing on
# first boot. Wait for it rather than starting a CUDA run that crashes; if it
# never comes up, the service's restarts and then the watchdog take over.
for _ in $(seq 1 60); do
  nvidia-smi >/dev/null 2>&1 && break
  sleep 10
done
nvidia-smi >> "$LOG" 2>&1 || echo "[$(date -Is)] WARNING: nvidia-smi still failing" >> "$LOG"

# Self-play runs several worker processes on the one GPU (loop.workers). Without
# MPS their kernels time-slice the device; with it they run concurrently
# (needtofixagain N7). Harmless for a single process.
nvidia-cuda-mps-control -d >> "$LOG" 2>&1 \
  || echo "[$(date -Is)] WARNING: could not start the CUDA MPS daemon" >> "$LOG"

# Train: restores $RUN_DIR from the bucket, resumes, syncs after every iteration.
# Exit 143 is the loop's clean SIGTERM (preemption) exit, not a failure.
cat > /etc/systemd/system/cz-train.service <<EOF
[Unit]
Description=ChaturangaZero training
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=3600
StartLimitBurst=5

[Service]
WorkingDirectory=$WORK
Environment=CZ_MIG=$MIG CZ_ZONE=$ZONE
ExecStart=/bin/bash $WORK/infra/run_training.sh $PYTHON train_chaturanga.py --config $CONFIG --output $RUN_DIR --resume auto --sync-to $BUCKET/$RUN --device cuda
Restart=on-failure
RestartSec=60
SuccessExitStatus=143
KillSignal=SIGTERM
TimeoutStopSec=25
StandardOutput=append:$LOG
StandardError=append:$LOG
EOF

# Watchdog: no new metrics for HEARTBEAT_MIN minutes -> shut down, MIG recreates
cat > /etc/systemd/system/cz-watchdog.service <<EOF
[Unit]
Description=ChaturangaZero training heartbeat watchdog

[Service]
ExecStart=/bin/bash $WORK/infra/watchdog.sh $RUN_DIR/metrics.json $(( HEARTBEAT_MIN * 60 )) 60 /sbin/shutdown -h now
StandardOutput=append:$LOG
StandardError=append:$LOG
EOF

systemctl daemon-reload
systemctl restart cz-watchdog.service
systemctl restart cz-train.service
trap - ERR
# Startup succeeded: reset the consecutive-failure counter
echo 0 | gcloud storage cp - "$BUCKET/$RUN/startup_failures" >/dev/null 2>&1 || true

# Ship the log to the bucket every few minutes, so it survives the VM
( while true; do sleep 300; gcloud storage cp "$LOG" "$BUCKET/$RUN/logs/$(hostname).log" >/dev/null 2>&1 || true; done ) &
