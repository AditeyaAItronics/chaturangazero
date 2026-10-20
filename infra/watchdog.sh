#!/usr/bin/env bash
# Training heartbeat watchdog (needtofixagain N4).
#
# The training loop rewrites metrics.json after every iteration. If that file
# stops changing for longer than MAX_AGE seconds, training is stuck or dead
# (systemd gave up restarting it, the GPU driver never came up, a hang...).
# Then the VM is only burning credits, so run ACTION — on the training VM
# that is `shutdown -h now`: the managed instance group sees the instance
# stop and recreates it, and the fresh VM resumes from the bucket.
#
# Until the file first exists, its age is measured from when the watchdog
# started, so a slow first iteration (pip install, bucket pull) gets MAX_AGE
# as its grace period.
#
# Usage: watchdog.sh FILE MAX_AGE_SECONDS CHECK_INTERVAL_SECONDS ACTION [ARGS...]
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: $0 FILE MAX_AGE_SECONDS CHECK_INTERVAL_SECONDS ACTION [ARGS...]" >&2
  exit 2
fi
FILE="$1"; MAX_AGE="$2"; INTERVAL="$3"; shift 3

started="$(date +%s)"
while true; do
  sleep "$INTERVAL"
  now="$(date +%s)"
  if [ -e "$FILE" ]; then
    last="$(stat -c %Y "$FILE" 2>/dev/null || date -r "$FILE" +%s)"
    # A heartbeat older than the watchdog itself (e.g. pulled from the bucket)
    # only starts counting from the watchdog's start
    [ "$last" -lt "$started" ] && last="$started"
  else
    last="$started"
  fi
  age=$(( now - last ))
  if [ "$age" -gt "$MAX_AGE" ]; then
    echo "[$(date -Is)] watchdog: $FILE unchanged for ${age}s (> ${MAX_AGE}s); running: $*"
    "$@"
    exit 0
  fi
done
