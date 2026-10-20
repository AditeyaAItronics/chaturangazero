#!/usr/bin/env bash
# Run the training command and stop the VM for good when retrying can't help
# (needtofixagain N4, N8).
#
# The training loop exits 0 only when every iteration is done (a preemption
# exits 143, a crash non-zero). Without this wrapper, a finished run would
# keep a billed VM alive: the heartbeat watchdog would shut it down, the
# managed instance group would recreate it, and the cycle would repeat
# forever. The same happens for a crash that repeats on every start (a bad
# config, broken code). So the instance group is resized to 0:
#   * when the command exits 0 (the run is complete), or
#   * after FAST_FAILURE_LIMIT failures in a row that each happened within
#     FAST_FAILURE_SECONDS of starting (systemd restarts it in between).
#
# Environment: CZ_MIG (instance group name) and CZ_ZONE. If CZ_MIG is unset
# (local use), nothing is resized. CZ_FAILURE_FILE overrides where the
# consecutive-failure count is kept.
#
# Usage: run_training.sh COMMAND [ARGS...]   (exit status is COMMAND's)
set -uo pipefail

FAST_FAILURE_SECONDS="${CZ_FAST_FAILURE_SECONDS:-600}"
FAST_FAILURE_LIMIT="${CZ_FAST_FAILURE_LIMIT:-5}"
FAILURE_FILE="${CZ_FAILURE_FILE:-/var/tmp/cz_training_fast_failures}"

stop_group() {
  [ -n "${CZ_MIG:-}" ] || return 0
  echo "[$(date -Is)] $1; resizing $CZ_MIG to 0"
  gcloud compute instance-groups managed resize "$CZ_MIG" --size 0 \
    --zone "${CZ_ZONE:?CZ_ZONE must be set with CZ_MIG}" --quiet \
    || echo "[$(date -Is)] WARNING: resize failed; delete $CZ_MIG by hand"
}

started="$(date +%s)"
"$@"
status=$?
elapsed=$(( $(date +%s) - started ))

if [ "$status" -eq 0 ]; then
  rm -f "$FAILURE_FILE"
  stop_group "training run complete"
elif [ "$status" -eq 143 ] || [ "$elapsed" -ge "$FAST_FAILURE_SECONDS" ]; then
  rm -f "$FAILURE_FILE"          # preemption, or a crash after real progress: retry
else
  failures=$(( $(cat "$FAILURE_FILE" 2>/dev/null || echo 0) + 1 ))
  echo "$failures" > "$FAILURE_FILE"
  echo "[$(date -Is)] training failed after ${elapsed}s (exit $status; fast failure $failures of $FAST_FAILURE_LIMIT)"
  if [ "$failures" -ge "$FAST_FAILURE_LIMIT" ]; then
    rm -f "$FAILURE_FILE"
    stop_group "training keeps failing right after start"
  fi
fi
exit "$status"
