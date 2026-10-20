#!/usr/bin/env bash
# Run the training command; when it finishes the whole run, stop the VM for
# good (needtofixagain N4).
#
# The training loop exits 0 only when every iteration is done (a preemption
# exits 143, a crash non-zero). Without this, a finished run would keep a
# billed VM alive: the heartbeat watchdog would shut it down, the managed
# instance group would recreate it, and the cycle would repeat forever.
# So on exit 0 the instance group is resized to 0.
#
# Environment: CZ_MIG (instance group name) and CZ_ZONE. If CZ_MIG is unset
# (local use), nothing is resized.
#
# Usage: run_training.sh COMMAND [ARGS...]   (exit status is COMMAND's)
set -uo pipefail

"$@"
status=$?
if [ "$status" -eq 0 ] && [ -n "${CZ_MIG:-}" ]; then
  echo "[$(date -Is)] training run complete; resizing $CZ_MIG to 0"
  gcloud compute instance-groups managed resize "$CZ_MIG" --size 0 \
    --zone "${CZ_ZONE:?CZ_ZONE must be set with CZ_MIG}" --quiet \
    || echo "[$(date -Is)] WARNING: resize failed; delete $CZ_MIG by hand"
fi
exit "$status"
