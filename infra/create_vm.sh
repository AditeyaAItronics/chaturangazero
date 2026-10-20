#!/usr/bin/env bash
# Create the spot training VM as a 1-instance managed instance group (MIG)
# (Phase 4b, needtofix I3).
#
# Why a MIG: when a spot VM is preempted, the MIG automatically creates a
# replacement. Its startup script (infra/startup.sh) restores the run from
# the bucket and resumes, so training continues unattended.
#
# Prerequisites (do these first, §2.4):
#   1. Upgrade to a paid account (keeps the $300 credit) and request
#      GPUS_ALL_REGIONS quota >= 1, plus NVIDIA_T4_GPUS in the region.
#   2. gcloud auth login && gcloud config set project YOUR_PROJECT
#   3. gcloud storage buckets create gs://YOUR_BUCKET --location=REGION
#   4. The repo must be reachable by the VM (public, or use a deploy token URL).
#
# Usage:
#   BUCKET=gs://chaturangazero REPO=https://github.com/OWNER/chaturangazero.git \
#     bash infra/create_vm.sh
# Stop and delete everything (checkpoints stay in the bucket):
#   gcloud compute instance-groups managed delete "$NAME-mig" --zone "$ZONE"
#   gcloud compute instance-templates delete "$NAME-template"
set -euo pipefail

NAME="${NAME:-cz-train}"
ZONE="${ZONE:-us-central1-a}"
MACHINE="${MACHINE:-n1-standard-16}"      # plan §2.4: many vCPUs, modest GPU
GPU="${GPU:-nvidia-tesla-t4}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-latest-gpu}"   # Deep Learning VM (CUDA + PyTorch)
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
DISK_GB="${DISK_GB:-100}"
BUCKET="${BUCKET:?set BUCKET=gs://your-bucket}"
REPO="${REPO:?set REPO=https://github.com/owner/repo.git}"
BRANCH="${BRANCH:-main}"
RUN="${RUN:-main}"
CONFIG="${CONFIG:-configs/cloud.yaml}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gcloud compute instance-templates create "$NAME-template" \
  --machine-type="$MACHINE" \
  --accelerator="type=$GPU,count=1" \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --maintenance-policy=TERMINATE \
  --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="${DISK_GB}GB" \
  --scopes=storage-rw \
  --metadata=install-nvidia-driver=True,cz-repo="$REPO",cz-branch="$BRANCH",cz-bucket="$BUCKET",cz-run="$RUN",cz-config="$CONFIG" \
  --metadata-from-file=startup-script="$HERE/startup.sh"

gcloud compute instance-groups managed create "$NAME-mig" \
  --zone="$ZONE" \
  --template="$NAME-template" \
  --size=1

echo "Created $NAME-mig in $ZONE. Follow progress with:"
echo "  gcloud storage cat $BUCKET/$RUN/metrics.json | tail"
echo "  gcloud compute ssh \$(gcloud compute instance-groups managed list-instances $NAME-mig --zone $ZONE --format='value(instance)') --zone $ZONE -- tail -f /var/log/chaturangazero.log"
