# ChaturangaZero training image (Phase 4b, needtofix I3).
#
# Alternative to the Deep Learning VM image used by infra/startup.sh — e.g. for
# a container-optimised VM, Vertex AI custom jobs, or a reproducible local run:
#   docker build -f infra/Dockerfile -t chaturangazero .
#   docker run --gpus all -v $PWD/runs:/app/runs chaturangazero \
#       --config configs/local.yaml --output runs/local --resume auto
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Runs the same entry point as the VM; extra arguments are appended
ENTRYPOINT ["python", "train_chaturanga.py"]
CMD ["--config", "configs/cloud.yaml", "--output", "runs/cloud", "--resume", "auto"]
