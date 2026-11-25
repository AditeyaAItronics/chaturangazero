"""
Download the trained models into runs/chaturanga/ for the web app and evaluation.

The models are not in the GitHub repository (runs/ is git-ignored); they are on
Hugging Face, ansh-k03/chaturangazero-500. This fetches the 1,500-iteration run:

    best.pt              iteration 1,350 (best model; equal in strength to 870)
    metrics.json         all 1,500 iterations (fills the web app's model card)
    checkpoint_1499.pt   iteration 1,500 (latest)
    checkpoint_0869.pt   iteration 870, the previous best (for old-vs-new games)

Every file is checked against a pinned SHA-256, so a stale or corrupt download
is never used. Standard library only.

    python tools/fetch_models.py               # into runs/chaturanga/
    python tools/fetch_models.py --dest DIR
    python tools/fetch_models.py --force       # re-download even if present
"""

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request

REPO = "https://huggingface.co/ansh-k03/chaturangazero-500/resolve/main"

# (path on Hugging Face, local name, SHA-256)
FILES = [
    ("run-1500/best.pt", "best.pt",
     "c18d3b787b9283df8e735dbc37737310c8cf2b353fe3032a74a10fb0ed2469c0"),
    ("run-1500/metrics.json", "metrics.json",
     "1eeae015e384636e2469f6f456a1e4220f9782a6ad9600a74fa1dc71a53b9405"),
    ("run-1500/checkpoint_1499.pt", "checkpoint_1499.pt",
     "ea05af87a16c8b8ff7ff38ab596b302a0a25e7cd25172867c5a9b195c4e5b7d4"),
    ("run-1000/best.pt", "checkpoint_0869.pt",
     "b00a6cfdc4cf2c24df3b3d82fc267d5b28c7dd4f400b47f52c30fbf16282a3d8"),
]


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: str, expected: str) -> None:
    """Download to a temporary file, check the hash, then move it into place."""
    folder = os.path.dirname(dest)
    fd, tmp = tempfile.mkstemp(dir=folder, suffix=".part")
    os.close(fd)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                out.write(block)
        got = sha256_of(tmp)
        if got != expected:
            raise RuntimeError(f"checksum mismatch for {url}: got {got[:12]}..., expected {expected[:12]}...")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dest", default=os.path.join(root, "runs", "chaturanga"),
                        help="folder to put the files in (default: runs/chaturanga)")
    parser.add_argument("--force", action="store_true", help="re-download files that are already correct")
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    failed = 0
    for remote, local, expected in FILES:
        dest = os.path.join(args.dest, local)
        if not args.force and os.path.exists(dest) and sha256_of(dest) == expected:
            print(f"ok       {local} (already up to date)")
            continue
        if os.path.exists(dest):
            print(f"replace  {local} (outdated or different file)")
        print(f"download {remote} -> {local}", flush=True)
        try:
            fetch(f"{REPO}/{remote}", dest, expected)
            print(f"ok       {local}")
        except Exception as e:  # keep going; report at the end
            failed += 1
            print(f"FAILED   {local}: {e}", file=sys.stderr)

    if failed:
        print(f"\n{failed} file(s) failed; the others are in {args.dest}", file=sys.stderr)
        return 1
    print(f"\nAll files ready in {args.dest}. Start the web app with: python web/app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
