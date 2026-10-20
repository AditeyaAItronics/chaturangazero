"""
Device selection — one place that decides CPU vs GPU.

The same code runs unchanged on a CPU-only laptop and on a CUDA machine
(local RTX 3060 or a GCP T4). Resolution order:

    1. explicit argument      get_device('cuda:1') / --device flag
    2. environment variable   CHATURANGA_DEVICE=cpu
    3. config file            device: auto   (training loop only)
    4. 'auto'                 CUDA if available, else Apple MPS, else CPU

Asking for a GPU that is not there falls back to CPU with a warning rather
than crashing, so a config written for the cloud still runs locally.
"""

import os
import warnings
from typing import Optional, Union

import torch

ENV_VAR = 'CHATURANGA_DEVICE'

DeviceLike = Union[str, torch.device, None]


def _mps_available() -> bool:
    backend = getattr(torch.backends, 'mps', None)
    return backend is not None and backend.is_available()


def get_device(preference: DeviceLike = None) -> torch.device:
    """
    Resolve a device preference to a usable torch.device.

    Args:
        preference: 'auto', 'cpu', 'cuda', 'cuda:N', 'gpu', 'mps', a
                    torch.device, or None (read CHATURANGA_DEVICE, then auto).
    """
    if isinstance(preference, torch.device):
        preference = str(preference)
    if preference is None or str(preference).strip() == '':
        preference = os.environ.get(ENV_VAR, 'auto')

    pref = str(preference).strip().lower()
    if pref == 'gpu':
        pref = 'cuda'

    if pref == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if _mps_available():
            return torch.device('mps')
        return torch.device('cpu')

    if pref == 'cpu':
        return torch.device('cpu')

    if pref.startswith('cuda'):
        if not torch.cuda.is_available():
            warnings.warn(f"Device '{preference}' requested but CUDA is not available; using CPU.")
            return torch.device('cpu')
        device = torch.device(pref)
        if device.index is not None and device.index >= torch.cuda.device_count():
            warnings.warn(f"Device '{preference}' requested but only "
                          f"{torch.cuda.device_count()} GPU(s) found; using cuda:0.")
            return torch.device('cuda:0')
        return device

    if pref == 'mps':
        if not _mps_available():
            warnings.warn("Device 'mps' requested but MPS is not available; using CPU.")
            return torch.device('cpu')
        return torch.device('mps')

    raise ValueError(f"Unknown device '{preference}'. "
                     f"Use auto, cpu, cuda, cuda:N, gpu or mps.")


def use_amp(device: DeviceLike, requested: bool = True) -> bool:
    """FP16 autocast is only enabled on CUDA; it is slower or unsupported elsewhere."""
    return bool(requested) and get_device(device).type == 'cuda'


def describe_device(device: DeviceLike = None) -> str:
    """Human-readable one-liner for logs, e.g. 'cuda:0 (NVIDIA GeForce RTX 3060, 12.0 GB)'."""
    device = get_device(device)
    if device.type == 'cuda':
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        return f"cuda:{index} ({props.name}, {props.total_memory / 1024**3:.1f} GB)"
    if device.type == 'mps':
        return "mps (Apple GPU)"
    return f"cpu ({torch.get_num_threads()} threads)"


def add_device_argument(parser) -> None:
    """Add the shared --device flag to an argparse parser."""
    parser.add_argument('--device', type=str, default=None,
                        help="auto (default), cpu, cuda, cuda:N or mps. "
                             f"Falls back to ${ENV_VAR}, then auto-detect.")
