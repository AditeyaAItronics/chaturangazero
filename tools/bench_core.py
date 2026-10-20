"""
Core-speed benchmark (Phase 4 profiling harness, needtofix S1-S7).

Measures the pure-Python costs that dominate self-play — move generation,
terminal checks, encoding, and MCTS tree work (with a free dummy evaluator,
so the network is excluded) — plus memory per MCTS node. No training.

Usage:
    python tools/bench_core.py            # prints a table
    python tools/bench_core.py --json     # machine-readable
"""

import argparse
import json
import os
import sys
import time
import tracemalloc

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from chaturanga.board import State          # noqa: E402
from chaturanga.perft import perft          # noqa: E402
from search.mcts import search              # noqa: E402


def _positions(n: int, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    out, state = [], State()
    while len(out) < n:
        if state.is_terminal()[0] or state.ply_count > 120:
            state = State()
        out.append((state.board.copy(), state.side_to_move, state.ply_count))
        state = state.apply(int(rng.choice(state.legal_moves())))
    return out


def _fresh(positions):
    """New State objects, as MCTS creates for every node (no cached results)."""
    return [State(board=b, side_to_move=s, ply_count=p) for b, s, p in positions]


def _rate(fn, items, repeat: int = 3) -> float:
    best = float('inf')
    for _ in range(repeat):
        states = _fresh(items)
        t = time.perf_counter()
        for st in states:
            fn(st)
        best = min(best, time.perf_counter() - t)
    return len(items) / best


def _uniform(state):
    return np.full(4096, 1.0 / 4096, dtype=np.float32), 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--sims', type=int, default=400)
    args = parser.parse_args()

    positions = _positions(2000)
    results = {
        'legal_moves_per_s': _rate(lambda s: s.legal_moves(), positions),
        'is_terminal_per_s': _rate(lambda s: s.is_terminal(), positions),
        'encode_per_s': _rate(lambda s: s.encode(), positions),
    }

    roots = _fresh(_positions(5, seed=1)[1::1][:5])
    t = time.perf_counter()
    for r in roots:
        np.random.seed(0)
        search(r, _uniform, n_sims=args.sims, add_noise=False)
    results['mcts_sims_per_s'] = len(roots) * args.sims / (time.perf_counter() - t)

    tracemalloc.start()
    snap0 = tracemalloc.take_snapshot()
    np.random.seed(0)
    _, root = search(State(), _uniform, n_sims=args.sims, add_noise=False, return_root=True)
    snap1 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    def count(node):
        return 1 + sum(count(c) for c in node.children.values())
    nodes = count(root)
    grown = sum(s.size_diff for s in snap1.compare_to(snap0, 'filename'))
    results['tree_nodes'] = nodes
    results['bytes_per_node'] = grown / nodes

    t = time.perf_counter()
    perft(State(), 4)
    results['perft4_start_seconds'] = time.perf_counter() - t

    if args.json:
        print(json.dumps(results))
        return
    for k, v in results.items():
        print(f"{k:24s} {v:>14,.1f}")


if __name__ == '__main__':
    main()
