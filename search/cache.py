"""
Evaluation cache — the transposition table for MCTS (§5 Phase 4, needtofix M2).

Positions recur: by transposition inside one search tree, across successive
moves of a game, and across the many concurrent games of a self-play batch
(every game starts from the same position). Each recurrence would otherwise
cost a terminal check, a move generation and a network evaluation.

The cache maps a position to what expansion needs:
    terminal positions  -> ('terminal', z)
    other positions     -> ('eval', legal_actions, priors over legal_actions, value)

Keyed on State.key() plus the ply count, because the network input includes
the ply plane (§3.2) and the move cap depends on it. Entries are only valid
for one network, so create a fresh cache whenever the weights change.

Only evaluations are shared; each tree keeps its own visit statistics, so the
search stays a tree and backup is unchanged.
"""

from collections import OrderedDict
from typing import Optional

import numpy as np


class EvalCache:
    """LRU cache of expansion results, with hit-rate statistics."""

    def __init__(self, max_entries: int = 50_000):
        self.max_entries = max_entries
        self._data: "OrderedDict" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._data)

    @staticmethod
    def key(state):
        return (state.key(), getattr(state, 'ply_count', 0))

    def get(self, state) -> Optional[tuple]:
        k = self.key(state)
        entry = self._data.get(k)
        if entry is None:
            self.misses += 1
            return None
        self._data.move_to_end(k)
        self.hits += 1
        return entry

    def _put(self, state, entry: tuple):
        if self.max_entries <= 0:
            return
        k = self.key(state)
        self._data[k] = entry
        self._data.move_to_end(k)
        if len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def put_terminal(self, state, z: float):
        self._put(state, ('terminal', float(z)))

    def put_eval(self, state, legal_actions: list[int], priors: np.ndarray, value: float):
        """priors: probabilities for legal_actions, in the same order (already masked)."""
        self._put(state, ('eval', list(legal_actions),
                          np.asarray(priors, dtype=np.float32).copy(), float(value)))

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict:
        return {'eval_cache_hits': self.hits, 'eval_cache_misses': self.misses,
                'eval_cache_hit_rate': self.hit_rate, 'eval_cache_entries': len(self)}


def apply_entry(node, entry: tuple) -> float:
    """Expand `node` from a cache entry. Returns the node's value (side to move)."""
    if entry[0] == 'terminal':
        node.set_terminal(entry[1])
        return entry[1]
    _, legal, priors, value = entry
    node.expand(legal, priors)
    return value
