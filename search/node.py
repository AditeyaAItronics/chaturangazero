"""
MCTS Node — per-edge statistics stored as small arrays over the LEGAL moves.

Each expanded Node stores, for its k legal moves (edge i <-> actions[i]):
- N[i]: visit count
- W[i]: total value accumulated through edge i (from this node's side to move)
- P[i]: prior probability from the policy network

Sizing the arrays to k (~20-40 in Chaturanga) instead of the full 4096-action
space cuts a node from ~48 KB to well under 1 KB and lets PUCT run as one
vectorised numpy expression (needtofix S1). visit_distribution() still returns
a dense (action_size,) vector, so callers see the same interface as before.
"""

import numpy as np
from typing import Optional

_EMPTY = np.zeros(0, dtype=np.float64)
_EMPTY_ACTIONS = np.zeros(0, dtype=np.int64)


class Node:
    """
    MCTS tree node with per-edge statistics over its legal moves.
    """
    __slots__ = (
        'state', 'parent', 'parent_action', 'parent_edge', 'action_size',
        'actions', 'legal_actions', 'N', 'W', 'P',
        'children', 'is_expanded', 'is_terminal', 'terminal_value',
    )

    def __init__(self, state, parent: Optional['Node'] = None,
                 parent_action: int = -1, action_size: int = 4096,
                 parent_edge: int = -1):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.parent_edge = parent_edge      # index of the edge in parent's arrays
        self.action_size = action_size

        self.actions = _EMPTY_ACTIONS       # edge -> action id
        self.legal_actions: list[int] = []
        self.N = _EMPTY
        self.W = _EMPTY
        self.P = _EMPTY

        self.children: dict[int, 'Node'] = {}   # action id -> child
        self.is_expanded = False
        self.is_terminal = False
        self.terminal_value = 0.0

    def expand(self, legal_actions, priors) -> None:
        """Give the node its edges: legal actions and their (normalised) priors."""
        self.legal_actions = list(legal_actions)
        self.actions = np.asarray(self.legal_actions, dtype=np.int64)
        k = len(self.legal_actions)
        self.N = np.zeros(k, dtype=np.float64)
        self.W = np.zeros(k, dtype=np.float64)
        self.P = np.asarray(priors, dtype=np.float64).copy()
        self.is_expanded = True

    def set_terminal(self, z: float) -> None:
        self.is_terminal = True
        self.terminal_value = z
        self.is_expanded = True

    def add_root_noise(self, alpha: float, eps: float) -> None:
        """Mix Dirichlet(alpha) noise into the priors: P <- (1 - eps) P + eps noise."""
        if len(self.legal_actions):
            noise = np.random.dirichlet([alpha] * len(self.legal_actions))
            self.P = (1 - eps) * self.P + eps * noise

    def edge_of(self, action: int) -> int:
        """Edge index of a legal action."""
        return self.legal_actions.index(action)

    @property
    def total_visits(self) -> float:
        return float(self.N.sum())

    def Q(self, action: int) -> float:
        """Mean value for an action."""
        i = self.edge_of(action)
        if self.N[i] == 0:
            return 0.0
        return float(self.W[i] / self.N[i])

    def best_child_by_visits(self) -> int:
        """Return the action with the most visits (first one on ties)."""
        return int(self.actions[int(np.argmax(self.N))])

    def visit_distribution(self) -> np.ndarray:
        """
        Return visit counts as a distribution over the full action space.
        This becomes the training target π.
        """
        dense = np.zeros(self.action_size, dtype=np.float32)
        if len(self.actions):
            dense[self.actions] = self.N
        return dense
