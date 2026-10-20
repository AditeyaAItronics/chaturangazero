"""
Monte Carlo Tree Search with PUCT selection.

Key design decisions:
- evaluate is a CALLABLE, not a network — this lets the batched inference
  server drop in without changing search code (§3.4).
- Backup flips signs correctly: parent sees -child_value.
- Stalemate returns +1 to the stalemated player (the sign trap from §2.1).
- Dirichlet noise at the root for exploration.
- Temperature schedule for move selection.
"""

import numpy as np
import math
from typing import Callable, Optional
from .node import Node


def search(
    state,
    evaluate: Callable,
    n_sims: int,
    action_size: int = 4096,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    add_noise: bool = True,
    root_node: Optional[Node] = None,
) -> np.ndarray:
    """
    Run MCTS from the given state.
    
    Args:
        state: Game state (Chaturanga State or Connect4State).
        evaluate: Callable that takes a state and returns (policy, value).
                  policy: np.ndarray of shape (action_size,) — prior probabilities.
                  value: float in [-1, +1] from side-to-move's perspective.
        n_sims: Number of simulations to run.
        action_size: Size of the action space.
        c_puct: Exploration constant.
        dirichlet_alpha: Dirichlet noise parameter.
        dirichlet_eps: Weight of Dirichlet noise at root.
        add_noise: Whether to add Dirichlet noise at root.
        root_node: Optional existing node for subtree reuse.
    
    Returns:
        visit_counts: np.ndarray of shape (action_size,) — raw visit counts.
    """
    # Create or reuse root
    if root_node is not None and root_node.is_expanded:
        root = root_node
        root.parent = None
        root.parent_action = -1
    else:
        root = Node(state, action_size=action_size)
        _expand(root, evaluate, action_size)
    
    # Add Dirichlet noise at the root
    if add_noise and root.legal_actions:
        noise = np.random.dirichlet(
            [dirichlet_alpha] * len(root.legal_actions)
        )
        for i, action in enumerate(root.legal_actions):
            root.P[action] = (1 - dirichlet_eps) * root.P[action] + dirichlet_eps * noise[i]
    
    # Run simulations
    for _ in range(n_sims):
        node = root
        
        # 1. SELECT — traverse tree using PUCT until we reach an unexpanded node
        while node.is_expanded and not node.is_terminal:
            action = _select_action(node, c_puct)
            if action in node.children:
                node = node.children[action]
            else:
                # Create child and expand
                child_state = _apply_action(node.state, action)
                child = Node(child_state, parent=node, parent_action=action,
                           action_size=action_size)
                node.children[action] = child
                node = child
                break
        
        # 2. EXPAND & EVALUATE
        if not node.is_expanded and not node.is_terminal:
            value = _expand(node, evaluate, action_size)
        elif node.is_terminal:
            value = node.terminal_value
        else:
            # Already expanded, evaluate leaf
            value = 0.0
        
        # 3. BACKUP — propagate value up the tree with sign flipping
        _backup(node, value)
    
    return root.visit_distribution()


def _expand(node: Node, evaluate: Callable, action_size: int) -> float:
    """
    Expand a node: check terminal, get policy/value from evaluator.
    Returns the value of this node from the side-to-move's perspective.
    """
    state = node.state
    
    # Check terminal
    done, z = state.is_terminal()
    if done:
        node.is_terminal = True
        node.terminal_value = z
        node.is_expanded = True
        return z
    
    # Get legal moves
    if hasattr(state, 'legal_moves'):
        node.legal_actions = state.legal_moves()
    else:
        # Chaturanga uses generate_moves
        from chaturanga.moves import generate_moves
        node.legal_actions = generate_moves(state)
    
    if not node.legal_actions:
        # No legal moves — this is stalemate, should be caught by is_terminal
        node.is_terminal = True
        node.terminal_value = 1.0  # Stalemate = win for stalemated player
        node.is_expanded = True
        return 1.0
    
    # Evaluate with the network
    policy, value = evaluate(state)
    
    # Mask illegal actions and renormalize
    legal_mask = np.zeros(action_size, dtype=bool)
    for a in node.legal_actions:
        legal_mask[a] = True
    
    # Apply mask
    policy = policy * legal_mask
    policy_sum = np.sum(policy)
    if policy_sum > 0:
        policy = policy / policy_sum
    else:
        # Uniform over legal moves if policy assigns zero to all legal
        policy = legal_mask.astype(np.float32) / len(node.legal_actions)
    
    node.P[:] = policy
    node.is_expanded = True
    
    return value


def _select_action(node: Node, c_puct: float) -> int:
    """
    Select action using PUCT formula:
    
    a* = argmax_a [ Q(s,a) + c_puct * P(s,a) * sqrt(sum_N) / (1 + N(s,a)) ]
    """
    total_n = node.total_visits
    # max(.., 1): with 0 visits sqrt(0) would zero the prior term, every score
    # would tie at 0 and the first visit would always go to legal_actions[0].
    sqrt_total = math.sqrt(max(total_n, 1.0))
    
    best_score = -float('inf')
    best_action = node.legal_actions[0]
    
    for a in node.legal_actions:
        q = node.Q(a)
        prior = node.P[a]
        exploration = c_puct * prior * sqrt_total / (1.0 + node.N[a])
        score = q + exploration
        
        if score > best_score:
            best_score = score
            best_action = a
    
    return best_action


def _backup(node: Node, value: float):
    """
    Backup value through the tree.
    
    CRITICAL: sign flips at each level.
    Parent sees -value because the child's value is from the child's
    side-to-move perspective, and the parent is the opposite side.
    """
    current = node
    v = value
    while current.parent is not None:
        action = current.parent_action
        parent = current.parent
        # Parent sees negated value
        v = -v
        parent.N[action] += 1
        parent.W[action] += v
        current = parent


def _apply_action(state, action: int):
    """Apply an action to a state, handling both Chaturanga and Connect-4."""
    if hasattr(state, 'apply'):
        return state.apply(action)
    else:
        from chaturanga.moves import apply_move
        return apply_move(state, action)


def select_action_with_temperature(
    visit_counts: np.ndarray,
    temperature: float,
    legal_actions: list[int] = None,
) -> tuple[int, np.ndarray]:
    """
    Select an action from visit counts using temperature.
    
    temperature = 1.0: proportional to visits (exploration)
    temperature -> 0:  argmax (exploitation)
    
    Returns:
        (selected_action, policy_target)
    
    The policy_target is the normalized visit distribution (training target π).
    """
    if legal_actions is not None:
        counts = np.zeros_like(visit_counts)
        for a in legal_actions:
            counts[a] = visit_counts[a]
    else:
        counts = visit_counts.copy()
    
    total = np.sum(counts)
    if total == 0:
        # Fallback: uniform over legal actions
        if legal_actions:
            action = np.random.choice(legal_actions)
            pi = np.zeros_like(counts, dtype=np.float32)
            for a in legal_actions:
                pi[a] = 1.0 / len(legal_actions)
            return action, pi
        return 0, counts.astype(np.float32)
    
    # Normalize to get π (training target)
    pi = counts / total
    
    if temperature < 1e-8:
        # Deterministic: pick the most-visited
        action = int(np.argmax(counts))
        # π is still the soft distribution for training
        return action, pi.astype(np.float32)
    
    if temperature == 1.0:
        probs = pi
    else:
        # Apply temperature
        log_counts = np.zeros_like(counts, dtype=np.float64)
        nonzero = counts > 0
        log_counts[nonzero] = np.log(counts[nonzero]) / temperature
        log_counts -= np.max(log_counts[nonzero]) if np.any(nonzero) else 0
        exp_counts = np.zeros_like(counts, dtype=np.float64)
        exp_counts[nonzero] = np.exp(log_counts[nonzero])
        probs = exp_counts / np.sum(exp_counts)
    
    action = int(np.random.choice(len(probs), p=probs))
    
    return action, pi.astype(np.float32)


def get_subtree(root: Node, action: int) -> Optional[Node]:
    """
    Extract the subtree rooted at the child reached by the given action.
    Returns None if the child doesn't exist.
    Used for subtree reuse between moves.
    """
    if action in root.children:
        child = root.children[action]
        child.parent = None
        child.parent_action = -1
        return child
    return None
