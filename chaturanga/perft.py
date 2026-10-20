"""
Perft (performance test) — enumerates all leaf nodes at depth N.

Used for differential testing against the reference engine.
"""

from .board import State
from .moves import generate_moves, apply_move
from .terminal import is_terminal


def perft(state: State, depth: int) -> int:
    """
    Count the number of leaf nodes at the given depth.
    
    At depth 0, returns 1 (the current position is a leaf).
    At depth N, generates all legal moves, applies each, and recurses.
    Terminal positions do not generate children.
    """
    if depth == 0:
        return 1
    
    done, _ = is_terminal(state)
    if done:
        return 0
    
    moves = generate_moves(state)
    if not moves:
        return 0
    if depth == 1:
        # Bulk count: every child is a depth-0 leaf (counted 1 even if terminal)
        return len(moves)

    count = 0
    for action in moves:
        child = apply_move(state, action)
        count += perft(child, depth - 1)

    return count


def perft_divide(state: State, depth: int) -> dict[int, int]:
    """
    Perft with move-by-move breakdown.
    
    Returns a dict mapping action -> node count, useful for debugging
    disagreements between engines.
    """
    if depth <= 0:
        return {}
    
    results = {}
    if is_terminal(state)[0]:
        return results
    moves = generate_moves(state)
    for action in moves:
        child = apply_move(state, action)
        results[action] = perft(child, depth - 1)
    
    return results


def run_perft_suite(positions: list[dict], max_depth: int = 4) -> bool:
    """
    Run perft on a list of test positions.
    
    Each entry should have:
        'fen': a custom FEN-like string or board setup
        'state': a State object
        'expected': dict mapping depth -> expected count
    
    Returns True if all pass.
    """
    all_pass = True
    for i, pos in enumerate(positions):
        state = pos['state']
        expected = pos.get('expected', {})
        name = pos.get('name', f'Position {i}')
        
        for d in range(1, max_depth + 1):
            if d in expected:
                result = perft(state, d)
                exp = expected[d]
                status = "PASS" if result == exp else "FAIL"
                if result != exp:
                    all_pass = False
                print(f"  {name} depth {d}: {result} (expected {exp}) [{status}]")
            else:
                result = perft(state, d)
                print(f"  {name} depth {d}: {result} (no reference)")
    
    return all_pass
