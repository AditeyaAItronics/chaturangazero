"""
Terminal condition detection for Chaturanga.

Terminal conditions evaluated in THIS EXACT ORDER (§2.1):
1. RAJA CAPTURED — S has no Raja → S loses (z = -1)
2. KING vs KING — both sides have only their Raja → draw (z = 0)
3. BARE KING — S has only their Raja, opponent has more → S loses (z = -1)
4. STALEMATE — S has no legal move → S WINS (z = +1)  ← THE SIGN TRAP
5. MOVE CAP — ply_count >= 300 → adjudicate on material

Order matters. Rule 2 must precede rule 3.
"""

from .board import State, WHITE, BLACK, RAJA, PIECE_VALUES, get_move_cap, get_adjudicate_margin


def is_terminal(state: State, adjudicate: bool = True) -> tuple[bool, float]:
    """
    Check if the position is terminal from the perspective of the side to move.

    Args:
        adjudicate: at the move cap, score on material (training, §2.2) if True;
            score as a draw if False. Arena and baseline games must use False —
            adjudication is a training-only device.

    Returns:
        (is_done, z) where z is the value for the side to move:
        +1 = side to move wins, -1 = side to move loses, 0 = draw.
    """
    done, z, _ = evaluate_terminal(state, adjudicate)
    return done, z


def terminal_reason(state: State) -> str:
    """
    Which rule ended the game: 'raja_captured', 'king_vs_king', 'bare_king',
    'stalemate', 'move_cap', or 'ongoing'. Same ladder as is_terminal().
    """
    return evaluate_terminal(state)[2]


def evaluate_terminal(state: State, adjudicate: bool = True) -> tuple[bool, float, str]:
    """The §2.1 terminal ladder, in order. Returns (done, z, reason)."""
    settings = (adjudicate, get_move_cap(), get_adjudicate_margin())
    cached = getattr(state, '_terminal', None)
    if cached is not None and cached[0] == settings:
        return cached[1]
    result = _evaluate_terminal(state, adjudicate)
    if hasattr(state, '_terminal'):
        state._terminal = (settings, result)
    return result


def _evaluate_terminal(state: State, adjudicate: bool) -> tuple[bool, float, str]:
    side = state.side_to_move
    opp = -side

    # 1. RAJA CAPTURED — side to move has no Raja
    if not state.has_raja(side):
        return (True, -1.0, 'raja_captured')

    # 2. KING vs KING — both sides have only their Raja
    side_count = state.piece_count(side)
    opp_count = state.piece_count(opp)

    if side_count == 1 and opp_count == 1:
        # Both have exactly one piece, which must be the Raja (since we passed rule 1)
        return (True, 0.0, 'king_vs_king')

    # 3. BARE KING — side to move has only their Raja, opponent has more
    if side_count == 1 and opp_count > 1:
        return (True, -1.0, 'bare_king')

    # 4. STALEMATE — side to move has no legal move → WINS (z = +1)
    # Import here to avoid circular imports
    from .moves import generate_moves
    moves = generate_moves(state)
    if len(moves) == 0:
        return (True, 1.0, 'stalemate')  # THE SIGN: +1 for the stalemated player

    # 5. MOVE CAP — training device
    if state.ply_count >= get_move_cap():
        if not adjudicate:
            return (True, 0.0, 'move_cap')
        z = _adjudicate_material(state)
        return (True, z, 'move_cap')

    return (False, 0.0, 'ongoing')


def _adjudicate_material(state: State) -> float:
    """
    Adjudicate on material when the move cap is reached.
    
    material = 5*Ratha + 3*Ashva + 1.5*Gaja + 1.5*Mantri + 1*Padati
    if |diff| >= margin (default 4): z = sign(diff) from side-to-move's perspective
    else: z = 0 (draw)
    """
    side = state.side_to_move
    opp = -side
    
    our_material = state.material_score(side)
    their_material = state.material_score(opp)
    
    diff = our_material - their_material
    
    if abs(diff) >= get_adjudicate_margin():
        return 1.0 if diff > 0 else -1.0
    else:
        return 0.0
