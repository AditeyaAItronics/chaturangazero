"""
Exact Connect-4 solver for late positions (Phase 2 gate, needtofix M7).

Memoised negamax (full window, so every stored value is exact; it stops
early only on a found win, which is safe because +1 is the maximum). It is
plain Python, so it is meant for positions with roughly a dozen empty cells
or fewer, which is what the "MCTS with a perfect evaluator plays optimally"
test uses.

Values are game-theoretic outcomes from the side to move: +1 win, 0 draw,
-1 loss (no distance-to-win weighting).
"""

from .game import Connect4State


class Connect4Solver:
    def __init__(self):
        self._memo: dict = {}

    def value(self, state: Connect4State) -> int:
        """Exact value of `state` for the side to move."""
        return self._negamax(state)

    def move_values(self, state: Connect4State) -> dict[int, int]:
        """Exact value of each legal move, from the mover's perspective."""
        return {a: -self.value(state.apply(a)) for a in state.legal_moves()}

    def best_moves(self, state: Connect4State) -> set[int]:
        values = self.move_values(state)
        best = max(values.values())
        return {a for a, v in values.items() if v == best}

    def _negamax(self, state: Connect4State) -> int:
        done, z = state.is_terminal()
        if done:
            return int(z)
        key = state.key()
        if key in self._memo:
            return self._memo[key]
        best = -1
        # Centre columns first: finds wins sooner
        for a in sorted(state.legal_moves(), key=lambda c: abs(c - 3)):
            v = -self._negamax(state.apply(a))
            if v > best:
                best = v
                if best == 1:
                    break
        self._memo[key] = best
        return best
