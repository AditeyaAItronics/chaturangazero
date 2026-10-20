# Agent instructions

Standing instructions from the repo owner for any AI agent working in this repository.

## Every code change gets an end-to-end review

After **each** code change (one fix or one logical unit of work, not just at the end of a session):

1. **Run the tests.** Run the tests for the area you touched, then the full suite (`python -m pytest -q`). A change isn't done while any test fails.
2. **Review the change.** Run a code review of the diff (e.g. `/code-review`) for:
   - **correctness:** the fix works, and nothing upstream or downstream broke;
   - **end-to-end behaviour:** self-play → replay buffer → training → arena → checkpoint/resume still fit together;
   - **efficiency:** no new slowdowns, extra memory or repeated work on hot paths (MCTS, move generation, encoding, replay sampling).
3. **Fix what the review finds**, then repeat steps 1–2 until both are clean.
4. **Report** briefly what was changed, what the tests showed, and what the review found.

Don't batch several unrelated fixes under one review.

## Git

- **Never push** (or open a PR) unless the owner explicitly says so in the current conversation. Commit locally on a feature branch only.
- Don't commit scratch or comparison write-ups that are meant to stay local (e.g. the old-vs-new improvements report).

## Work order

Follow `needtofixagain.md`, "Suggested order of work": N1, N2 → N5/P5 → N4 → N6, N7, N11, N12 → the rest. P-items go into `IMPLEMENTATION_PLAN.md`.

## Rules that must not change silently

`RULES.md` is frozen (§2.1). Don't add king-safety filtering or an `is_square_attacked()` function. Stalemate is **+1** for the stalemated player. Material adjudication at the move cap is **training-only**.
