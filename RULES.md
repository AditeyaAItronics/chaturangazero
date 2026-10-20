# Chaturanga — Frozen Ruleset

**Changes require all four students to sign off.**

---

## Board and Setup

8×8, files a–h, ranks 1–8. White moves first.

```
8  r  n  b  m  k  b  n  r     Black
7  p  p  p  p  p  p  p  p
6  .  .  .  .  .  .  .  .
5  .  .  .  .  .  .  .  .
4  .  .  .  .  .  .  .  .
3  .  .  .  .  .  .  .  .
2  P  P  P  P  P  P  P  P
1  R  N  B  M  K  B  N  R     White
   a  b  c  d  e  f  g  h

R/r Ratha   N/n Ashva   B/b Gaja
M/m Mantri  K/k Raja    P/p Padati
```

Raja opposes Raja on the e-file; Mantri opposes Mantri on the d-file. Mirror-symmetric.

## Movement

- **Raja** — one square in any of eight directions. **No castling.** May legally move onto a square attacked by an enemy piece.
- **Mantri** — exactly one square diagonally.
- **Ratha** — any number of squares orthogonally, blocked by the first occupied square; captures on it if enemy.
- **Gaja** — exactly two squares diagonally, **jumping**: the intervening square's contents are irrelevant. Cannot move one square. Destination must be empty or enemy-occupied.
- **Ashva** — standard knight leap.
- **Padati** — one square straight forward to an empty square. **No two-square opening move. No en passant.** Captures one square diagonally forward. On reaching the last rank, **promotes to Mantri**, always, with no choice.

All captures are by displacement. There are no restrictions on capturing any piece, including the Raja.

## There Is No Check

This is the defining rule of this variant and it must be implemented as an *absence*, not as a special case:

- A player **may** move a piece to a square where it can be captured, including the Raja.
- A player **may** leave their Raja attacked, and is under no obligation to respond.
- A player **may** move their Raja onto a currently guarded square. **The opponent may then capture it on their turn, and that ends the game.** The capture is not automatic — the opponent must actually play the capturing move. If they overlook it, play continues normally.
- There is **no checkmate**. The game ends when a Raja is **actually captured**.
- Consequently, **every pseudo-legal move is legal**. `legal_moves()` performs no king-safety filtering whatsoever.

## Terminal Conditions — Evaluated in This Exact Order

Checked **after** each move is applied, from the perspective of the player now to move (call them S):

```
1. RAJA CAPTURED
   S has no Raja on the board          →  S loses        (z = −1)

2. KING vs KING
   Both sides have only their Raja     →  draw           (z =  0)

3. BARE KING
   S has only their Raja, opponent
   has at least one other piece        →  S loses        (z = −1)

4. STALEMATE
   S has no legal move at all          →  S WINS         (z = +1)

5. MOVE CAP  (training device, not a rule)
   ply_count >= 300                    →  adjudicate, §2.2
```

Order matters. Rule 2 must precede rule 3, or King-vs-King resolves as a loss for whoever happens to be to move.

## Deviation Summary

| Rule | This project |
| --- | --- |
| Castling | **None** |
| Pawn double-step / en passant | **None** |
| Promotion | **Mantri only, forced** |
| Check / checkmate | **Does not exist** — Raja capture ends the game |
| Raja onto a guarded square | **Legal**; opponent may capture on their turn, not automatically |
| Stalemate | **Win for the side to move** (the stalemated player) |
| Repetition | **No rule** |
| 50-move | **No rule** |
| Bare king | **Loss**, except K-v-K → draw |

## Move Cap and Adjudication (§2.2)

There is **no rule that guarantees a game ends**: no repetition draw and no 50-move rule. The move cap is therefore the sole termination guarantee and is enforced as a hard invariant: applying a move from a position at the cap raises an error.

- **Cap:** `move_cap = 300` plies by default (`play.move_cap` in the config).
- **Training games** that reach the cap are adjudicated on material, from the perspective of the side to move:

  ```
  material = 5·Ratha + 3·Ashva + 1.5·Gaja + 1.5·Mantri + 1·Padati   (Raja not counted)
  diff     = our material − their material
  |diff| >= 4   →  z = sign(diff)     (decisive)
  otherwise     →  z = 0              (draw)
  ```

  The margin is `play.adjudicate_margin` (default 4.0).
- **Adjudication is a training-only device.** Arena, baseline and evaluation games never adjudicate: a game that reaches the cap is scored as a **draw**.
- **Monitoring:** log the fraction of self-play games ending at the cap every iteration. Under 15% is healthy. Above 25% means the agent has found the shuffling attractor. Respond in this order:
  1. tighten the cap to 200;
  2. lower the margin to 2.0;
  3. raise the root Dirichlet ε.

## Note: King-vs-King cannot arise in play

A player never loses a piece on their own move, and a side reduced to a bare Raja loses at the start of its turn (rule 3). So rule 2 (King-vs-King draw) can only fire in a position that was set up that way. It never fires during a game that starts from the opening position. This is tested in `tests/test_low_and_speed.py`.

## Sign-off

Any change to this file requires all four students to sign off (initials and date) below.

| Version | Change | Aditeya | Ansh | Adit | Sarthak |
| --- | --- | --- | --- | --- | --- |
| v2.1 | Initial frozen ruleset (§2.1), terminal ladder in order, §2.2 move cap and adjudication | | | | |
