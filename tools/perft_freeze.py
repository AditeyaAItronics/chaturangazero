"""
Freeze the perft reference numbers (§2.3, needtofix M9).

For each of the 15 positions, computes perft at depths 1..N with BOTH the
production engine (chaturanga/perft.py) and the independent reference engine
(chaturanga/reference.py), fails loudly on any disagreement (with a per-move
breakdown), and writes the agreed numbers to tests/perft_reference.json.
That file is then the ground truth for the regression tests.

Usage:
    python tools/perft_freeze.py                      # depths 1-6, all positions
    python tools/perft_freeze.py --max-depth 4        # quick
    python tools/perft_freeze.py --only initial_position stalemate
    python tools/perft_freeze.py --workers 8          # processes for deep searches

Depths already in the JSON for an unchanged FEN are skipped, so an
interrupted run can simply be restarted.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from chaturanga.board import State, WHITE, BLACK, to_fen, from_fen      # noqa: E402
from chaturanga.moves import generate_moves                              # noqa: E402

JSON_PATH = os.path.join(ROOT, 'tests', 'perft_reference.json')

# name, description, FEN (ranks 8 -> 1; side to move; ply)
POSITIONS = [
    ("initial_position", "Standard Chaturanga starting position",
     "rnbmkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBMKBNR w 0"),
    ("pawn_promotion", "White Padati on d7 and Black Padati on c2, each one step from promoting to Mantri",
     "r5k1/3P4/8/8/8/8/2p5/4K2R w 0"),
    ("bare_king", "Black to move can capture White's last piece (Ratha takes the Ashva), leaving a bare king",
     "1r5k/6p1/8/8/8/1N6/8/K7 b 0"),
    ("king_vs_king", "Both sides have only their Raja: terminal draw at the root (rule 2 before rule 3)",
     "8/8/5k2/8/3K4/8/8/8 w 0"),
    ("stalemate", "White to move has no legal move: terminal WIN for White (+1), the sign trap",
     "7k/8/8/3p4/3P4/p1pP4/PBP5/KB6 w 0"),
    ("raja_on_guarded_square", "White Raja's only moves are onto squares Black guards; they must generate as legal",
     "1r2k3/8/8/7p/7P/8/7r/K7 w 0"),
    ("gaja_jump", "White Gaja on c1 can only move by jumping over occupied b2 and d2",
     "7k/p7/8/8/8/8/1P1p4/2B4K w 0"),
    ("elephant_orbits", "Gajas on all four file/rank-parity classes for both sides; orbits never interact",
     "2b1kb2/8/8/8/8/8/BB6/BB2K3 w 0"),
    ("lone_ratha_endgame", "Endgame: White Raja and lone Ratha against Black Raja and one Padati",
     "4k3/7p/8/8/8/8/8/R3K3 w 0"),
    ("complex_midgame_1", "Midgame after 20 random plies from the start (seed 11)",
     "rn1mkb2/pp2Bppr/b1pp1n1p/8/8/1P2P1PP/P1PP1P1R/RN1MKBN1 w 20"),
    ("complex_midgame_2", "Midgame after 30 random plies from the start (seed 23)",
     "r2mkbnr/2ppp1pp/pp6/NB1n1Np1/6b1/PP5P/2PPPPP1/R2MK2R w 30"),
    ("raja_capture_imminent", "White Ratha on e2 can capture the Black Raja on e7 immediately",
     "8/p3k3/8/8/8/8/4R3/K7 w 0"),
    ("all_pieces_present", "All 32 pieces on the board after 1. e3 f6 2. Nc3 Nc6 (no captures)",
     "r1bmkbnr/ppppp1pp/2n2p2/8/8/2N1P3/PPPP1PPP/R1BMKBNR w 4"),
    ("padati_captures", "Several Padati diagonal-capture choices for both sides",
     "4k3/8/8/1p1p1p1p/2P1P1P1/8/8/4K3 w 0"),
    ("ratha_open_file", "Rathas facing each other on the open d-file",
     "3rk3/ppp2ppp/8/8/8/8/PPP2PPP/3RK3 w 0"),
]


def _check_position_claims():
    """Fail early if a hand-built position doesn't have the property its name claims."""
    pos = {name: from_fen(fen) for name, _, fen in POSITIONS}
    for name, _, fen in POSITIONS:
        assert to_fen(pos[name]) == fen, f"{name}: FEN does not round-trip"
    assert pos['stalemate'].is_terminal() == (True, 1.0)
    assert pos['king_vs_king'].is_terminal() == (True, 0.0)
    assert not pos['bare_king'].is_terminal()[0]
    assert len(pos['all_pieces_present'].board.nonzero()[0]) == 32

    # Raja's moves all land on squares Black attacks
    s = pos['raja_on_guarded_square']
    black_view = State(board=s.board, side_to_move=BLACK)
    attacked = {a % 64 for a in generate_moves(black_view)}
    raja_sq = int((s.board == 6).nonzero()[0][0])
    raja_moves = [a for a in generate_moves(s) if a // 64 == raja_sq]
    assert raja_moves and all(a % 64 in attacked for a in raja_moves)

    # Gaja's moves: exactly the two jumps
    s = pos['gaja_jump']
    gaja = [a % 64 for a in generate_moves(s) if a // 64 == 2]     # c1
    assert sorted(gaja) == [16, 20], gaja                          # a3, e3

    # Imminent Raja capture is legal
    s = pos['raja_capture_imminent']
    assert any(s.board[a % 64] == -6 for a in generate_moves(s))


def _subtree(job):
    """Perft of one root move's subtree, with one engine. Top-level for 'spawn'."""
    engine, fen, action, depth = job
    state = from_fen(fen)
    if engine == 'prod':
        from chaturanga.perft import perft
        from chaturanga.moves import apply_move
        return action, perft(apply_move(state, action), depth - 1)
    from chaturanga.reference import ref_perft, ref_apply_move
    return action, ref_perft(ref_apply_move(state, action), depth - 1)


def perft_divide_parallel(engine: str, fen: str, depth: int, pool) -> dict:
    """Per-root-move counts for one engine; the two engines must agree move by move."""
    state = from_fen(fen)
    if engine == 'prod':
        from chaturanga.terminal import is_terminal
        terminal, moves = is_terminal(state)[0], generate_moves(state)
    else:
        from chaturanga.reference import ref_is_terminal, ref_generate_moves
        terminal, moves = ref_is_terminal(state)[0], ref_generate_moves(state)
    if terminal:
        return {}
    if depth == 1:
        return {a: 1 for a in moves}
    jobs = [(engine, fen, a, depth) for a in moves]
    if pool is None:
        return dict(map(_subtree, jobs))
    return dict(pool.imap_unordered(_subtree, jobs))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--only', nargs='*', default=None, help='position names to (re)compute')
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--parallel-from-depth', type=int, default=5,
                        help='use the process pool at this depth and deeper')
    args = parser.parse_args()

    _check_position_claims()

    existing = {}
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, encoding='utf-8') as f:
            for p in json.load(f).get('positions', []):
                existing[p['name']] = p

    from chaturanga.terminal import is_terminal
    from chaturanga.reference import ref_is_terminal

    out = []
    pool = mp.get_context('spawn').Pool(args.workers) if args.workers > 1 else None
    try:
        for name, description, fen in POSITIONS:
            old = existing.get(name, {})
            expected = dict(old.get('expected', {})) if old.get('fen') == fen else {}
            state = from_fen(fen)
            prod_t, ref_t = is_terminal(state), ref_is_terminal(state)
            assert prod_t == ref_t, f"{name}: root terminal disagrees: prod {prod_t} ref {ref_t}"
            entry = {'name': name, 'description': description, 'fen': fen,
                     'root_terminal': {'done': prod_t[0], 'z': prod_t[1]}, 'expected': expected}
            out.append(entry)

            if args.only is not None and name not in args.only:
                continue
            for depth in range(1, args.max_depth + 1):
                if str(depth) in expected:
                    continue
                t0 = time.time()
                use_pool = pool if depth >= args.parallel_from_depth else None
                prod = perft_divide_parallel('prod', fen, depth, use_pool)
                ref = perft_divide_parallel('ref', fen, depth, use_pool)
                if prod != ref:
                    diff = {a: (prod.get(a), ref.get(a)) for a in set(prod) | set(ref)
                            if prod.get(a) != ref.get(a)}
                    sys.exit(f"MISMATCH {name} depth {depth}: total prod={sum(prod.values())} "
                             f"ref={sum(ref.values())}; per-move (prod, ref): {diff}")
                expected[str(depth)] = sum(prod.values())
                print(f"{name:24s} depth {depth}: {expected[str(depth)]:>13,}  "
                      f"({time.time() - t0:.1f}s, engines agree)", flush=True)
                _write(out + [_carry(existing, n, d, f) for n, d, f in POSITIONS
                              if n not in {e['name'] for e in out}])
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    _write(out)
    print(f"Wrote {JSON_PATH}")


def _carry(existing, name, description, fen):
    old = existing.get(name, {})
    state = from_fen(fen)
    done, z = state.is_terminal()
    return {'name': name, 'description': description, 'fen': fen,
            'root_terminal': {'done': done, 'z': z},
            'expected': dict(old.get('expected', {})) if old.get('fen') == fen else {}}


def _write(positions):
    data = {
        '_comment': 'Perft reference numbers, frozen by tools/perft_freeze.py. Production and '
                    'reference engines agree exactly at every listed depth (§2.3).',
        '_semantics': 'perft(s, 0) = 1; a terminal position at depth >= 1 counts 0. '
                      'Positions are FENs: ranks 8->1, side to move, ply count.',
        'positions': positions,
    }
    tmp = JSON_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
    os.replace(tmp, JSON_PATH)


if __name__ == '__main__':
    main()
