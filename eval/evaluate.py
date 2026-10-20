"""
Phase 6 evaluation toolkit (needtofix I5).

Every result the plan asks for in Part 8 "Results", from checkpoints and
metrics.json. The game (Chaturanga / Connect-4) and the network architecture
are read from each checkpoint.

    ladder    RUN_DIR      Elo across checkpoints (each plays its next K)  -> elo.json, elo_curve.png
    baselines CKPT         win rates vs random, greedy and 3-ply alpha-beta -> baselines.json
    sims      CKPT         sims-vs-strength at fixed weights (10/30/100/400) -> sims_strength.json/.png
    match     CKPT CKPT    head-to-head, e.g. final nets of the mirror ablation runs
    plots     RUN_DIR      loss split, value head, game length, cap rate, terminal reasons,
                           missed Raja captures -> PNGs in RUN_DIR/plots
    game      CKPT         one annotated sample game (vs alpha-beta by default) -> sample_game.md

All arena games use shared random openings per pair of games, argmax play and
no material adjudication (§2.2). Examples:
    python eval/evaluate.py ladder runs/chaturanga --games 40 --sims 50
    python eval/evaluate.py baselines runs/chaturanga/best.pt --games 40
    python eval/evaluate.py plots runs/chaturanga
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np   # noqa: E402

from utils.device import get_device, add_device_argument   # noqa: E402


# --------------------------------------------------------------------------- helpers

def load(path, device):
    """(net, game name, state factory, action size) for a checkpoint."""
    from net.model import load_net_from_checkpoint
    net, _ = load_net_from_checkpoint(path, device=device)
    if net.config['input_planes'] == 3:
        from connect4.game import Connect4State
        return net, 'connect4', Connect4State, net.config['action_size']
    from chaturanga.board import State
    return net, 'chaturanga', State, net.config['action_size']


def mcts_player(net, device, n_sims, action_size):
    from net.train import create_evaluator
    from search.mcts import search
    evaluate = create_evaluator(net, device=device, action_size=action_size)

    def player(state):
        return int(np.argmax(search(state, evaluate, n_sims=n_sims,
                                    action_size=action_size, add_noise=False,
                                    adjudicate=False)))
    return player


def _save(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
    print(f"wrote {path}")


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


# --------------------------------------------------------------------------- commands

def cmd_ladder(args, device):
    """Elo learning curve: every checkpoint plays the next --neighbors checkpoints."""
    from eval.elo import compute_elo
    from pipeline.inference_server import run_batched_arena
    from pipeline.resume import list_checkpoints

    ckpts = list_checkpoints(args.run_dir)[::args.every]
    if len(ckpts) < 2:
        sys.exit("need at least two checkpoints")
    nets = {it: load(path, device) for it, path in ckpts}
    results = []
    for i, (it_a, _) in enumerate(ckpts):
        for it_b, _ in ckpts[i + 1:i + 1 + args.neighbors]:
            net_a, _, create_state, action_size = nets[it_a]
            stats = run_batched_arena(net_a, nets[it_b][0], create_state, n_games=args.games,
                                      n_sims=args.sims, action_size=action_size, device=device,
                                      opening_plies=args.opening_plies)
            for r in stats['results']:
                results.append({'player1': f'iter_{it_a}', 'player2': f'iter_{it_b}',
                                'score': (r + 1) / 2})
            print(f"  iter {it_a} vs iter {it_b}: {stats['p1_wins']}-{stats['draws']}-{stats['p2_wins']}")
    ratings = compute_elo(results)
    curve = [(it, ratings[f'iter_{it}']) for it, _ in ckpts]
    _save({'ratings': ratings, 'curve': curve, 'games_per_pair': args.games, 'sims': args.sims},
          os.path.join(args.run_dir, 'elo.json'))
    plt = _plt()
    plt.figure(figsize=(8, 5))
    plt.plot([c[0] for c in curve], [c[1] for c in curve], 'b-o', markersize=3)
    plt.xlabel('Iteration'); plt.ylabel('Elo'); plt.title('Learning curve (Elo)'); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(args.run_dir, 'elo_curve.png'), dpi=150, bbox_inches='tight')
    print(f"wrote {os.path.join(args.run_dir, 'elo_curve.png')}")


def cmd_baselines(args, device):
    from search.arena import arena
    from search.baselines import random_player, greedy_material_player, alpha_beta_player
    net, game, create_state, action_size = load(args.checkpoint, device)
    agent = mcts_player(net, device, args.sims, action_size)
    opponents = {'random': random_player, 'greedy': greedy_material_player}
    if game == 'chaturanga':
        opponents['alpha_beta_3ply'] = lambda s: alpha_beta_player(s, depth=3)
    out = {}
    for name, opp in opponents.items():
        stats = arena(agent, opp, create_state, n_games=args.games, action_size=action_size,
                      opening_plies=args.opening_plies)
        out[name] = {k: stats[k] for k in ('p1_wins', 'draws', 'p2_wins', 'p1_win_rate')}
        print(f"  vs {name}: {stats['p1_wins']}W-{stats['draws']}D-{stats['p2_wins']}L "
              f"(score {stats['p1_win_rate']:.1%})")
    _save({'checkpoint': args.checkpoint, 'sims': args.sims, 'games': args.games, 'results': out},
          args.out or os.path.join(os.path.dirname(args.checkpoint) or '.', 'baselines.json'))


def cmd_sims(args, device):
    """Same weights, different search budgets: Elo per budget (search buys strength)."""
    from eval.elo import compute_elo
    from pipeline.inference_server import run_batched_arena
    net, _, create_state, action_size = load(args.checkpoint, device)
    levels = sorted(args.levels)
    results = []
    for i, a in enumerate(levels):
        for b in levels[i + 1:]:
            stats = run_batched_arena(net, net, create_state, n_games=args.games, n_sims=a,
                                      n_sims_b=b, action_size=action_size, device=device,
                                      opening_plies=args.opening_plies)
            for r in stats['results']:
                results.append({'player1': f'{a}', 'player2': f'{b}', 'score': (r + 1) / 2})
            print(f"  {a} sims vs {b} sims: {stats['p1_wins']}-{stats['draws']}-{stats['p2_wins']}")
    ratings = compute_elo(results)
    curve = [(lvl, ratings[str(lvl)]) for lvl in levels]
    base = os.path.join(os.path.dirname(args.checkpoint) or '.', 'sims_strength')
    _save({'checkpoint': args.checkpoint, 'curve': curve, 'games_per_pair': args.games}, base + '.json')
    plt = _plt()
    plt.figure(figsize=(7, 4.5))
    plt.semilogx([c[0] for c in curve], [c[1] for c in curve], 'g-o')
    plt.xlabel('MCTS simulations per move (log)'); plt.ylabel('Elo'); plt.grid(alpha=0.3)
    plt.title('Sims vs strength at fixed weights')
    plt.savefig(base + '.png', dpi=150, bbox_inches='tight')
    print(f"wrote {base}.png")


def cmd_match(args, device):
    from pipeline.inference_server import run_batched_arena
    net_a, _, create_state, action_size = load(args.checkpoint_a, device)
    net_b = load(args.checkpoint_b, device)[0]
    stats = run_batched_arena(net_a, net_b, create_state, n_games=args.games, n_sims=args.sims,
                              action_size=action_size, device=device, opening_plies=args.opening_plies)
    print(f"A {stats['p1_wins']}W-{stats['draws']}D-{stats['p2_wins']}L B (A score {stats['p1_win_rate']:.1%})")
    if args.out:
        _save({'a': args.checkpoint_a, 'b': args.checkpoint_b, **{k: v for k, v in stats.items()}}, args.out)


def cmd_plots(args, device):
    with open(os.path.join(args.run_dir, 'metrics.json'), encoding='utf-8') as f:
        metrics = json.load(f)
    out_dir = os.path.join(args.run_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)
    plt = _plt()
    it = [m['iteration'] for m in metrics]

    def series(key):
        return [m.get(key, np.nan) for m in metrics]

    def single(name, title, keys, ylabel):
        plt.figure(figsize=(8, 4.5))
        for key, label in keys:
            plt.plot(it, series(key), label=label)
        plt.xlabel('Iteration'); plt.ylabel(ylabel); plt.title(title); plt.grid(alpha=0.3)
        if len(keys) > 1:
            plt.legend()
        path = os.path.join(out_dir, name)
        plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
        print(f"wrote {path}")

    single('loss.png', 'Training loss (policy / value split)',
           [('mean_policy_loss', 'policy'), ('mean_value_loss', 'value')], 'loss')
    single('value_head.png', 'Value head', [('value_calibration_error', 'calibration error'),
                                            ('value_sign_accuracy', 'sign accuracy'),
                                            ('value_mean_abs', 'mean |v|')], '')
    single('game_length.png', 'Mean game length', [('avg_plies', 'plies')], 'plies')
    single('cap_and_draws.png', 'Cap rate and draw rate (watch for the shuffling attractor)',
           [('cap_rate', 'move-cap rate'), ('draw_rate', 'draw rate')], 'fraction of games')
    single('missed_raja.png', 'Missed free Raja captures', [('raja_capture_miss_rate', 'miss rate')], 'fraction')
    single('throughput.png', 'Self-play throughput', [('games_per_hour', 'games/hour')], 'games/hour')

    # Terminal reasons over training (stacked shares)
    reasons = sorted({r for m in metrics for r in m.get('terminal_stats', {})})
    if reasons:
        totals = np.array([max(sum(m.get('terminal_stats', {}).values()), 1) for m in metrics], float)
        shares = [np.array([m.get('terminal_stats', {}).get(r, 0) for m in metrics]) / totals for r in reasons]
        plt.figure(figsize=(8, 4.5))
        plt.stackplot(it, shares, labels=reasons)
        plt.xlabel('Iteration'); plt.ylabel('share of games'); plt.ylim(0, 1)
        plt.title('How games end, across training'); plt.legend(loc='upper left', fontsize=8)
        path = os.path.join(out_dir, 'terminal_reasons.png')
        plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
        print(f"wrote {path}")


def _move_name(state, action, game):
    if game == 'connect4':
        return f"col {action}"
    from chaturanga.board import PIECE_NAMES, PADATI, rank_of, file_of
    f, t = action // 64, action % 64
    sq = lambda s: f"{'abcdefgh'[file_of(s)]}{rank_of(s) + 1}"
    piece = int(state.board[f])
    name = PIECE_NAMES[abs(piece)].upper()
    capture = 'x' if state.board[t] != 0 else '-'
    last = 7 if state.side_to_move == 1 else 0
    promo = '=M' if abs(piece) == PADATI and rank_of(t) == last else ''
    return f"{'' if abs(piece) == PADATI else name}{sq(f)}{capture}{sq(t)}{promo}"


def cmd_game(args, device):
    """One annotated game: the agent's value estimate and top candidate moves at each of its turns."""
    from net.train import create_evaluator
    from search.mcts import search
    from search.baselines import alpha_beta_player, random_player
    net, game, create_state, action_size = load(args.checkpoint, device)
    evaluate = create_evaluator(net, device=device, action_size=action_size)
    opponent = (lambda s: alpha_beta_player(s, depth=3)) if (game == 'chaturanga' and args.opponent == 'alpha_beta') \
        else random_player
    np.random.seed(args.seed)
    state = create_state()
    agent_side = -state.side_to_move if args.agent_second else state.side_to_move
    lines = [f"# Annotated sample game", "",
             f"Agent: `{args.checkpoint}` ({args.sims} sims) as "
             f"{'second' if args.agent_second else 'first'} player, vs {args.opponent}.", "",
             "| Ply | Side | Move | Agent's value v | Agent's top candidates (visit share) |",
             "|---|---|---|---|---|"]
    while not state.is_terminal(adjudicate=False)[0]:
        if state.side_to_move == agent_side:
            visits, root = search(state, evaluate, n_sims=args.sims, action_size=action_size,
                                  add_noise=False, return_root=True, adjudicate=False)
            action = int(np.argmax(visits))
            _, v = evaluate(state)
            top = np.argsort(visits)[::-1][:3]
            total = max(visits.sum(), 1)
            cands = ', '.join(f"{_move_name(state, int(a), game)} {visits[a] / total:.0%}"
                              for a in top if visits[a] > 0)
            lines.append(f"| {state.ply_count + 1} | agent | {_move_name(state, action, game)} | {v:+.2f} | {cands} |")
        else:
            action = int(opponent(state))
            lines.append(f"| {state.ply_count + 1} | {args.opponent} | {_move_name(state, action, game)} | | |")
        state = state.apply(action)
    done, z = state.is_terminal(adjudicate=False)
    winner = 'draw' if z == 0 else ('agent' if (z > 0) == (state.side_to_move == agent_side) else args.opponent)
    from pipeline.selfplay import _classify_terminal
    lines += ["", f"**Result:** {winner} ({_classify_terminal(state, z)}, {state.ply_count} plies)", "",
              "Final position:", "", "```", state.display(), "```"]
    path = args.out or os.path.join(os.path.dirname(args.checkpoint) or '.', 'sample_game.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"wrote {path} — {winner}")


# --------------------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_device_argument(p)
    sub = p.add_subparsers(dest='command', required=True)

    def common(sp, games=40, sims=50):
        sp.add_argument('--games', type=int, default=games, help='games per pairing (even)')
        sp.add_argument('--sims', type=int, default=sims)
        sp.add_argument('--opening-plies', type=int, default=4)

    sp = sub.add_parser('ladder'); sp.add_argument('run_dir'); common(sp, games=20)
    sp.add_argument('--neighbors', type=int, default=3); sp.add_argument('--every', type=int, default=1)
    sp = sub.add_parser('baselines'); sp.add_argument('checkpoint'); common(sp); sp.add_argument('--out')
    sp = sub.add_parser('sims'); sp.add_argument('checkpoint'); common(sp, games=20)
    sp.add_argument('--levels', type=int, nargs='+', default=[10, 30, 100, 400])
    sp = sub.add_parser('match'); sp.add_argument('checkpoint_a'); sp.add_argument('checkpoint_b')
    common(sp); sp.add_argument('--out')
    sp = sub.add_parser('plots'); sp.add_argument('run_dir')
    sp = sub.add_parser('game'); sp.add_argument('checkpoint'); sp.add_argument('--sims', type=int, default=100)
    sp.add_argument('--opponent', choices=['alpha_beta', 'random'], default='alpha_beta')
    sp.add_argument('--agent-second', action='store_true'); sp.add_argument('--seed', type=int, default=0)
    sp.add_argument('--out')

    args = p.parse_args(argv)
    device = get_device(args.device)
    {'ladder': cmd_ladder, 'baselines': cmd_baselines, 'sims': cmd_sims, 'match': cmd_match,
     'plots': cmd_plots, 'game': cmd_game}[args.command](args, device)


if __name__ == '__main__':
    main()
