"""CLI entry point for the F4C2 traffic-MDP solver.

Examples
--------
Replicate Table 1 of Haijema & van der Wal (2008) (rho in {0.4, 0.6, 0.8})
with both Value Iteration and Policy Iteration:

    python main.py table1

Solve at a single workload (e.g. rho = 0.4) with VI only:

    python main.py single --rho 0.4 --algo vi

Knobs
-----
    --K              per-flow buffer cap (default 10).  Larger -> more states,
                     more accuracy at high rho. K=10 is fine for rho <= 0.6;
                     try K=12 or 15 for rho = 0.8.
    --algo           vi | pi | both (default both)
    --tol            VI relative-span stopping tolerance (default 1e-6)
    --log-interval   seconds between progress lines (default 1.0)
"""

from __future__ import annotations

import argparse
import sys

from src.experiments import run_single, run_table1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # common kwargs
    def _add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--K", type=int, default=10, help="buffer cap per flow")
        sp.add_argument(
            "--algo",
            choices=("vi", "pi", "both"),
            default="both",
            help="which solver(s) to run",
        )
        sp.add_argument("--tol", type=float, default=1e-6, help="VI stopping tolerance")
        sp.add_argument(
            "--log-interval",
            type=float,
            default=1.0,
            help="seconds between progress log lines",
        )

    p_t1 = sub.add_parser("table1", help="replicate Haijema & van der Wal Table 1 (symmetric F4C2)")
    _add_common(p_t1)

    p_s = sub.add_parser("single", help="solve at a single workload")
    p_s.add_argument("--rho", type=float, required=True, help="symmetric workload in (0, 1)")
    _add_common(p_s)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "table1":
        run_table1(
            K=args.K,
            algo=args.algo,
            tol=args.tol,
            log_interval=args.log_interval,
        )
    elif args.cmd == "single":
        run_single(
            rho=args.rho,
            K=args.K,
            algo=args.algo,
            tol=args.tol,
            log_interval=args.log_interval,
        )
    else:                                    # argparse already enforces required=True
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
