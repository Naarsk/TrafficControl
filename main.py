"""CLI entry point for the F4C2 traffic-MDP solver.

Recommended end-to-end run for the homework
-------------------------------------------
    python main.py table1 --K 10 --lp-K 6

VI and PI run at K=10 (paper-Table-1 numbers). LP runs at K=6 (small
state space, fast HiGHS solve), with a companion VI@K=6 so the LP/VI
cross-check is at matching K. This sidesteps LP's pathological
dual-simplex degeneracy on the K=10 polytope at higher rho.

Other forms
-----------
    python main.py table1                    # all 3 at K=10 (LP slow at rho>=0.6)
    python main.py table1 --algo vi-pi       # paper match only
    python main.py single --rho 0.4 --algo lp --K 6

Knobs
-----
    --K              per-flow buffer cap for VI, PI (default 10).
    --lp-K           per-flow buffer cap for LP (default: same as --K).
                     Set to 6 or 8 when running LP to keep the LP tractable.
    --algo           vi | pi | lp | vi-pi | vi-lp | pi-lp | all | both
                     (default all). 'both' is a legacy alias for vi-pi.
    --tol            VI relative-span stopping tolerance (default 1e-6)
    --log-interval   seconds between progress lines (default 1.0)

Notes
-----
PI is presented for completeness. The notes (p. 6) prove PI's correctness
under the *unichain* MDP assumption; the F4C2 traffic MDP is only
*weakly unichain* (notes p. 11), so PI's intermediate iterations can
generate reducible policies for which the Poisson recursion does not
converge cleanly. Empirically PI works at rho <= 0.6 and fails at rho = 0.8.
The notes recommend LP for the weakly-unichain regime; it works at every rho
but the K=10 LP is computationally pathological -- hence the small --lp-K.
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
        sp.add_argument("--K", type=int, default=10, help="buffer cap per flow (used for VI, PI)")
        sp.add_argument(
            "--lp-K",
            type=int,
            default=None,
            dest="lp_K",
            help=(
                "buffer cap for the LP run only (default: same as --K). "
                "Set to a smaller value (e.g. 6) to keep LP tractable at higher rho. "
                "When this differs from --K, a companion VI@lp_K run is also performed "
                "so the LP/VI cross-check is at the same K."
            ),
        )
        sp.add_argument(
            "--algo",
            choices=(
                "vi", "pi", "lp",
                "vi-pi", "vi-lp", "pi-lp",
                "all", "both",
            ),
            default="all",
            help="which solver(s) to run (default: all 3)",
        )
        sp.add_argument("--tol", type=float, default=1e-6, help="VI stopping tolerance")
        sp.add_argument(
            "--lp-method",
            choices=("highs", "highs-ds", "highs-ipm"),
            default="highs",
            dest="lp_method",
            help=(
                "scipy.linprog method for LP. 'highs' lets HiGHS auto-pick "
                "(usually dual simplex). 'highs-ipm' = interior point, much "
                "faster on the heavily degenerate MDP polytope at higher rho. "
                "Recommended for any rho >= 0.6: --lp-method highs-ipm."
            ),
        )
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
            lp_K=args.lp_K,
            lp_method=args.lp_method,
        )
    elif args.cmd == "single":
        run_single(
            rho=args.rho,
            K=args.K,
            algo=args.algo,
            tol=args.tol,
            log_interval=args.log_interval,
            lp_K=args.lp_K,
            lp_method=args.lp_method,
        )
    else:                                    # argparse already enforces required=True
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
