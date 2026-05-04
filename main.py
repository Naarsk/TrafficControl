"""CLI entry point for the F4C2 traffic-MDP solver.

Recommended end-to-end run for the homework
-------------------------------------------
    python main.py table1 --K 15

Both Value Iteration and Policy Iteration run at the configured K.
Use K = 10 for fast smoke tests (paper match within ~0.1 s for
rho <= 0.6); use K = 15 for the full paper match (within ~0.1 s
across all rho).

Other forms
-----------
    python main.py table1                    # K=10 default, both VI and PI
    python main.py table1 --algo vi          # VI only
    python main.py single --rho 0.4 --K 15

Knobs
-----
    --K              per-flow buffer cap (default 10).
    --algo           vi | pi | both          (default both)
    --tol            VI relative-span stopping tolerance (default 1e-6)
    --log-interval   seconds between progress lines (default 1.0)

Notes
-----
PI is fast and tight where it works (rho <= 0.6) but its correctness
theorem requires a strict unichain MDP, which the F4C2 model does not
satisfy: at rho = 0.8 the improvement step generates a reducible
intermediate policy on which the Poisson recursion does not converge.
At that workload only VI returns a clean result. See the report
(docs/report.tex, sections 'Unichain and aperiodicity' and 'PI failure
at rho = 0.8') for the proof and the empirical observation.
"""

from __future__ import annotations

import argparse
import sys

from src.experiments import run_single, run_table1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def _add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--K", type=int, default=10, help="buffer cap per flow")
        sp.add_argument(
            "--algo",
            choices=("vi", "pi", "both"),
            default="both",
            help="which solver(s) to run (default: both)",
        )
        sp.add_argument("--tol", type=float, default=1e-6, help="VI stopping tolerance")
        sp.add_argument(
            "--log-interval",
            type=float,
            default=1.0,
            help="seconds between progress log lines",
        )

    p_t1 = sub.add_parser(
        "table1",
        help="replicate Haijema & van der Wal Table 1 (symmetric F4C2)",
    )
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
    else:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
