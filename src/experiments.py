"""Experiment orchestration for the F4C2 traffic-MDP homework.

Replicates Table 1 (symmetric F4C2) of Haijema & van der Wal (2008).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .model import F4C2Model, F4C2Params
from .solvers import value_iteration, policy_iteration


# Paper Table 1, MDP-optimal cyclic E[W] in seconds (symmetric F4C2):
PAPER_TABLE1: dict[float, float] = {
    0.40: 4.89,
    0.60: 6.95,
    0.80: 13.5,
}


def little_law_E_W(g: float, q: tuple[float, ...], slot_seconds: float = 2.0) -> float:
    """Convert avg cars per slot (g) to E[W] in seconds.

    By Little's law L = lambda * W, with L = g (avg cars in system) and
    lambda = total arrival rate. Arrival rate is sum_f q_f cars per slot,
    or sum_f q_f / slot_seconds cars per second.
    """
    arrivals_per_second = sum(q) / slot_seconds
    return g / arrivals_per_second


def q_for_rho(rho: float) -> tuple[float, float, float, float]:
    """Per-flow Bernoulli arrival prob giving symmetric workload rho.

    For F4C2 with C_1 = {1, 3}, C_2 = {2, 4}, the workload is
        rho = max_{f in C_1} q_f + max_{f in C_2} q_f.
    Symmetric case (all q_f equal): q_f = rho / 2.
    """
    q_each = rho / 2.0
    return (q_each, q_each, q_each, q_each)


@dataclass
class WorkloadResult:
    rho: float
    K: int
    g_vi: float | None = None
    g_pi: float | None = None
    e_w_vi: float | None = None
    e_w_pi: float | None = None
    iters_vi: int | None = None
    iters_pi: int | None = None
    time_build: float = 0.0
    time_vi: float = 0.0
    time_pi: float = 0.0


def run_single(
    rho: float,
    K: int = 10,
    algo: str = "both",
    tol: float = 1e-6,
    log_interval: float = 1.0,
) -> WorkloadResult:
    """Build the model at workload rho and solve with the chosen algorithm(s)."""
    q = q_for_rho(rho)
    params = F4C2Params(q=q, K=K)
    res = WorkloadResult(rho=rho, K=K)

    print(f"\n=== rho = {rho:.2f}    q_f = {q[0]:.3f}    K = {K}    tau = {params.tau} ===")
    t0 = time.time()
    model = F4C2Model(params)
    res.time_build = time.time() - t0

    if algo in ("vi", "both"):
        print("\n--- Value Iteration ---")
        vi = value_iteration(model, tol=tol, log_interval=log_interval)
        res.g_vi = vi.g
        res.iters_vi = vi.iterations
        res.time_vi = vi.elapsed
        res.e_w_vi = little_law_E_W(vi.g, q, params.slot_seconds)
        ps = model.policy_summary(vi.policy_sa)
        print(
            f"  -> g_VI = {vi.g:.6f}    E[W]_VI = {res.e_w_vi:.3f} s    "
            f"iters = {vi.iterations}    "
            f"second-action picks = {ps['n_pick_second_action']}/{ps['n_decision_states']}"
        )

    if algo in ("pi", "both"):
        print("\n--- Policy Iteration ---")
        pi = policy_iteration(model, log_interval=log_interval)
        res.g_pi = pi.g
        res.iters_pi = pi.iterations
        res.time_pi = pi.elapsed
        res.e_w_pi = little_law_E_W(pi.g, q, params.slot_seconds)
        ps = model.policy_summary(pi.policy_sa)
        print(
            f"  -> g_PI = {pi.g:.6f}    E[W]_PI = {res.e_w_pi:.3f} s    "
            f"iters = {pi.iterations}    "
            f"second-action picks = {ps['n_pick_second_action']}/{ps['n_decision_states']}"
        )

    if res.g_vi is not None and res.g_pi is not None:
        gap = abs(res.g_vi - res.g_pi)
        print(f"\n  cross-check |g_VI - g_PI| = {gap:.3e}")

    if rho in PAPER_TABLE1:
        target = PAPER_TABLE1[rho]
        print(f"  paper Table 1 target E[W] = {target} s")
        if res.e_w_vi is not None:
            print(f"    delta_VI = {res.e_w_vi - target:+.3f} s")
        if res.e_w_pi is not None:
            print(f"    delta_PI = {res.e_w_pi - target:+.3f} s")

    return res


def run_table1(
    K: int = 10,
    algo: str = "both",
    tol: float = 1e-6,
    log_interval: float = 1.0,
) -> list[WorkloadResult]:
    """Replicate Table 1: rho in {0.40, 0.60, 0.80} with identical arrival probs."""
    results = []
    for rho in (0.40, 0.60, 0.80):
        results.append(run_single(rho, K=K, algo=algo, tol=tol, log_interval=log_interval))

    # final summary
    print("\n" + "=" * 78)
    print("Summary  (paper Table 1 MDP-optimal cyclic, symmetric F4C2)")
    print("=" * 78)
    header = f"{'rho':>5}  {'paper E[W]':>10}  {'E[W]_VI':>9}  {'E[W]_PI':>9}  {'|VI-PI|':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        target = PAPER_TABLE1.get(r.rho, float("nan"))
        ew_vi = r.e_w_vi if r.e_w_vi is not None else float("nan")
        ew_pi = r.e_w_pi if r.e_w_pi is not None else float("nan")
        gap = (
            abs(r.g_vi - r.g_pi)
            if (r.g_vi is not None and r.g_pi is not None)
            else float("nan")
        )
        print(f"{r.rho:>5.2f}  {target:>10.3f}  {ew_vi:>9.3f}  {ew_pi:>9.3f}  {gap:>9.2e}")
    print("=" * 78)

    return results
