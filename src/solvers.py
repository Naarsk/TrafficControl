"""Average-cost MDP solvers: Value Iteration and Policy Iteration.

Implements the algorithms from ``docs/algorithms.md``. Both work on
the sparse representation produced by ``F4C2Model``: a
(n_sa, n_states) CSR transition matrix ``T``, a (n_sa,) cost vector,
and a CSR-style ``sa_start`` segmentation that tells us which
(state, action) rows belong to each state.

The state-action layout is built in encoded-state order, so the rows
of ``T`` belonging to state ``s`` form the contiguous slice
``[sa_start[s], sa_start[s+1])``. This lets us compute the
state-wise minimum/argmin over actions with a couple of vectorised
ops instead of a Python loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .model import F4C2Model
from .progress import ProgressLogger


# ----- shared helpers ------------------------------------------------------


def _state_min_argmin(
    Q: np.ndarray, sa_start: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-state min and argmin of the (n_sa,) Q-value vector.

    Each state owns the contiguous slice ``Q[sa_start[s]:sa_start[s+1]]``.
    For the F4C2 MDP every state has 1 or 2 feasible actions; we take a
    fast path for that case (avoids a Python-level reduction).

    Returns
    -------
    V_state : ndarray (n_states,)
        ``V_state[s] = min_a Q[sa]`` over feasible (s, a).
    policy_sa : ndarray (n_states,)
        Index into ``Q`` of the chosen state-action pair at each state.
    """
    n_states = sa_start.shape[0] - 1
    nacts = np.diff(sa_start)                 # 1 or 2 per state for F4C2
    if nacts.max() > 2:
        # Generic fallback: segment-wise reduction with reduceat.
        V_state = np.minimum.reduceat(Q, sa_start[:-1])
        # argmin: scan within each segment (slow path, only used if F4C2 invariant breaks)
        policy_sa = np.empty(n_states, dtype=np.int64)
        for s in range(n_states):
            seg = Q[sa_start[s]:sa_start[s + 1]]
            policy_sa[s] = sa_start[s] + int(np.argmin(seg))
        return V_state, policy_sa

    # Fast path: 1 or 2 actions per state.
    first = sa_start[:-1]                     # sa-index of the first action per state
    second = first + 1                        # sa-index of the second action (only valid where nacts == 2)
    V_first = Q[first]
    V_state = V_first.copy()
    policy_sa = first.copy()
    two_act = nacts == 2
    if two_act.any():
        idx2 = np.where(two_act)[0]
        V_second = Q[second[idx2]]
        better_second = V_second < V_first[idx2]
        V_state[idx2] = np.where(better_second, V_second, V_first[idx2])
        policy_sa[idx2] = np.where(better_second, second[idx2], first[idx2])
    return V_state, policy_sa


# ----- Value Iteration -----------------------------------------------------


@dataclass
class VIResult:
    g: float                          # near-optimal cost-rate (avg of m_n, M_n at stop)
    V: np.ndarray                     # relative value function (normalised so min = 0)
    policy_sa: np.ndarray             # chosen sa-index per state
    iterations: int
    m_n: float                        # final lower bound  m_n <= g*
    M_n: float                        # final upper bound  M_n >= g^{R_n}
    elapsed: float


def value_iteration(
    model: F4C2Model,
    tol: float = 1e-6,
    max_iter: int = 20_000,
    log_interval: float = 1.0,
) -> VIResult:
    """Average-cost VI for a strongly aperiodic MDP (notes algorithm 3).

    Stops when ``(M_n - m_n) <= tol * |m_n|``, where
    ``m_n = min_x (V_n(x) - V_{n-1}(x))`` and likewise for ``M_n``.

    The model's transition matrix is already the tau-transformed kernel,
    so strong aperiodicity holds.
    """
    n_states = model.n_states
    V = np.zeros(n_states, dtype=np.float64)
    log = ProgressLogger(min_interval=log_interval, prefix="  [VI] ")
    log.log(f"start  n_states={n_states:,}  n_sa={model.n_sa:,}  tol={tol:g}", force=True)

    m_n = -np.inf
    M_n = np.inf
    policy_sa = np.zeros(n_states, dtype=np.int64)

    for n in range(1, max_iter + 1):
        # Bellman backup: Q[sa] = c[sa] + (T @ V)[sa]
        Q = model.cost + model.T @ V
        V_new, policy_sa = _state_min_argmin(Q, model.sa_start)

        diff = V_new - V
        m_n = float(diff.min())
        M_n = float(diff.max())
        span = M_n - m_n
        denom = max(abs(m_n), 1e-12)

        log.log(
            f"iter {n:>5d}  m_n={m_n:+.6f}  M_n={M_n:+.6f}  "
            f"span={span:.3e}  rel={span / denom:.3e}  "
            f"elapsed={log.elapsed():6.1f}s"
        )

        # stop criterion (notes algorithm 3, step 3)
        if span <= tol * denom:
            g_est = 0.5 * (m_n + M_n)
            log.log(
                f"converged at iter {n}  g={g_est:.6f}  "
                f"[m_n={m_n:.6f}, M_n={M_n:.6f}]  total {log.elapsed():.1f}s",
                force=True,
            )
            return VIResult(
                g=g_est,
                V=V_new - V_new.min(),     # bias unique up to constant; pin min to 0
                policy_sa=policy_sa,
                iterations=n,
                m_n=m_n,
                M_n=M_n,
                elapsed=log.elapsed(),
            )
        V = V_new

    raise RuntimeError(
        f"VI failed to converge in {max_iter} iters; last span={M_n - m_n:.3e}, "
        f"m_n={m_n:.6f}, M_n={M_n:.6f}"
    )


# ----- Policy Iteration ---------------------------------------------------


@dataclass
class PIResult:
    g: float                          # exact cost-rate at convergence
    V: np.ndarray                     # relative value function (normalised V[ref]=0)
    policy_sa: np.ndarray             # chosen sa-index per state
    iterations: int
    elapsed: float


def _policy_kernel(model: F4C2Model, policy_sa: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
    """Build the policy-induced transition matrix and cost vector."""
    P_R = model.T[policy_sa]                     # (n_states, n_states), CSR row-slice
    r_R = model.cost[policy_sa].astype(np.float64, copy=True)
    return P_R.tocsr(), r_R


def evaluate_policy(
    model: F4C2Model,
    policy_sa: np.ndarray,
    ref_state: int = 0,
    tol: float = 1e-6,                       # match VI's stopping tolerance
    max_iter: int = 10_000,                  # bound the wasted time when policy is near-reducible
    log_interval: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Solve the Poisson equations by successive approximation (iterative).

    The "direct" approach -- assemble the augmented (n+1) x (n+1) sparse
    system and call ``scipy.sparse.linalg.spsolve`` -- works for tiny
    state spaces but performs a sparse LU on a 117k x 117k matrix here,
    with heavy fill-in: each call takes minutes (or longer).

    Instead we run VI fixed at the policy: V_{n+1} = r_R + P_R V_n.
    Under strong aperiodicity (already enforced by the tau-transform in
    the model build), this converges geometrically with V_{n+1} - V_n -> g.
    Each step is one sparse mat-vec, ~30 ms here, so a typical eval
    takes a few hundred steps and a few seconds.

    At convergence,  V_n = V_inf + n * g + o(1).  Subtracting V_n[ref]
    cancels the linear-in-n drift and yields the bounded relative value
    function with V[ref] = 0.

    Stops when ``(M_n - m_n) <= tol * |m_n|``.
    """
    n_states = model.n_states
    P_R, r_R = _policy_kernel(model, policy_sa)
    V = np.zeros(n_states, dtype=np.float64)

    log = ProgressLogger(min_interval=log_interval, prefix="    [eval] ")
    log.log(f"start  iterative policy evaluation  tol={tol:g}", force=True)

    m_n = -np.inf
    M_n = np.inf
    span = np.inf

    for n in range(1, max_iter + 1):
        V_new = r_R + P_R @ V
        diff = V_new - V
        m_n = float(diff.min())
        M_n = float(diff.max())
        span = M_n - m_n
        denom = max(abs(m_n), 1e-12)

        log.log(
            f"iter {n:>5d}  m_n={m_n:+.6f}  M_n={M_n:+.6f}  "
            f"span={span:.3e}  rel={span / denom:.3e}  elapsed={log.elapsed():5.1f}s"
        )

        if span <= tol * denom:
            g = 0.5 * (m_n + M_n)
            # V_new ~ V_inf + n*g; subtract V_new[ref] to remove the n*g drift
            V_relative = V_new - V_new[ref_state]
            log.log(
                f"converged at iter {n}  g={g:.6f}  total {log.elapsed():.1f}s",
                force=True,
            )
            return g, V_relative
        V = V_new

    raise RuntimeError(
        f"policy evaluation did not converge in {max_iter} iters "
        f"(last span={span:.3e}, m_n={m_n:.6f}, M_n={M_n:.6f}). "
        f"This typically means the current policy induces a reducible "
        f"Markov chain (multiple recurrent classes or a slow-mixing "
        f"absorbing region near the buffer cap). PI's correctness "
        f"theorem assumes a strict unichain MDP; the F4C2 traffic MDP "
        f"is only weakly unichain (see report.tex section 'Unichain "
        f"and aperiodicity'). Use VI in this regime."
    )


def initial_unichain_policy(model: F4C2Model) -> np.ndarray:
    """A deterministic policy that is guaranteed to be unichain.

    Picks the *second* feasible action whenever a state has two:
      * Phase 0 with non-empty queues: switch to first yellow
      * Phase 3 with non-empty queues: advance to next combination's green
    Single-action states (forced yellow advance, frozen green/all-red on
    empty queues) keep their only action.

    The induced chain alternates phases (l, 0) -> (l, 1) -> (l, 2) ->
    (l, 3) -> ((l+1) mod S, 0) -> ..., visiting all (l, i) phases in a
    single 8-slot cycle and reaching every queue configuration with
    positive probability through arrivals. Hence irreducible / unichain.

    This policy is intentionally wasteful (only one slot of green per
    combination per cycle) and is unstable for high rho, but PI only
    needs it to bootstrap the Poisson solve; subsequent iterations
    improve to the optimum.
    """
    sa_start = model.sa_start
    nacts = np.diff(sa_start)
    policy = sa_start[:-1].astype(np.int64).copy()           # default first action
    has_two = nacts == 2
    policy[has_two] = sa_start[:-1][has_two] + 1             # pick second when available
    return policy


def policy_iteration(
    model: F4C2Model,
    max_iter: int = 100,
    ref_state: int = 0,
    initial_policy: np.ndarray | None = None,
    log_interval: float = 0.0,         # 0 -> log every iteration (PI iters are expensive)
) -> PIResult:
    """Average-cost Policy Iteration.

    Alternates policy evaluation (Poisson solve) with one-step greedy
    improvement, stopping when the policy is stable.

    Limitation: PI's correctness theorem assumes a strict unichain MDP
    (every policy induces a unichain Markov chain). The F4C2 traffic
    MDP is only weakly unichain -- the optimal policy is unichain, but
    intermediate policies generated by improvement can be reducible
    (chain absorbed at the buffer cap with no service). When this
    happens ``evaluate_policy`` raises a RuntimeError; use VI instead
    in that regime.
    """
    n_states = model.n_states
    log = ProgressLogger(min_interval=log_interval, prefix="  [PI] ")
    log.log(f"start  n_states={n_states:,}  n_sa={model.n_sa:,}", force=True)

    # Step 0: pick a feasible stationary policy. "First action everywhere"
    # picks 'keep green' / 'keep all-red' which leaves the chain stuck in
    # the starting phase (multiple recurrent classes -> singular Poisson).
    # The "always-switch when possible" alternative below cycles through
    # all 8 phase configurations and is unichain.
    if initial_policy is None:
        policy_sa = initial_unichain_policy(model)
    else:
        policy_sa = initial_policy.astype(np.int64).copy()

    g = float("nan")
    V = np.zeros(n_states, dtype=np.float64)

    for it in range(1, max_iter + 1):
        t_eval0 = time.time()
        log.log(f"iter {it}  evaluating policy (Poisson via successive approximation) ...", force=True)
        g, V = evaluate_policy(model, policy_sa, ref_state=ref_state)
        eval_dt = time.time() - t_eval0

        # policy improvement
        t_imp0 = time.time()
        Q = model.cost + model.T @ V
        _, policy_new = _state_min_argmin(Q, model.sa_start)
        imp_dt = time.time() - t_imp0

        n_changed = int((policy_new != policy_sa).sum())
        log.log(
            f"iter {it}  g={g:.6f}  changed={n_changed:>7d}/{n_states:,}  "
            f"eval={eval_dt:5.1f}s  improve={imp_dt:4.1f}s  total={log.elapsed():6.1f}s",
            force=True,
        )

        if n_changed == 0:
            log.log(
                f"converged at iter {it}  g={g:.6f}  total {log.elapsed():.1f}s",
                force=True,
            )
            return PIResult(
                g=g,
                V=V - V[ref_state],
                policy_sa=policy_sa,
                iterations=it,
                elapsed=log.elapsed(),
            )
        policy_sa = policy_new

    raise RuntimeError(f"PI failed to converge in {max_iter} iters")
