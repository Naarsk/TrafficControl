"""Average-cost MDP solvers: Value Iteration and Policy Iteration.

Implements the algorithms from ``docs/algorithms.md`` (lecture-notes
versions). Both work on the sparse representation produced by
``F4C2Model``: a (n_sa, n_states) CSR transition matrix ``T``, a
(n_sa,) cost vector, and a CSR-style ``sa_start`` segmentation that
tells us which (state, action) rows belong to each state.

The state-action layout is built in encoded-state order, so the rows
of ``T`` belonging to state ``s`` form the contiguous slice
``[sa_start[s], sa_start[s+1])``.  This lets us compute the
state-wise minimum/argmin over actions with a couple of vectorised
ops instead of a Python loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

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
    model: F4C2Model, policy_sa: np.ndarray, ref_state: int = 0
) -> tuple[float, np.ndarray]:
    """Solve the Poisson equations for a fixed policy.

    Build the augmented sparse system ((n+1) x (n+1)):
        (I - P_R) V + 1 * g = r_R           (n equations, one per state)
                       V[ref] = 0           (1 normalisation row)
    and solve with sparse LU. Returns (g, V).
    """
    n = model.n_states
    P_R, r_R = _policy_kernel(model, policy_sa)

    I = sp.eye(n, format="csr")
    ones_col = sp.csr_matrix(np.ones((n, 1)))
    top = sp.hstack([I - P_R, ones_col], format="csr")
    bot = sp.csr_matrix(
        (np.ones(1), (np.zeros(1, dtype=np.int64), np.array([ref_state], dtype=np.int64))),
        shape=(1, n + 1),
    )
    A = sp.vstack([top, bot], format="csc")      # csc for spsolve
    b = np.concatenate([r_R, np.array([0.0])])

    x = spsolve(A, b)
    V = x[:n]
    g = float(x[n])
    return g, V


def policy_iteration(
    model: F4C2Model,
    max_iter: int = 100,
    ref_state: int = 0,
    initial_policy: np.ndarray | None = None,
    log_interval: float = 0.0,         # 0 -> log every iteration (PI iters are expensive)
) -> PIResult:
    """Average-cost Policy Iteration (notes algorithm 1).

    Alternates exact policy evaluation (sparse Poisson solve) with
    one-step greedy improvement, stopping when the policy is stable.
    """
    n_states = model.n_states
    log = ProgressLogger(min_interval=log_interval, prefix="  [PI] ")
    log.log(f"start  n_states={n_states:,}  n_sa={model.n_sa:,}", force=True)

    # initial policy: take the first feasible action at every state
    if initial_policy is None:
        policy_sa = model.sa_start[:-1].copy().astype(np.int64)
    else:
        policy_sa = initial_policy.astype(np.int64).copy()

    g = float("nan")
    V = np.zeros(n_states, dtype=np.float64)

    for it in range(1, max_iter + 1):
        t_eval0 = time.time()
        log.log(f"iter {it}  evaluating policy (sparse Poisson solve) ...", force=True)
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
