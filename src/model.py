"""F4C2 traffic-MDP model.

Implements the discrete-time average-cost MDP from
    Haijema & van der Wal (2008), "An MDP Decomposition Approach for
    Traffic Control at Isolated Signalized Intersections",
    Probability in the Engineering and Informational Sciences 22(4).

Scope: the small "F4C2" infrastructure (4 flows in 2 symmetric combinations).
Conventions follow the paper's section 3 with one notational tweak: phases
and combinations are 0-indexed here for convenience.

State                    s = (k_1, k_2, k_3, k_4, l, i)
    k_f in {0, ..., K}   queue length per flow, capped at buffer K
    l   in {0, 1}        currently-active combination (0 = C_1, 1 = C_2)
    i   in {0, 1, 2, 3}  signal phase (green / yellow1 / yellow2 / all-red)

Action set A(s)
    The action specifies *next slot's* light state (l', i'):
        i = 0 (green):    [keep (l, 0)]                       if all queues empty
                          [keep (l, 0), switch to (l, 1)]     otherwise
        i = 1 (yellow1):  [advance to (l, 2)]                 forced
        i = 2 (yellow2):  [advance to (l, 3)]                 forced
        i = 3 (all-red):  [keep (l, 3)]                       if all queues empty
                          [keep (l, 3), advance to (l', 0)]   otherwise
    "Keep all-red when queues non-empty" is wasteful but allowed by the
    paper -- including it for fidelity.

Per-flow transition (paper section 3.3, with buffer cap K)
    If flow f gets green/yellow next slot (in active combo and i' in {0,1,2}):
        k_f -> max(k_f - 1, 0)        with prob 1 - q_f
        k_f -> max(k_f - 1, 0) + 1    with prob q_f
    Otherwise (red for flow f):
        k_f -> k_f                    with prob 1 - q_f
        k_f -> min(k_f + 1, K)        with prob q_f       [arrivals rejected at cap]

Cost (paper section 3.4)
    c(s) = sum_f k_f                  total cars in system this slot

Strong-aperiodicity transformation (notes p. 11)
    barP(s, a, s') = tau * P(s, a, s') + (1 - tau) * 1{s' = s}
    Required for Value Iteration. PI tolerates the original kernel but we
    apply the same transformation for both, so VI and PI use the *same*
    transition matrix and their cost-rates are directly comparable.

Sparse representation
    n_states                = total state count
    n_sa                    = total (state, action) pair count
    cost[sa]                = c(s) for the state of pair sa            shape (n_sa,)
    sa_state[sa]            = s for pair sa                             shape (n_sa,)
    sa_start[s], sa_start[s+1]  = [start, end) range of pairs at state s, shape (n_states+1,)
    T[sa, s']               = barP(s, a, s')                           shape (n_sa, n_states)
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix

from .progress import ProgressLogger, fmt_eta


# Combinations for F4C2: C_1 = {flow 1, flow 3}, C_2 = {flow 2, flow 4}.
# 0-indexed: combo 0 has flows {0, 2}, combo 1 has flows {1, 3}.
F4C2_COMBINATIONS: tuple[frozenset[int], ...] = (
    frozenset({0, 2}),
    frozenset({1, 3}),
)


@dataclass(frozen=True)
class F4C2Params:
    """Hyperparameters for the F4C2 model."""
    q: tuple[float, float, float, float] = (0.2, 0.2, 0.2, 0.2)
    K: int = 10                         # buffer cap per flow
    tau: float = 0.5                    # strong-aperiodicity transformation
    slot_seconds: float = 2.0


@dataclass
class F4C2Model:
    """Sparse MDP encoding for the F4C2 intersection."""
    params: F4C2Params
    n_states: int = field(init=False)
    n_sa: int = field(init=False)
    T: csr_matrix = field(init=False)         # transition kernel, (n_sa, n_states)
    cost: np.ndarray = field(init=False)      # one-step cost per (s, a), (n_sa,)
    sa_state: np.ndarray = field(init=False)  # state index per (s, a), (n_sa,)
    sa_start: np.ndarray = field(init=False)  # CSR-style segment starts, (n_states+1,)

    F: int = field(init=False, default=4)
    S: int = field(init=False, default=2)
    PHASES: int = field(init=False, default=4)

    def __post_init__(self) -> None:
        K = self.params.K
        # mixed-radix sizes for (k_1, k_2, k_3, k_4, l, i)
        self._radix = (K + 1, K + 1, K + 1, K + 1, self.S, self.PHASES)
        self.n_states = int(np.prod(self._radix))
        self._build()

    # -------- state encoding (mixed-radix flat int) --------

    def encode(self, k: tuple[int, int, int, int], l: int, i: int) -> int:
        K1 = self.params.K + 1
        return ((((k[0] * K1 + k[1]) * K1 + k[2]) * K1 + k[3]) * self.S + l) * self.PHASES + i

    def decode(self, s: int) -> tuple[tuple[int, int, int, int], int, int]:
        s, i = divmod(s, self.PHASES)
        s, l = divmod(s, self.S)
        K1 = self.params.K + 1
        s, k4 = divmod(s, K1)
        s, k3 = divmod(s, K1)
        s, k2 = divmod(s, K1)
        s, k1 = divmod(s, K1)
        return (k1, k2, k3, k4), l, i

    # -------- decision space --------

    def feasible_actions(
        self, l: int, i: int, all_zero: bool
    ) -> list[tuple[int, int]]:
        """Return list of next-light-state tuples (l', i') feasible at (l, i).

        ``all_zero`` is the indicator that every queue is empty in the
        current state (triggers the freeze rule of the paper).
        """
        if i == 0:                          # green
            if all_zero:
                return [(l, 0)]                          # frozen: keep
            return [(l, 0), (l, 1)]                      # keep | switch to yellow
        if i == 1:                          # first yellow
            return [(l, 2)]                              # forced advance
        if i == 2:                          # second yellow
            return [(l, 3)]                              # forced advance
        if i == 3:                          # all-red
            if all_zero:
                return [(l, 3)]                          # frozen: keep all-red
            return [(l, 3), ((l + 1) % self.S, 0)]       # keep | advance to next combo
        raise ValueError(f"invalid phase {i}")

    # -------- per-flow transition distribution --------

    def _per_flow_dist(
        self, k_f: int, can_depart: bool, q_f: float
    ) -> list[tuple[int, float]]:
        """Per-flow next-queue distribution: list of (k_f_next, prob).

        Paper section 2.6 dynamics inside one slot:
            (i)   observe state at slot start
            (ii)  arrival Bernoulli(q_f) joins the queue
            (iii) departure (one car leaves if queue non-empty AND active)
            (iv)  observe state at slot end
        Therefore a car arriving at an empty active queue gets served in
        the same slot. The paper's formula 3.3 captures this:
            p_f(k, a, (k - 1)^+) = 1 - q,  p_f(k, a, k) = q   if can_depart
            p_f(k, a, k)         = 1 - q,  p_f(k, a, k + 1) = q   else
        Note: at k = 0 with can_depart, both transitions land on 0 — the
        arriving car is immediately served.
        """
        K = self.params.K
        if can_depart:
            no_arr = max(k_f - 1, 0)        # no arrival, one departure
            with_arr = k_f                  # arrival + departure cancel; at k=0 stays 0
            return [(no_arr, 1.0 - q_f), (with_arr, q_f)]
        # red for this flow: only arrivals
        if k_f >= K:
            return [(K, 1.0)]                            # arrival rejected at the cap
        return [(k_f, 1.0 - q_f), (k_f + 1, q_f)]

    # -------- sparse kernel build --------

    def _build(self) -> None:
        """Build the sparse transition kernel.

        Dispatches to a compiled Cython kernel
        (``src._build_kernel.build_f4c2_kernel``) when available; falls
        back to the pure-Python implementation in ``_build_python``
        otherwise. Both paths produce the same arrays
        (``sa_state``, ``sa_start``, ``cost``, plus the raw triplets
        for CSR construction); the strongly-aperiodicity
        ``tau``-transform is applied identically in Python afterwards.
        """
        params = self.params
        n_states = self.n_states
        tau = params.tau
        log = ProgressLogger(min_interval=1.0, prefix="  [build] ")
        log.log(f"K={params.K}, n_states={n_states:,}", force=True)

        # Try the fastest available backend.
        # 1. Cython (compiled .pyd/.so). Fastest, but needs MSVC/gcc to build.
        # 2. Numba (LLVM JIT). Pip-installable, no compiler needed.
        # 3. Pure-Python (the existing reference impl). Always works.
        build_f4c2_kernel = None
        backend = "python"
        try:
            from src._build_kernel import build_f4c2_kernel as _cy_kernel
            build_f4c2_kernel = _cy_kernel
            backend = "cython"
        except ImportError:
            try:
                from src._build_kernel_numba import build_f4c2_kernel as _nb_kernel
                build_f4c2_kernel = _nb_kernel
                backend = "numba"
            except ImportError:
                log.log(
                    "Neither Cython extension nor Numba available; falling back "
                    "to Python (slow at large K).",
                    force=True,
                )

        if backend in ("cython", "numba"):
            log.log(f"enumerating transitions ({backend}) ...", force=True)
            q_arr = np.asarray(params.q, dtype=np.float64)
            rows, cols, data, sa_state, sa_start, cost = build_f4c2_kernel(
                params.K, q_arr,
            )
            n_sa = int(sa_state.shape[0])
            log.log(
                f"  -> n_sa={n_sa:,}  nnz_raw={rows.shape[0]:,}  "
                f"elapsed={log.elapsed():.2f}s",
                force=True,
            )
        else:
            rows, cols, data, sa_state, sa_start, cost = self._build_python(log)
            n_sa = int(sa_state.shape[0])

        self.n_sa = n_sa
        self.sa_state = sa_state
        self.cost = cost
        self.sa_start = sa_start

        # Assemble raw kernel into a CSR matrix.
        log.log("assembling sparse CSR ...", force=True)
        T_raw = csr_matrix(
            (data, (rows, cols)),
            shape=(n_sa, n_states),
        )

        # tau-transformation: bar T = tau * T_raw + (1 - tau) * I_sa_to_s
        # I_sa_to_s has a 1 at (sa, sa_state[sa]) for every sa.
        log.log(f"applying strong-aperiodicity transformation (tau={tau}) ...", force=True)
        I_sa = csr_matrix(
            (np.ones(n_sa, dtype=np.float64),
             (np.arange(n_sa, dtype=np.int64), self.sa_state)),
            shape=(n_sa, n_states),
        )
        self.T = (tau * T_raw + (1.0 - tau) * I_sa).tocsr()

        log.log(
            f"model ready  backend={backend}  n_states={n_states:,}  n_sa={n_sa:,}  "
            f"nnz={self.T.nnz:,}  total build {log.elapsed():.1f}s",
            force=True,
        )

    def _build_python(
        self, log: "ProgressLogger"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pure-Python reference implementation of the kernel build.

        Returns the same six arrays as the Cython kernel:
        ``(rows, cols, data, sa_state, sa_start, cost)``. Used as a
        fallback when the Cython extension isn't compiled, and as a
        reference for correctness testing.
        """
        params = self.params
        F = self.F
        n_states = self.n_states
        q = params.q

        sa_start = np.zeros(n_states + 1, dtype=np.int64)
        sa_state_list: list[int] = []
        cost_list: list[float] = []
        rows_list: list[int] = []
        cols_list: list[int] = []
        data_list: list[float] = []

        sa_idx = 0

        for s in range(n_states):
            k, l, i = self.decode(s)
            all_zero = (k[0] == 0 and k[1] == 0 and k[2] == 0 and k[3] == 0)
            cost_s = float(k[0] + k[1] + k[2] + k[3])
            actions = self.feasible_actions(l, i, all_zero)

            for (l_next, i_next) in actions:
                if i_next == 3:
                    active_flows: frozenset[int] = frozenset()
                else:
                    active_flows = F4C2_COMBINATIONS[l_next]

                dists = [
                    self._per_flow_dist(k[f], f in active_flows, q[f])
                    for f in range(F)
                ]

                for outcome in itertools.product(*dists):
                    k_next = (outcome[0][0], outcome[1][0], outcome[2][0], outcome[3][0])
                    p = outcome[0][1] * outcome[1][1] * outcome[2][1] * outcome[3][1]
                    if p <= 0.0:
                        continue
                    s_next = self.encode(k_next, l_next, i_next)
                    rows_list.append(sa_idx)
                    cols_list.append(s_next)
                    data_list.append(p)

                cost_list.append(cost_s)
                sa_state_list.append(s)
                sa_idx += 1

            sa_start[s + 1] = sa_idx

            if (s & 0x3FFF) == 0:
                frac = (s + 1) / n_states
                log.log(
                    f"states {s + 1:>9,}/{n_states:,} ({100 * frac:5.1f}%)  "
                    f"sa={sa_idx:>9,}  nnz={len(data_list):>10,}  "
                    f"elapsed={log.elapsed():5.1f}s  eta={fmt_eta(log.elapsed(), frac)}"
                )

        log.log(
            f"states {n_states:,}/{n_states:,} (100.0%)  sa={sa_idx:,}  "
            f"nnz={len(data_list):,}  elapsed={log.elapsed():.1f}s",
            force=True,
        )

        return (
            np.asarray(rows_list, dtype=np.int64),
            np.asarray(cols_list, dtype=np.int64),
            np.asarray(data_list, dtype=np.float64),
            np.asarray(sa_state_list, dtype=np.int64),
            sa_start,
            np.asarray(cost_list, dtype=np.float64),
        )

    # -------- diagnostics --------

    def policy_summary(self, policy_sa: np.ndarray) -> dict:
        """Quick policy-shape report: how many states pick the 'switch'/'advance' action."""
        # n_actions[s] = sa_start[s+1] - sa_start[s]; states with 2 actions are decision states.
        nacts = np.diff(self.sa_start)
        decision_mask = nacts == 2
        # picked_second[s] = True iff policy chose the second action at state s
        picked_second = (policy_sa - self.sa_start[:-1]).astype(bool)
        n_decision = int(decision_mask.sum())
        n_pick_second = int((decision_mask & picked_second).sum())
        return {
            "n_decision_states": n_decision,
            "n_pick_second_action": n_pick_second,
            "fraction_pick_second": (n_pick_second / n_decision) if n_decision else 0.0,
        }
