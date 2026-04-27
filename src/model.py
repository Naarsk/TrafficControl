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
        """Per-flow next-queue distribution: list of (k_f_next, prob)."""
        K = self.params.K
        if can_depart:
            # one car leaves deterministically (if any), then 0/1 arrival
            no_arr = max(k_f - 1, 0)
            with_arr = no_arr + 1                        # always <= K since k_f <= K
            return [(no_arr, 1.0 - q_f), (with_arr, q_f)]
        # red for this flow: only arrivals
        if k_f >= K:
            return [(K, 1.0)]                            # arrival rejected at the cap
        return [(k_f, 1.0 - q_f), (k_f + 1, q_f)]

    # -------- sparse kernel build --------

    def _build(self) -> None:
        """Enumerate all (s, a) pairs and assemble the sparse transition kernel.

        Strategy:
          * iterate states in encoded order, so sa indices are consecutive
            within a state and ``sa_start`` is just a cumulative count;
          * for each (s, a), enumerate up to 2^F per-flow outcome combos;
          * store rows/cols/data lists, then build CSR once at the end;
          * apply the strong-aperiodicity transformation in matrix form
            (T = tau * T_raw + (1 - tau) * I_sa_to_s).
        """
        params = self.params
        F = self.F
        n_states = self.n_states
        q = params.q
        tau = params.tau

        log = ProgressLogger(min_interval=1.0, prefix="  [build] ")
        log.log(f"K={params.K}, n_states={n_states:,}", force=True)

        # output buffers; sa_start[s+1] - sa_start[s] gives # actions at state s
        sa_start = np.zeros(n_states + 1, dtype=np.int64)
        sa_state_list: list[int] = []
        cost_list: list[float] = []
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        sa_idx = 0

        for s in range(n_states):
            k, l, i = self.decode(s)
            all_zero = (k[0] == 0 and k[1] == 0 and k[2] == 0 and k[3] == 0)
            cost_s = float(k[0] + k[1] + k[2] + k[3])
            actions = self.feasible_actions(l, i, all_zero)

            for (l_next, i_next) in actions:
                # which flows can depart in the coming slot?
                #   active combination flows during green/yellow phases (i' in {0,1,2});
                #   nobody departs during all-red (i' = 3).
                if i_next == 3:
                    active_flows: frozenset[int] = frozenset()
                else:
                    active_flows = F4C2_COMBINATIONS[l_next]

                # per-flow distributions (length 4)
                dists = [
                    self._per_flow_dist(k[f], f in active_flows, q[f])
                    for f in range(F)
                ]

                # joint successors via cartesian product (<= 16 outcomes for F=4)
                for outcome in itertools.product(*dists):
                    k_next = (outcome[0][0], outcome[1][0], outcome[2][0], outcome[3][0])
                    p = outcome[0][1] * outcome[1][1] * outcome[2][1] * outcome[3][1]
                    if p <= 0.0:
                        continue
                    s_next = self.encode(k_next, l_next, i_next)
                    rows.append(sa_idx)
                    cols.append(s_next)
                    data.append(p)

                cost_list.append(cost_s)
                sa_state_list.append(s)
                sa_idx += 1

            sa_start[s + 1] = sa_idx

            # progress log: every ~1 s
            if (s & 0x3FFF) == 0:
                frac = (s + 1) / n_states
                log.log(
                    f"states {s + 1:>9,}/{n_states:,} ({100 * frac:5.1f}%)  "
                    f"sa={sa_idx:>9,}  nnz={len(data):>10,}  "
                    f"elapsed={log.elapsed():5.1f}s  eta={fmt_eta(log.elapsed(), frac)}"
                )

        log.log(
            f"states {n_states:,}/{n_states:,} (100.0%)  sa={sa_idx:,}  "
            f"nnz={len(data):,}  elapsed={log.elapsed():.1f}s",
            force=True,
        )

        n_sa = sa_idx
        self.n_sa = n_sa
        self.sa_state = np.asarray(sa_state_list, dtype=np.int64)
        self.cost = np.asarray(cost_list, dtype=np.float64)
        self.sa_start = sa_start

        # raw kernel
        log.log("assembling sparse CSR ...", force=True)
        T_raw = csr_matrix(
            (np.asarray(data, dtype=np.float64),
             (np.asarray(rows, dtype=np.int64),
              np.asarray(cols, dtype=np.int64))),
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
            f"model ready  n_states={n_states:,}  n_sa={n_sa:,}  "
            f"nnz={self.T.nnz:,}  total build {log.elapsed():.1f}s",
            force=True,
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
