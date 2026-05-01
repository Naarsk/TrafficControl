"""Numba-JIT'd implementation of the F4C2 kernel build.

Algorithmically identical to the Cython version in ``_build_kernel.pyx``;
chosen as the default fast path on systems that don't have an MSVC /
GCC C compiler available, since Numba ships with a precompiled LLVM
runtime and JIT-compiles at first call (cached on disk afterwards).

Hard-codes F = 4, S = 2 (the F4C2 paper case).
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, boundscheck=False)
def _kernel_inner(K: int, q_in: np.ndarray):
    """The hot loop. JIT-compiled by Numba on first call.

    Returns the same six arrays as the Cython kernel:
        rows, cols, data, sa_state, sa_start, cost
    """
    F = 4
    S = 2
    PHASES = 4
    K1 = K + 1
    n_states = K1 * K1 * K1 * K1 * S * PHASES

    # F4C2 active mask: combo 0 = {0, 2}, combo 1 = {1, 3}
    active_mask = np.zeros((2, 4), dtype=np.int64)
    active_mask[0, 0] = 1; active_mask[0, 2] = 1
    active_mask[1, 1] = 1; active_mask[1, 3] = 1

    # local copy of arrival probs
    q_arr = np.empty(4, dtype=np.float64)
    q_arr[0] = q_in[0]; q_arr[1] = q_in[1]
    q_arr[2] = q_in[2]; q_arr[3] = q_in[3]

    # upper-bound preallocation
    max_sa = n_states * 2
    max_nnz = n_states * 2 * 16
    rows = np.empty(max_nnz, dtype=np.int64)
    cols = np.empty(max_nnz, dtype=np.int64)
    data = np.empty(max_nnz, dtype=np.float64)
    sa_state = np.empty(max_sa, dtype=np.int64)
    cost = np.empty(max_sa, dtype=np.float64)
    sa_start = np.empty(n_states + 1, dtype=np.int64)
    sa_start[0] = 0

    kk = np.empty(4, dtype=np.int64)
    can_dep = np.empty(4, dtype=np.int64)
    dist_lo = np.empty(4, dtype=np.int64)
    dist_hi = np.empty(4, dtype=np.int64)
    dist_plo = np.empty(4, dtype=np.float64)
    dist_phi = np.empty(4, dtype=np.float64)
    actions_l = np.empty(2, dtype=np.int64)
    actions_i = np.empty(2, dtype=np.int64)

    sa_idx = 0
    nnz = 0

    for s in range(n_states):
        # decode state s: mixed-radix unpacking of (k0, k1, k2, k3, l, ph)
        tmp = s
        ph = tmp % PHASES; tmp //= PHASES
        l = tmp % S;       tmp //= S
        k3 = tmp % K1;     tmp //= K1
        k2 = tmp % K1;     tmp //= K1
        k1 = tmp % K1;     tmp //= K1
        k0 = tmp % K1

        kk[0] = k0; kk[1] = k1; kk[2] = k2; kk[3] = k3
        all_zero = (k0 == 0) and (k1 == 0) and (k2 == 0) and (k3 == 0)
        cost_s = float(k0 + k1 + k2 + k3)

        # feasible actions
        if ph == 0:
            if all_zero:
                actions_l[0] = l; actions_i[0] = 0
                n_actions = 1
            else:
                actions_l[0] = l; actions_i[0] = 0
                actions_l[1] = l; actions_i[1] = 1
                n_actions = 2
        elif ph == 1:
            actions_l[0] = l; actions_i[0] = 2
            n_actions = 1
        elif ph == 2:
            actions_l[0] = l; actions_i[0] = 3
            n_actions = 1
        else:  # ph == 3
            if all_zero:
                actions_l[0] = l; actions_i[0] = 3
                n_actions = 1
            else:
                actions_l[0] = l; actions_i[0] = 3
                actions_l[1] = (l + 1) % S; actions_i[1] = 0
                n_actions = 2

        for ai in range(n_actions):
            l_next = actions_l[ai]
            i_next = actions_i[ai]

            # which flows can depart in the coming slot?
            if i_next == 3:
                for f in range(F):
                    can_dep[f] = 0
            else:
                for f in range(F):
                    can_dep[f] = active_mask[l_next, f]

            # per-flow distributions: paper formula 3.3
            for f in range(F):
                if can_dep[f]:
                    if kk[f] == 0:
                        dist_lo[f] = 0;        dist_plo[f] = 1.0 - q_arr[f]
                        dist_hi[f] = 0;        dist_phi[f] = q_arr[f]
                    else:
                        dist_lo[f] = kk[f] - 1; dist_plo[f] = 1.0 - q_arr[f]
                        dist_hi[f] = kk[f];     dist_phi[f] = q_arr[f]
                else:
                    if kk[f] >= K:
                        dist_lo[f] = K;        dist_plo[f] = 1.0
                        dist_hi[f] = K;        dist_phi[f] = 0.0
                    else:
                        dist_lo[f] = kk[f];     dist_plo[f] = 1.0 - q_arr[f]
                        dist_hi[f] = kk[f] + 1; dist_phi[f] = q_arr[f]

            # 4 nested loops -> 16 successors max
            for o0 in range(2):
                if o0 == 0:
                    kn0 = dist_lo[0]; p0 = dist_plo[0]
                else:
                    kn0 = dist_hi[0]; p0 = dist_phi[0]
                if p0 <= 0.0:
                    continue
                for o1 in range(2):
                    if o1 == 0:
                        kn1 = dist_lo[1]; p01 = p0 * dist_plo[1]
                    else:
                        kn1 = dist_hi[1]; p01 = p0 * dist_phi[1]
                    if p01 <= 0.0:
                        continue
                    for o2 in range(2):
                        if o2 == 0:
                            kn2 = dist_lo[2]; p012 = p01 * dist_plo[2]
                        else:
                            kn2 = dist_hi[2]; p012 = p01 * dist_phi[2]
                        if p012 <= 0.0:
                            continue
                        for o3 in range(2):
                            if o3 == 0:
                                kn3 = dist_lo[3]; p0123 = p012 * dist_plo[3]
                            else:
                                kn3 = dist_hi[3]; p0123 = p012 * dist_phi[3]
                            if p0123 <= 0.0:
                                continue
                            s_next = ((((kn0 * K1 + kn1) * K1 + kn2) * K1 + kn3) * S + l_next) * PHASES + i_next
                            rows[nnz] = sa_idx
                            cols[nnz] = s_next
                            data[nnz] = p0123
                            nnz += 1

            cost[sa_idx] = cost_s
            sa_state[sa_idx] = s
            sa_idx += 1

        sa_start[s + 1] = sa_idx

    # trim oversized buffers to actual length
    return (
        rows[:nnz].copy(),
        cols[:nnz].copy(),
        data[:nnz].copy(),
        sa_state[:sa_idx].copy(),
        sa_start,
        cost[:sa_idx].copy(),
    )


def build_f4c2_kernel(K: int, q_in: np.ndarray):
    """Public wrapper. JIT-compiles the kernel on first call (cached afterwards)."""
    return _kernel_inner(int(K), np.ascontiguousarray(q_in, dtype=np.float64))
