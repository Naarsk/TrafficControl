# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cython implementation of the F4C2 MDP transition-kernel build.

Mirrors ``F4C2Model._build_python`` in ``src/model.py`` but with full
C-typing, no Python object allocations in the inner loops, and no
``itertools.product``. Hard-codes F=4, S=2 (the F4C2 paper case).

Returns the *raw* (untransformed) kernel triplets; the
strongly-aperiodicity ``tau``-transform is applied in Python afterwards
via the existing CSR arithmetic, so this kernel is bit-equivalent to
the existing Python implementation up to floating-point summation
order.
"""

import numpy as np
cimport numpy as cnp


def build_f4c2_kernel(int K, cnp.ndarray[double, ndim=1] q_in):
    """Build the raw (untransformed) F4C2 transition kernel triplets.

    Parameters
    ----------
    K : int
        Per-flow buffer cap.
    q_in : ndarray of shape (4,), dtype float64
        Per-flow Bernoulli arrival probabilities.

    Returns
    -------
    rows, cols, data : ndarray (int64, int64, float64)
        Triplets for sparse CSR construction. Length == nnz_actual.
    sa_state : ndarray (int64), shape (n_sa,)
        State index for each (state, action) pair.
    sa_start : ndarray (int64), shape (n_states + 1,)
        CSR-style segmentation: actions of state s are
        sa_start[s] : sa_start[s + 1].
    cost : ndarray (float64), shape (n_sa,)
        One-step cost for each (state, action) pair.
    """
    cdef:
        int F = 4
        int S = 2
        int PHASES = 4
        int K1 = K + 1
        long n_states = (<long>K1) * K1 * K1 * K1 * S * PHASES

        # F4C2 active mask: combo 0 = {0, 2}, combo 1 = {1, 3}
        int[2][4] active_mask

        # max sizes for upper-bound preallocation
        long max_sa = n_states * 2
        long max_nnz = n_states * 2 * 16

        # output buffers
        cnp.ndarray[long, ndim=1] rows_arr = np.empty(max_nnz, dtype=np.int64)
        cnp.ndarray[long, ndim=1] cols_arr = np.empty(max_nnz, dtype=np.int64)
        cnp.ndarray[double, ndim=1] data_arr = np.empty(max_nnz, dtype=np.float64)
        cnp.ndarray[long, ndim=1] sa_state_arr = np.empty(max_sa, dtype=np.int64)
        cnp.ndarray[double, ndim=1] cost_arr = np.empty(max_sa, dtype=np.float64)
        cnp.ndarray[long, ndim=1] sa_start_arr = np.empty(n_states + 1, dtype=np.int64)

        # C-typed memory views for nogil access
        long[:] rows_v = rows_arr
        long[:] cols_v = cols_arr
        double[:] data_v = data_arr
        long[:] sa_state_v = sa_state_arr
        double[:] cost_v = cost_arr
        long[:] sa_start_v = sa_start_arr

        # local copies of arrival probs in a fixed-size C array
        double q_arr[4]

        long sa_idx = 0
        long nnz = 0
        long s, tmp, s_next
        int k0, k1, k2, k3
        int kk[4]
        int l, ph
        int all_zero
        int n_actions, ai
        int actions_l[2]
        int actions_i[2]
        int l_next, i_next
        int can_dep[4]
        int dist_lo[4]
        int dist_hi[4]
        double dist_plo[4]
        double dist_phi[4]
        int o0, o1, o2, o3
        int kn0, kn1, kn2, kn3
        double p0, p01, p012, p0123
        double cost_s
        int f

    q_arr[0] = q_in[0]; q_arr[1] = q_in[1]; q_arr[2] = q_in[2]; q_arr[3] = q_in[3]

    active_mask[0][0] = 1; active_mask[0][1] = 0; active_mask[0][2] = 1; active_mask[0][3] = 0
    active_mask[1][0] = 0; active_mask[1][1] = 1; active_mask[1][2] = 0; active_mask[1][3] = 1

    sa_start_v[0] = 0

    with nogil:
        for s in range(n_states):
            # decode state s: mixed-radix unpacking of (k0,k1,k2,k3,l,ph)
            tmp = s
            ph = <int>(tmp % PHASES); tmp //= PHASES
            l = <int>(tmp % S); tmp //= S
            k3 = <int>(tmp % K1); tmp //= K1
            k2 = <int>(tmp % K1); tmp //= K1
            k1 = <int>(tmp % K1); tmp //= K1
            k0 = <int>(tmp % K1)

            kk[0] = k0; kk[1] = k1; kk[2] = k2; kk[3] = k3
            all_zero = (k0 == 0) and (k1 == 0) and (k2 == 0) and (k3 == 0)
            cost_s = <double>(k0 + k1 + k2 + k3)

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
                        can_dep[f] = active_mask[l_next][f]

                # per-flow distributions: paper formula 3.3
                #   active flow:  k -> (k-1)^+ w.p. 1-q, k w.p. q
                #   red flow:     k -> k w.p. 1-q, k+1 w.p. q (capped at K)
                for f in range(F):
                    if can_dep[f]:
                        if kk[f] == 0:
                            dist_lo[f] = 0;       dist_plo[f] = 1.0 - q_arr[f]
                            dist_hi[f] = 0;       dist_phi[f] = q_arr[f]
                        else:
                            dist_lo[f] = kk[f] - 1; dist_plo[f] = 1.0 - q_arr[f]
                            dist_hi[f] = kk[f];     dist_phi[f] = q_arr[f]
                    else:
                        if kk[f] >= K:
                            dist_lo[f] = K;       dist_plo[f] = 1.0
                            dist_hi[f] = K;       dist_phi[f] = 0.0
                        else:
                            dist_lo[f] = kk[f];     dist_plo[f] = 1.0 - q_arr[f]
                            dist_hi[f] = kk[f] + 1; dist_phi[f] = q_arr[f]

                # 4 nested loops over per-flow outcomes -- 2^4 = 16 successors
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
                                # encode next state
                                s_next = (((((<long>kn0) * K1 + kn1) * K1 + kn2) * K1 + kn3) * S + l_next) * PHASES + i_next
                                rows_v[nnz] = sa_idx
                                cols_v[nnz] = s_next
                                data_v[nnz] = p0123
                                nnz += 1

                cost_v[sa_idx] = cost_s
                sa_state_v[sa_idx] = s
                sa_idx += 1

            sa_start_v[s + 1] = sa_idx

    # Trim oversized buffers to actual length and return.
    return (
        rows_arr[:nnz].copy(),
        cols_arr[:nnz].copy(),
        data_arr[:nnz].copy(),
        sa_state_arr[:sa_idx].copy(),
        sa_start_arr,
        cost_arr[:sa_idx].copy(),
    )
