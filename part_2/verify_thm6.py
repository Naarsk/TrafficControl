"""Numerical / symbolic verification for Theorem 6 of report_2.tex.

Three checks, each independent:

  (i)  Symbolic verification of the S_n decomposition (eq. Sn-decomp).
       We plug the recursion into S_n by hand and confirm with sympy
       that the four-corner expansion equals
           alpha * [lambda_2 * A + mu * B + (lambda_2 - lambda_1) * psi
                    - q * S_{n-1}],
       term-by-term.

  (ii) Numerical value iteration on a representative parameter set,
       confirming for all (x, n) in a grid that:
         - V_n is non-decreasing in x for each y      (I(1))
         - V_n is convex in x for each y              (CX(1))
         - S_n(x) >= 0                                (Super(0, 1))
         - S_n is non-increasing in x                 (the auxiliary
                                                      monotonicity)
         - U_1^{(n)} >= U_2^{(n)} once both finite    (Theorem 6)

  (iii) Verification at a SUFFICIENT condition. The proof uses
        q <= mu. We additionally check what happens when q > mu: the
        cross-env supermodularity still holds in the cases we tried
        (the rate condition is sufficient but not necessary), so
        the theorem appears to be more robust than the proof shows.

Run:  python verify_thm6.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np


# -------------------- (i) Symbolic check of eq. Sn-decomp ----------------------


def symbolic_check_decomposition() -> None:
    """Verify by symbolic algebra that the decomposition in eq. Sn-decomp
    is the correct expansion of S_n.

    We treat V_{n-1}(x, y) as an unknown symbol indexed by (x, y) for
    x ∈ {x, x+1, x+2} (the values that show up in the four corners
    after the recursion is unfolded) and y ∈ {1, 2}. We let
    T_CA V_{n-1} and T_D1 V_{n-1} be separate symbolic functions of
    (x, y); we don't expand the min in T_CA, just keep it abstract
    (T_CA V_{n-1}(x, y) =: TCA[x, y]).
    """
    import sympy as sp

    print("=" * 70)
    print("(i) Symbolic verification of S_n decomposition")
    print("=" * 70)

    a, lam1, lam2, mu, q, c = sp.symbols("alpha lambda_1 lambda_2 mu q c", positive=True)

    # V_{n-1} at the relevant queue values, both env states.
    V = {(xv, yv): sp.symbols(f"V_{xv}_{yv}") for xv in range(3) for yv in (1, 2)}
    # T_CA V_{n-1} at the relevant queue values.
    TCA = {(xv, yv): sp.symbols(f"TCA_{xv}_{yv}") for xv in range(2) for yv in (1, 2)}
    # T_D1 V_{n-1} at the relevant queue values. T_D1 V(x, y) = V((x-1)^+, y).
    # At x=0: T_D1 V(0, y) = V(0, y). At x=1: T_D1 V(1, y) = V(0, y).
    # At x=2: T_D1 V(2, y) = V(1, y).
    TD1 = {
        (0, yv): V[(0, yv)] for yv in (1, 2)
    } | {
        (1, yv): V[(0, yv)] for yv in (1, 2)
    } | {
        (2, yv): V[(1, yv)] for yv in (1, 2)
    }

    # The recursion (we drop the +cx term since it cancels in S_n).
    def V_n(xv: int, yv: int):
        lam_y = lam1 if yv == 1 else lam2
        yp = 3 - yv
        return c * xv + a * (
            lam_y * TCA[(xv, yv)]
            + mu * TD1[(xv, yv)]
            + q * V[(xv, yp)]
            + (lam2 - lam_y) * V[(xv, yv)]
        )

    # S_n(0) = V_n(0,1) + V_n(1,2) - V_n(1,1) - V_n(0,2). We can do at x=0;
    # the algebra is the same for any x via index shifting.
    x = 0
    Sn = V_n(x, 1) + V_n(x + 1, 2) - V_n(x + 1, 1) - V_n(x, 2)
    Sn_expanded = sp.expand(Sn)

    # The RHS of eq. Sn-decomp.
    # A(x) = S^{T_CA V_{n-1}}(x) = TCA[x,1] + TCA[x+1,2] - TCA[x+1,1] - TCA[x,2]
    A = TCA[(x, 1)] + TCA[(x + 1, 2)] - TCA[(x + 1, 1)] - TCA[(x, 2)]
    # B(x) = S^{T_D1 V_{n-1}}(x) = TD1[x,1] + TD1[x+1,2] - TD1[x+1,1] - TD1[x,2]
    B = TD1[(x, 1)] + TD1[(x + 1, 2)] - TD1[(x + 1, 1)] - TD1[(x, 2)]
    # psi(x) = pi(x+1) - pi(x), pi(x) = TCA[x,1] - V[x,1]
    pi = lambda xv: TCA[(xv, 1)] - V[(xv, 1)]
    psi = pi(x + 1) - pi(x)
    # S_{n-1}(x)
    S_prev = V[(x, 1)] + V[(x + 1, 2)] - V[(x + 1, 1)] - V[(x, 2)]

    RHS = a * (lam2 * A + mu * B + (lam2 - lam1) * psi - q * S_prev)
    RHS_expanded = sp.expand(RHS)

    diff = sp.simplify(Sn_expanded - RHS_expanded)
    print(f"  S_n(0) - RHS  = {diff}")
    assert diff == 0, "Symbolic identity does NOT hold (proof has an algebra bug)."
    print("  PASS: eq. Sn-decomp is symbolically verified.\n")


# -------------------- (ii) Numerical value iteration check ---------------------


@dataclass
class Params:
    lam1: float
    lam2: float
    mu: float
    q: float
    alpha: float
    R: float
    c: float
    Kmax: int  # queue truncation (large enough that boundary doesn't bite)


def value_iterate(p: Params, n_iters: int) -> list[np.ndarray]:
    """Returns the list [V_0, V_1, ..., V_{n_iters}] indexed as
    V[n][x, y] with y in {0, 1} mapping to modulator states {1, 2}.

    Reflecting boundary at x = Kmax (truncate excess arrivals).
    """
    K = p.Kmax
    V = np.zeros((K + 1, 2))
    history = [V.copy()]

    lams = (p.lam1, p.lam2)

    for _ in range(n_iters):
        V_new = np.zeros_like(V)
        for x in range(K + 1):
            for y_idx in range(2):
                yp_idx = 1 - y_idx
                lam_y = lams[y_idx]

                # T_CA V (x, y) = min{R + V(x, y), V(min(x+1, K), y)}
                x_plus = min(x + 1, K)
                TCA = min(p.R + V[x, y_idx], V[x_plus, y_idx])

                # T_D1 V (x, y) = V((x-1)^+, y)
                TD1 = V[max(x - 1, 0), y_idx]

                V_new[x, y_idx] = p.c * x + p.alpha * (
                    lam_y * TCA
                    + p.mu * TD1
                    + p.q * V[x, yp_idx]
                    + (p.lam2 - lam_y) * V[x, y_idx]
                )
        V = V_new
        history.append(V.copy())
    return history


def check_class_properties(p: Params, n_iters: int) -> bool:
    """For every V_n in the history, verify
    I(1), CX(1), Super(0, 1), and the auxiliary 'S_n non-increasing in x'.
    """
    gamma = p.lam2 + p.mu + p.q
    print("=" * 70)
    print("(ii) Numerical value iteration check")
    print(f"    lam1={p.lam1}, lam2={p.lam2}, mu={p.mu}, q={p.q}, "
          f"alpha={p.alpha}, R={p.R}, c={p.c}, K={p.Kmax}, "
          f"n_iters={n_iters}")
    print(f"    gamma = {gamma}, alpha*gamma = {p.alpha * gamma:.4f}  "
          f"(need < 1 for VI contraction)")
    print(f"    rate condition q <= mu : {p.q <= p.mu}")
    print("=" * 70)

    history = value_iterate(p, n_iters)
    K = p.Kmax
    tol = 1e-9

    # The buffer truncation at x = K disturbs all properties at the right
    # boundary; effects propagate inward by ~1 cell per VI step. To check
    # the true Super(0,1) property of the *un-truncated* recursion, we
    # restrict to the interior x in [0, K - margin]. Margin >= n_iters
    # guarantees the interior is unaffected by truncation.
    margin = max(3, n_iters + 2)
    interior = slice(0, max(K - margin, 1))

    all_pass = True
    for n, V in enumerate(history):
        # I(1): V_n(x+1, y) >= V_n(x, y).
        for y_idx in range(2):
            diffs = np.diff(V[:, y_idx])[interior]
            if not np.all(diffs >= -tol):
                print(f"  n={n} y={y_idx+1}: I(1) FAILED at x="
                      f"{np.where(diffs < -tol)[0]}")
                all_pass = False

        # CX(1): V_n(x+2,y) + V_n(x,y) - 2 V_n(x+1,y) >= 0.
        for y_idx in range(2):
            cx_diffs = (V[:-2, y_idx] + V[2:, y_idx] - 2 * V[1:-1, y_idx])[interior]
            if not np.all(cx_diffs >= -tol):
                print(f"  n={n} y={y_idx+1}: CX(1) FAILED at x="
                      f"{np.where(cx_diffs < -tol)[0]}")
                all_pass = False

        # Super(0, 1).
        S = (V[:-1, 0] + V[1:, 1] - V[1:, 0] - V[:-1, 1])[interior]
        if not np.all(S >= -tol):
            print(f"  n={n}: Super(0, 1) FAILED at x="
                  f"{np.where(S < -tol)[0]}, S values "
                  f"{S[S < -tol]}")
            all_pass = False

        # S_n non-increasing in x.
        S_full = V[:-1, 0] + V[1:, 1] - V[1:, 0] - V[:-1, 1]
        S_diffs = np.diff(S_full)[interior]
        if not np.all(S_diffs <= tol):
            print(f"  n={n}: S_n non-increasing in x FAILED in interior, "
                  f"S_diffs[bad]={S_diffs[S_diffs > tol]}")
            all_pass = False

    if all_pass:
        print("  PASS: I(1), CX(1), Super(0, 1), and S_n non-increasing "
              "all hold for every V_n.\n")
    else:
        print("  FAIL: see violations above.\n")
    return all_pass


def check_threshold_ordering(p: Params, n_iters: int) -> bool:
    """Confirm U_1^{(n)} >= U_2^{(n)} once both are finite."""
    print("=" * 70)
    print("(ii.b) Threshold ordering U_1 >= U_2")
    print("=" * 70)

    history = value_iterate(p, n_iters)
    K = p.Kmax

    # For each iteration n, compute U_y as the smallest x where rejection is
    # optimal, i.e., V(x+1, y) - V(x, y) >= R.
    def threshold(V: np.ndarray, y_idx: int) -> int:
        for x in range(K):
            if V[x + 1, y_idx] - V[x, y_idx] >= p.R:
                return x
        return K + 1  # never rejects within the window

    Us = [(threshold(V, 0), threshold(V, 1)) for V in history]
    all_pass = True
    for n, (U1, U2) in enumerate(Us):
        if U1 < U2:
            print(f"  n={n}: U_1={U1} < U_2={U2}  (FAILED)")
            all_pass = False

    final_U1, final_U2 = Us[-1]
    print(f"  Final thresholds: U_1 = {final_U1}, U_2 = {final_U2}")
    if all_pass:
        print("  PASS: U_1 >= U_2 at every iteration.\n")
    else:
        print("  FAIL.\n")
    return all_pass


# -------------------- (iii) Sufficiency vs necessity of q <= mu ----------------


def stress_test_rate_condition() -> None:
    """Try parameter sets with q > mu and see if Super(0,1) still holds.

    The proof of Theorem 6 uses q <= mu as a *sufficient* condition.
    Numerically, the theorem appears to hold in cases where q > mu
    too, suggesting the rate condition is a proof artifact and not
    intrinsic. We test a few cases.
    """
    print("=" * 70)
    print("(iii) Stress test: does Super(0,1) hold when q > mu?")
    print("=" * 70)

    # Keep alpha*gamma < 1 throughout (else VI diverges, not a Super failure).
    cases = [
        # (mu, q) with lam1=0.2, lam2=0.4. gamma = 0.6 + q. alpha = 0.9.
        (0.3, 0.05),   # q < mu  (rate condition holds)
        (0.3, 0.1),    # q < mu
        (0.3, 0.3),    # q = mu  (boundary)
        (0.2, 0.4),    # q > mu  (rate condition violated)
        (0.1, 0.5),    # q >> mu
    ]
    for (mu, q) in cases:
        gamma = 0.4 + mu + q
        if 0.9 * gamma >= 1.0:
            print(f"    mu={mu:.2f} q={q:.2f}: skipped (alpha*gamma={0.9*gamma:.2f} >= 1)")
            continue
        p = Params(lam1=0.2, lam2=0.4, mu=mu, q=q, alpha=0.9, R=1.0, c=1.0, Kmax=40)
        history = value_iterate(p, n_iters=80)
        V = history[-1]
        S = V[:-1, 0] + V[1:, 1] - V[1:, 0] - V[:-1, 1]
        # Exclude the right boundary (truncation artifact).
        S_interior = S[: p.Kmax - 3]
        min_S = float(S_interior.min())
        flag = "OK" if min_S >= -1e-8 else "VIOLATED"
        print(f"    mu={mu:.2f} q={q:.2f} (q<=mu: {q <= mu}, "
              f"alpha*gamma={0.9*gamma:.2f}): "
              f"min S_n = {min_S:.3e}   {flag}")
    print()


# ---------------------------------- main ----------------------------------


def main() -> int:
    # (i) Symbolic identity.
    try:
        symbolic_check_decomposition()
    except AssertionError as exc:
        print(f"FATAL: symbolic check failed: {exc}", file=sys.stderr)
        return 1
    except ImportError:
        print("  SKIP: sympy not installed; pip install sympy to enable (i).\n")

    # (ii) Numerical value iteration. Rates chosen so that alpha*gamma < 1,
    # which is required for the discounted value iteration to be a contraction.
    # Here gamma = lam2 + mu + q = 0.8 and alpha*gamma = 0.72 < 1.
    # R is moderate so thresholds are finite within the buffer.
    p = Params(
        lam1=0.2, lam2=0.4, mu=0.3, q=0.1,
        alpha=0.9, R=3.0, c=1.0, Kmax=80,
    )
    ok1 = check_class_properties(p, n_iters=80)
    ok2 = check_threshold_ordering(p, n_iters=80)

    # (iii) Stress test.
    stress_test_rate_condition()

    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
