# MDP Solution Algorithms — Notes Transcription

Transcribed from the handwritten lecture notes
(`source_pdf/MDP Notes.pdf`, lectures 1–2).
This document is the implementation spec: each algorithm appears in the
form actually used in class, with the supporting theorems that justify
its correctness or stopping rule.

---

## Preamble — setting and assumptions

We work with a discrete-time **Markov Decision Process** with

- a finite state space $\mathcal{X}$, $|\mathcal{X}| = n < \infty$,
- a finite action set $\mathcal{A}_x$ available in state $x$,
- a one-step **reward / cost** $r(x, a)$,
- transition probabilities $p(x, a, y) = \mathbb{P}(X_{t+1} = y \mid X_t = x, A_t = a)$.

A **stationary policy** is a map $R : \mathcal{X} \to \mathcal{A}$.

We focus on the **average cost-rate criterion**:
$$
g^R \;=\; \lim_{T \to \infty} \frac{1}{T}\,\mathbb{E}_R\!\left[\sum_{t=0}^{T-1} r(X_t, R(X_t)) \,\Big|\, X_0 = x\right].
$$

### Assumptions (notes pp. 4–5)

1. $|\mathcal{X}| < \infty$ (finite).
2. **Unichain**: there is at least one state $x^*$ such that there is a path
   from every state $y \in \mathcal{X}$ to $x^*$. Equivalently, the
   stationary distribution exists and is independent of the initial
   distribution.
3. **Aperiodic**: the gcd of all path lengths from $x$ back to itself is 1.
   *Strong aperiodicity*: under every policy $R$ and every state $x$,
   $p(x, R(x), x) > 0$.

Under (1)–(3), $\pi_t \xrightarrow{t \to \infty} \pi_*$ exists, satisfies
$\pi_* = \pi_* P$, $\pi_* \mathbf{e} = 1$, and is the long-run occupancy
distribution.

The **Poisson equation** for a fixed policy $R$ (notes p. 7):
$$
V(x) + g \;=\; r(x, R(x)) + \sum_{y \in \mathcal{X}} p(x, R(x), y)\, V(y),
\quad \forall x \in \mathcal{X},
$$
with normalisation $\sum_{x} \pi_*(x)\, V(x) = 0$ to fix the bias up to a
constant. $V(x)$ is the *relative value* — the difference in expected
total reward starting in $x$ vs. starting in steady state.

The **Bellman optimality equation** (notes p. 9):
$$
V^*(x) + g^* \;=\; \min_{a \in \mathcal{A}_x}\!\left\{ r(x, a) + \sum_{y} p(x, a, y)\, V^*(y) \right\}.
$$
Any policy attaining the min is optimal and $g^*$ is the optimal cost-rate.

### Strong-aperiodicity transformation (notes p. 11)

Value Iteration requires **strong aperiodicity**. If the original MDP has
some $p(x, R(x), x) = 0$, transform via $\tau \in (0, 1)$:

| | Original | Transformed |
|---|---|---|
| state space | $\mathcal{X}$ | $\bar{\mathcal{X}} = \mathcal{X}$ |
| actions | $\mathcal{A}_x$ | $\bar{\mathcal{A}}_x = \mathcal{A}_x$ |
| reward | $r(x, a)$ | $\bar{r}(x, a) = r(x, a)$ |
| transition | $p(x, a, y)$ | $\bar{p}(x, a, y) = \tau\, p(x, a, y) + (1 - \tau)\, \mathbf{1}\{x = y\}$ |

The transition probabilities *conditional on leaving* $x$ are identical.
The sojourn time in $x$ becomes geometric. Balance equations are unchanged
up to a factor of $\tau$, so $g^* = \bar{g}^*$ and the optimal policy is
the same.

---

## Algorithm 1 — Policy Iteration  *(notes pp. 8–9)*

**Idea.** Pick a policy, evaluate it (Poisson equations), greedy-improve,
repeat. Converges in finitely many iterations because the policy set is
finite and $g$ is monotone.

### Pseudocode

```
Input: MDP (X, A, p, r)
Output: optimal policy R*, optimal cost-rate g*

(0)  Pick any feasible stationary policy R.

(1)  POLICY EVALUATION.
     Solve the linear system in (V(x))_{x∈X} and g:

         V(x) + g  =  r(x, R(x)) + Σ_y p(x, R(x), y) · V(y),   ∀x ∈ X
         Σ_x π_*(x) · V(x)  =  0                              (normalisation)

     This is |X| + 1 equations in |X| + 1 unknowns. The normalisation row
     can be replaced by V(x_0) = 0 for an arbitrary reference state x_0
     (cost-rate g is unchanged; bias V differs only by a constant).

(2)  POLICY IMPROVEMENT.
     For each x ∈ X compute

         R'(x) = argmin_{a ∈ A_x} { r(x, a) + Σ_y p(x, a, y) · V(y) }.

(3)  STOPPING.
     If R' = R: STOP — R is optimal, return (R, g).
     Else:     R ← R', go to (1).
```

### Improvement Theorem (notes p. 9)

If a policy $R'$ satisfies the **Poisson inequality**
$$
r(x, R'(x)) - g + \sum_y p(x, R'(x), y)\, V(y) \;\le\; V(x), \quad \forall x,
$$
where $(g, V)$ are the cost-rate / bias of *some* reference policy $R$, then
$g^{R'} \le g$. The Step (2) update produces such an $R'$, hence $g$
weakly decreases each iteration.

**Convergence.** $|\mathcal{X}|$ and $|\mathcal{A}|$ finite ⇒ the policy
space is finite. $g$ is non-increasing along the iterates and a strict
decrease occurs whenever $R' \ne R$, so the algorithm halts in
$O(|\mathcal{A}|^{|\mathcal{X}|})$ iterations (in practice, very few).
Complexity per iteration is dominated by the Poisson solve, $O(n^3)$ for
the dense linear system (notes margin: $O(n^3)$).

---

## Algorithm 2 — Value Iteration  *(notes pp. 12–13)*

**Idea.** Iterate the Bellman *operator* on a value function. Despite its
finite-horizon flavour, the *increments* $V_n - V_{n-1}$ converge to $g^*$
under strong aperiodicity, giving a sandwich on $g^*$ with a clean stop
criterion.

### Pseudocode

```
Input: MDP (X, A, p, r), tolerance ε > 0  (e.g. 1e-6).
       (MDP must be STRONGLY APERIODIC — apply the τ-transformation if not.)
Output: near-optimal policy R_n, cost-rate estimates m_n ≤ g* ≤ M_n.

(0)  V_0 ≡ 0,  n ← 1.

(1)  For each x ∈ X:
         V_n(x) = min_{a ∈ A_x} { r(x, a) + Σ_y p(x, a, y) · V_{n-1}(y) }
         R_n(x) = argmin of the same.

(2)  m_n = min_{x ∈ X} (V_n(x) − V_{n-1}(x))
     M_n = max_{x ∈ X} (V_n(x) − V_{n-1}(x))

(3)  If 0 ≤ M_n − m_n ≤ ε · m_n:
         STOP — R_n is ε-optimal, g* ∈ [m_n, M_n].
     Else:
         n ← n + 1, go to (1).
```

### Stop-Criterion Theorem (notes p. 13)

For all $n \in \mathbb{N}$:
$$
m_n \;\le\; g^* \;\le\; g^{R_n} \;\le\; M_n,
$$
so $(M_n - m_n) / m_n$ bounds the relative gap of the current policy
$R_n$ to the optimal cost-rate.

*Proof sketch.* Since $m_n \le V_n(x) - V_{n-1}(x) \le M_n$ for all $x$,
combine with the Bellman recursion and apply the Improvement Theorem to
both $R_n$ and the optimal policy.

### Improved-Bounds Theorem (notes p. 13)

The bounds tighten monotonically:
$$
m_{n+1} \;\ge\; m_n, \qquad M_{n+1} \;\le\; M_n.
$$

So the sandwich $[m_n, M_n]$ shrinks; combined with strong aperiodicity
this guarantees $M_n - m_n \to 0$.

### Practical notes

- **Strong aperiodicity is required.** If the MDP is not strongly
  aperiodic, apply the $\tau$-transformation first (typical $\tau = 0.5$).
- **Implement in vectorised form** (NumPy / Cython): the Bellman update
  is a sparse tensor contraction $V_n[x] = \min_a \{r[x,a] + (P[x,a,:] \cdot V_{n-1})\}$.
- **Finite-horizon use.** The same recursion with terminal reward
  $V_0(x)$ solves a finite-horizon problem exactly; for $T$ large
  ($\sim$500), $V_T$ approximates the average-cost relative value
  function up to a constant.

---

## Quick comparison

| | Per-iteration cost | # iterations | Solver dep | Notes |
|---|---|---|---|---|
| Policy Iteration | $O(n^3)$ Poisson solve | very few (often $\le$ 10) | linear solver | Exact $g$ at each step; requires strict unichain. |
| Value Iteration  | $O(\sum_x \|\mathcal{A}_x\| \cdot \text{nnz}(P))$ | many but cheap | none | Easiest to code; ε-optimal; needs strong aperiodicity. |
