# Traffic Control at an Isolated Intersection — Paper Brief

Reference: **Haijema & van der Wal (2008).** *An MDP Decomposition Approach
for Traffic Control at Isolated Signalized Intersections.* Probability in
the Engineering and Informational Sciences, 22(4), 587–602.
PDF in `source_pdf/AN MDP DECOMPOSITION APPROACH...pdf`.

---

## 1. Introduction

The paper studies the *dynamic* control of traffic lights at a single,
isolated signalised intersection. The controller observes per-lane queue
lengths (from loop detectors / cameras) at the start of each time slot,
and must decide *when* to switch the light from green to yellow, and
*which* group of non-conflicting flows gets the next green.

Two classical baselines exist: **fixed-cycle (FC)** control, which fixes
both the order *and* the green durations, and **exhaustive (XH)** control,
which keeps a green active until all served queues are empty. The paper's
contribution is a discrete-time **MDP formulation** of the dynamic problem
plus a one-step **policy-improvement** step on top of the optimal FC,
yielding the **Relative-Value-Cyclic (RVC)** heuristic that scales to
intersections far too large for direct value iteration. For *small*
intersections the MDP is solved exactly, and that is the regime
relevant to the homework.

## 2. The MDP

### 2.1 State

$$
s = (\underline{k},\, x), \qquad \underline{k} = (k_1, \dots, k_F),\;\; x = (l, i)
$$

- $F$ = number of independent traffic flows (lanes/turns).
- $k_f \in \{0, 1, \dots, K\}$ = number of cars queued in flow $f$ at the
  start of the slot (truncated at buffer cap $K$ for finiteness).
- The flows are partitioned into $S$ disjoint **combinations**
  $C_1, \dots, C_S$ (sets of non-conflicting flows that can hold green
  together).
- $l \in \{1, \dots, S\}$ = current/most-recent combination index.
- $i \in \{0, 1, 2, 3\}$ = signal phase: $0$ green, $1$ first yellow,
  $2$ second yellow, $3$ all-red.

The slot length is **2 s** (one car-pass time at typical urban speeds).

### 2.2 Actions

Phase-dependent decision space $\mathcal{A}((\underline{k}, (l, i)))$:

$$
\mathcal{A} \;=\;
\begin{cases}
\{(l, 0),\, (l, 1)\} & i = 0 \text{ (green: keep or change to yellow)} \\
\{(l, i+1)\}        & i \in \{1, 2\} \text{ (forced advance)} \\
\{(l, 3),\, (l', 0)\} & i = 3 \text{ (all-red: stay or switch to next non-empty combination $l'$)}
\end{cases}
$$

When **all queues are empty** the lights freeze (special action), so the
all-zero state is absorbing-until-arrival.

### 2.3 Transitions

Per-flow Bernoulli arrivals (probability $q_f$ per slot), deterministic
1-car-per-slot departures during green and yellow. With $y^+ = \max(y, 0)$:

- If action $a$ gives **green/yellow** to flow $f$:
  $p_f(k_f, a, (k_f - 1)^+) = 1 - q_f, \quad p_f(k_f, a, k_f) = q_f$.
- If action $a$ gives **red** to flow $f$:
  $p_f(k_f, a, k_f) = 1 - q_f, \quad p_f(k_f, a, k_f + 1) = q_f$.

Flows are independent given the action, so the joint transition
factorises:
$$
p(\underline{k}, x;\, \underline{k}', a) \;=\; \prod_{f=1}^{F} p_f(k_f, a, k_f').
$$

(At the buffer cap $K$, an arrival is *rejected* and a fixed externality
penalty is added to the cost — paper §3.5.)

### 2.4 Cost

Per-slot cost = total cars in the system:
$$
c(\underline{k}, x) \;=\; \sum_{f=1}^{F} k_f.
$$

By **Little's law**, minimising the long-run average of $c$ is equivalent
to minimising the average waiting time per car.

### 2.5 Unichain & aperiodic

- **Unichain.** From any state, under any policy, there is positive
  probability of reaching $\underline{k} = \mathbf{0}$ within finitely many
  slots (every flow eventually gets green or has zero arrivals). Hence
  $(\underline{0}, \cdot)$ is a recurrent state reachable from every other
  state.
- **Aperiodic.** *Not* trivially aperiodic: the deterministic
  yellow → all-red → green sequence creates fixed-length cycles. We make
  the MDP **strongly aperiodic** by the $\tau$-transformation (notes p. 11)
  with $\tau = 0.5$ before running Value Iteration.

### 2.6 Finite-state reduction

Truncate each $k_f$ at $K$ (buffer cap). Choose $K$ large enough that
the rejection probability at the optimum is negligible (paper §3.5
recommends extrapolating relative values past the cap if rejection is
non-trivial — for the small homework cases at $\rho \le 0.8$, $K = 15$
is generous).

State count for **F4C2 with $K = 15$**:
$(K+1)^4 \times S \times 4 = 16^4 \times 2 \times 4 = 524{,}288$ states.
Reduced by the fact that only one combination is "live" per state and the
yellow / all-red phases have collapsed action sets — the *reachable* state
space is much smaller in practice. Manageable for both VI and PI.

---

## 3. Replicable Results

The paper benchmarks two infrastructures (Fig. 3):

- **F4C2** — 4 flows in 2 symmetric combinations, $C_1 = \{1, 3\}$,
  $C_2 = \{2, 4\}$. Switching cost = 3 slots (2 yellow + 1 all-red).
  Small enough to solve the MDP exactly. **This is our target.**
- **F12C4** — 12 flows in 4 asymmetric combinations. The paper itself
  states "computation of the optimal MDP strategy is practically
  impossible". **Out of scope.**

### Target — Table 1 (symmetric F4C2), MDP-optimal cyclic strategy

Mean overall waiting time per car $E[W]$ in seconds, identical arrival
probabilities $q_f$ for all flows.

| $\rho$ | $q_f$ per flow | $E[W]$ (MDP-optimal) | FC baseline | XHC baseline | FC cycle / departure slots |
|---|---|---|---|---|---|
| 0.40 | 0.20 | **4.89** s | 5.43 s (+11%) | 5.76 s (+18%) | 16 s, (6, 6) |
| 0.60 | 0.30 | **6.95** s | 8.27 s (+19%) | 8.82 s (+27%) | 24 s, (10, 10) |
| 0.80 | 0.40 | **13.5** s | 17.0 s (+26%) | 19.9 s (+47%) | 44 s, (20, 20) |

The bolded MDP column is what our VI/PI solver should reproduce. FC and
XHC are useful sanity baselines (cheap to simulate; they should bracket
our optimal from above).

### Stretch — Table 2 (asymmetric F4C2 at $\rho = 0.6$)

Two interesting asymmetric cases at $\rho = 0.6$:

| Case | $\underline{q}$ | $E[W]$ overall (MDP-optimal) |
|---|---|---|
| 2:1 by combination | $(0.15,\, 0.45,\, 0.15,\, 0.45)$ | 5.9 s |
| Single light flow | $(0.10,\, 0.30,\, 0.30,\, 0.30)$ | 6.3 s |

These re-use exactly the same model with different $q_f$ — perfect for
the homework's part (f) sensitivity study.

---

## 4. Implementation Plan — Value Iteration + Policy Iteration

We replicate the Table 1 row-by-row, with both algorithms, and
cross-check that they agree. Plan:

1. **Model layer** (`src/model.py` or similar — to be decided when
   coding starts).
   - Encode state $(k_1, k_2, k_3, k_4, l, i)$ as a flat integer via
     mixed-radix packing with cap $K$. Total slots: $(K+1)^4 \cdot S \cdot 4$.
   - Build the action set $\mathcal{A}(s)$ from the §2.2 case analysis.
   - Build the transition kernel **lazily** as a list-of-lists sparse
     structure (`scipy.sparse.csr_matrix`, one per action), exploiting
     per-flow factorisation: a state has at most $2^F = 16$ successors
     per action.

2. **Aperiodicity fix.** Apply the strongly-aperiodic transformation
   $\bar{p}(s, a, s') = \tau\, p(s, a, s') + (1 - \tau)\, \mathbf{1}\{s = s'\}$
   with $\tau = 0.5$. This is required for Value Iteration; Policy
   Iteration tolerates the original kernel but using the same
   transformed kernel keeps results directly comparable. Justification
   for part (c) of the homework writeup.

3. **Value Iteration** (notes Algorithm 3).
   - $V_0 \equiv 0$, vectorised Bellman update via sparse matvecs.
   - Stop on relative span $(M_n - m_n) / m_n \le 10^{-6}$.
   - Output: $\hat{g}^* \in [m_n, M_n]$, near-optimal policy $R_n$.
   - Convert $\hat{g}^*$ (avg cars per slot) to $E[W]$ in seconds:
     $E[W] = \hat{g}^* \cdot 2\,\text{s} \,/\, \sum_f q_f$
     by Little's law (slot length 2 s).

4. **Policy Iteration** (notes Algorithm 1).
   - Initialise with the FC policy (gives a meaningful warm start and
     also gives us the FC baseline column).
   - Policy evaluation: build the $|\mathcal{X}_{\text{reach}}| \times |\mathcal{X}_{\text{reach}}|$
     transition matrix under the current $R$, append the normalisation
     $V(s_0) = 0$, solve with `scipy.sparse.linalg.spsolve`.
   - Policy improvement: greedy on $r + P V$.
   - Stop when the policy is stable.

5. **Cross-check (homework part (e)).** For each $\rho \in \{0.4, 0.6, 0.8\}$
   assert $|g_{\text{VI}} - g_{\text{PI}}| < 10^{-5}$ and the two policies
   agree on every recurrent state.

6. **Sensitivity sweep (homework part (f)).** Sweep $\rho \in [0.30, 0.85]$
   in steps of 0.05 (with identical $q_f$). Plot:
   - $\hat{g}^*$ vs. $\rho$,
   - $E[W]$ vs. $\rho$ alongside the FC baseline,
   - one diagnostic of the policy — e.g. the smallest queue size at
     which the optimal policy switches the light, as a function of
     $\rho$.

### Effort estimate

Modelling + VI is the bulk of the work (~½ day). PI on top is small
once the kernel is built (~2 hours). The sensitivity sweep and plots
are mostly waiting on the solver.

### Validation criteria

- VI and PI agree on $\hat{g}^*$ to 5 decimals.
- The three Table-1 numbers (4.89 / 6.95 / 13.5 s) match within ≈0.1 s
  of the paper after the slot→seconds conversion. Any larger gap is
  almost certainly the buffer cap $K$ being too small or a sign error
  in the cost.
- The optimal policy is *cyclic* (it visits combinations in a fixed
  order $1, 2, 1, 2, \dots$) — the paper notes the optimal MDP strategy
  *is* cyclic for the symmetric F4C2.
