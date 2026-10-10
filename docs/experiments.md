# Experiments Documentation

## IDC612 (2026) — Random Walk Laboratory

---

## Experiment A: MSD vs Time

**Purpose:** Verify $E[R^2(t)] = t$ against simulation.

**Method:**
- Simulate ensemble of $N$ walkers for $n$ steps
- At each checkpoint $t$, compute $\langle R^2(t) \rangle = \frac{1}{N}\sum_{i=1}^N |X^{(i)}_t|^2$
- Plot against $t$ and theoretical line $y = t$
- Fit linear regression: slope should be $\approx 1$

**Expected result:** Simulation curve overlaps theory within Monte Carlo fluctuations.

---

## Experiment B: Scaling Exponent

**Purpose:** Measure diffusive scaling exponent $\alpha$.

**Method:**
- Compute $\langle |X_t| \rangle$ (mean displacement) at each time step
- Fit $\log(\langle|X_t|\rangle) = \alpha \log(t) + \text{const}$ via linear regression
- Compare fitted $\alpha$ with theoretical $0.5$

**Expected result:** $\hat{\alpha} \approx 0.5$, improving with more walkers and steps.

---

## Experiment C: Monte Carlo Convergence

**Purpose:** Demonstrate that $\widehat{\text{MSD}}_N \to n$ as $N \to \infty$.

**Method:**
- For walker counts $N \in \{10, 100, 1000, 10000, \ldots\}$:
  - Simulate $N$ walkers for $n$ steps
  - Compute $\widehat{\text{MSD}}_N = \frac{1}{N}\sum |X_n^{(i)}|^2$
  - Record absolute and relative error vs theory $n$
- Plot error vs $N$ on log-log axes
- Overlay $1/\sqrt{N}$ reference line

**Expected result:** Error decreases approximately as $1/\sqrt{N}$.

---

## Experiment D: Coordinate Distribution

**Purpose:** Verify CLT prediction $X_i \to \mathcal{N}(0, n/d)$.

**Method:**
- After $n$ steps, collect coordinate $x_1$ of all $N$ walkers
- Plot histogram (density normalized)
- Overlay $\mathcal{N}(0, n/d)$ PDF
- Perform Kolmogorov-Smirnov test

**Expected result:** KS p-value $> 0.05$ for large $n$, indicating Gaussian fit is acceptable.

---

## Experiment E: Radial Distribution

**Purpose:** Compare empirical $|X_n|$ distribution with chi distribution.

**Method:**
- Compute $|X^{(i)}_n|$ for all walkers
- Plot histogram vs $\chi(d, \sigma=\sqrt{n/d})$ PDF

**Expected result:** Good agreement for large $n$; better in high dimensions.

---

## Experiment F: Recurrence / Transience

**Purpose:** Demonstrate the Pólya recurrence/transience transition at $d = 3$.

**Method:**
- For each $d \in \{1, 2, 3, 4, 5\}$:
  - Simulate $N$ walkers for $n$ steps with return tracking
  - Record fraction of walkers that returned at least once
  - Compare with theoretical infinite-time probabilities

**Expected result:**
- 1D/2D: high return fraction (approaches 1 as $n \to \infty$)
- 3D/4D/5D: return fraction noticeably $< 1$

**Caveat:** Finite simulation underestimates return fraction for recurrent walks.

---

## Experiment G: Distinct Sites

**Purpose:** Count $S_n$ = number of distinct lattice sites visited.

**Method:**
- Track set of visited positions as a Python `set` of tuples
- Plot $S_n$ vs $n$

**Observations:**
- 1D: $S_n \sim O(\sqrt{n})$ by known theory
- 2D: $S_n \sim O(n / \log n)$
- 3D+: $S_n \sim O(n)$ (linear growth, fewer revisits)

---

## Experiment H: First Passage Time

**Purpose:** Measure first time the walk reaches $|X| \geq L$.

**Method:**
- For each walker: simulate step-by-step until $|X_t| \geq L$ or $t = T_{\max}$
- Record first passage time (None if not reached)
- Plot histogram; compute mean, median, quantiles

**Caveat:** Walkers that do not reach target within $T_{\max}$ are right-censored.

---

## Experiment I: Biased Walk

**Purpose:** Compare drift vs diffusion.

**Method:**
- For unbiased ($p=0.5$) and biased ($p > 0.5$) walks:
  - Simulate ensemble
  - Compare histograms of $x_1$ coordinate
  - Compare mean displacement with theory

**Theory:**
- Unbiased: $E[X_1] = 0$, displacement $\sim \sqrt{n}$
- Biased: $E[X_1] = n \cdot \frac{1}{d} \cdot (2p-1)$, drift dominates for large $n$

---

## Dimension Comparison Summary

| Quantity | 1D | 2D | 3D | 4D | 5D |
|----------|----|----|----|----|-----|
| MSD = n | ✓ | ✓ | ✓ | ✓ | ✓ |
| Scaling α = 0.5 | ✓ | ✓ | ✓ | ✓ | ✓ |
| Recurrent | ✓ | ✓ | ✗ | ✗ | ✗ |
| Return prob (theory) | 1.0 | 1.0 | 0.34 | 0.19 | 0.14 |
| Sites grow linearly | ✗ | ✗ | ✓ | ✓ | ✓ |
| Gaussian coords (CLT) | ✓ | ✓ | ✓ | ✓ | ✓ |
