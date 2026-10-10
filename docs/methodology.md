# Theoretical Methodology: $N$-Dimensional Random Walk

This document provides the mathematical framework and statistical mechanics theory powering the Random Walk Laboratory calculations. It details the exact theoretical formulations used to benchmark the numerical simulations.

## 1. The Core Model

The system simulates an independent, symmetric random walk on a $d$-dimensional hypercubic lattice $\mathbb{Z}^d$. 
At each discrete time step $t$, the walker takes a step $\vec{S}_t$. In this model, the walk is independent across all dimensions, meaning that for each coordinate axis $k \in \{1, \dots, d\}$, the displacement is:

$$ S_{t,k} = \begin{cases} +1 & \text{with probability } \frac{1}{2} \\ -1 & \text{with probability } \frac{1}{2} \end{cases} $$

The total position of the walker at time $T$ is the sum of all individual step vectors:

$$ \vec{X}(T) = \sum_{t=1}^T \vec{S}_t $$

Because the coordinates are independent, the squared magnitude of a single step vector $\vec{S}_t$ is precisely $\|\vec{S}_t\|^2 = \sum_{k=1}^d (\pm 1)^2 = d$.

---

## 2. Mean Squared Displacement (MSD) and Scaling

The most fundamental macroscopic observable of a random walk is how far the walker spreads out over time, quantified by the **Mean Squared Displacement (MSD)**.

By linearity of expectation and the independence of steps (i.e., $\mathbb{E}[\vec{S}_i \cdot \vec{S}_j] = 0$ for $i \neq j$), the MSD is:

$$ \text{MSD}(t) = \mathbb{E}[\|\vec{X}(t)\|^2] = \mathbb{E}\left[ \left\| \sum_{i=1}^t \vec{S}_i \right\|^2 \right] = \sum_{i=1}^t \mathbb{E}[\|\vec{S}_i\|^2] $$

Since each step has a squared magnitude of $d$, we arrive at the theoretical MSD:

$$ \mathbb{E}[R^2(t)] = d \cdot t $$

### The Scaling Exponent ($\alpha$)
Diffusion processes are characterized by their scaling exponent $\alpha$, defined by the relationship between the Root Mean Square (RMS) displacement and time:

$$ R_{\text{RMS}}(t) = \sqrt{\langle R^2(t) \rangle} \sim t^\alpha $$

For a standard random walk (Brownian motion), the theoretical scaling exponent is exactly **$\alpha = 0.5$**. 

---

## 3. Probability Distributions (The Continuous Limit)

While the random walk operates on a discrete lattice, for a large number of steps $t$, the spatial distribution of walkers is governed by the **Central Limit Theorem (CLT)**.

### 3.1. Single Coordinate Distribution
Along any single axis $x_k$, the position is the sum of $t$ independent random variables (each $\pm 1$). The mean is $\mu = 0$ and the variance is $\sigma^2 = t$. By the CLT, the probability density function (PDF) for the coordinate $x$ approaches a Gaussian distribution:

$$ \rho(x, t) = \frac{1}{\sqrt{2\pi t}} \exp\left( - \frac{x^2}{2t} \right) $$

### 3.2. Radial Distribution (Distance from Origin)
The radial distance from the origin is $r = \|\vec{X}\| = \sqrt{x_1^2 + x_2^2 + \dots + x_d^2}$.
Since each coordinate $x_k$ is an independent Gaussian $\mathcal{N}(0, t)$, the sum of their squares follows a generalized **Chi distribution** (scaled by $t$). 

The exact theoretical radial probability density function in $d$ dimensions is given by:

$$ P(r, t) = \frac{2}{2^{d/2} \, \Gamma(d/2) \, t^{d/2}} \, r^{d-1} \, \exp\left( -\frac{r^2}{2t} \right) $$

Where $\Gamma(z)$ is the Gamma function. For specific dimensions, this reduces to well-known distributions:
- **1D:** Half-normal distribution.
- **2D:** Rayleigh distribution.
- **3D:** Maxwell-Boltzmann distribution.

---

## 4. Pólya's Recurrence Theorem

A profound result in statistical mechanics is **Pólya's Random Walk Theorem** (1921), which answers the question: *Will the walker eventually return to its starting point?*

A walk is defined as **recurrent** if the probability of returning to the origin as $t \to \infty$ is exactly $1$. It is **transient** if the probability is strictly less than $1$.

- **$d = 1$ and $d = 2$:** The walk is **Recurrent**. A walker is mathematically guaranteed to return to the origin an infinite number of times. (Note: The expected *time* until the first return is infinite).
- **$d \ge 3$:** The walk is **Transient**. There is a finite probability that the walker wanders off to infinity and never returns.

### Theoretical Return Probabilities
The exact probability $P_{\text{return}}(d)$ that a walker on a $d$-dimensional cubic lattice ever returns to the origin is given by Watson's integrals. The theoretical values used for benchmarking are:
- $d = 1: 1.00000$
- $d = 2: 1.00000$
- $d = 3: \approx 0.340537$ (Pólya's Constant)
- $d = 4: \approx 0.193201$
- $d = 5: \approx 0.135178$

*(In the numerical simulations, a finite step cutoff $T$ is used, meaning the observed return fraction for 1D/2D will approach, but not reach, 1.0)*.

---

## 5. Unique Sites Visited

The number of distinct lattice sites $V(t)$ visited by the walker over time reveals the space-filling properties of the trajectory. The theoretical asymptotic expected value $\mathbb{E}[V(t)]$ for large $t$ depends heavily on the dimension:

- **1D:** $\mathbb{E}[V(t)] \sim \sqrt{\frac{8t}{\pi}}$ (The walk heavily overlaps itself, growing sub-linearly).
- **2D:** $\mathbb{E}[V(t)] \sim \frac{\pi t}{\ln(8t)}$ (Grows almost linearly, but logarithmically suppressed due to recurrence).
- **3D+:** $\mathbb{E}[V(t)] \sim (1 - P_{\text{return}}) \cdot t$ (Because the walk is transient, the number of new sites grows strictly linearly with time, scaled by the probability of escaping to infinity).
