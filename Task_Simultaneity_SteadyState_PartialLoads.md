# Task: Steady-State Simultaneity via Partial Load Ratios

> **Prerequisite:** This document assumes familiarity with the concepts from *Task_Simultaneity_TimeShift_Optimization.md*.

## 1. Motivation

In steady-state simulations, there is no time dimension - each consumer has a constant heat demand. The time-shift approach from the dynamic scenario cannot be applied. Instead, we introduce **partial load ratios** to capture the effect of simultaneity.

The key insight: In reality, not all consumers operate at full capacity at the same moment. A partial load ratio represents the fraction of maximum demand that a consumer contributes to the steady-state calculation.

---

## 2. Problem Formulation

### 2.1 Decision Variables

For each consumer `i`, define a partial load ratio:

```
r_i ∈ (0, 1]
```

The effective steady-state demand of consumer `i` becomes:

```
Q_i = r_i * Q_max_i
```

### 2.2 Simultaneity at an Edge

For an edge `e` with `n_e` downstream buildings, the achieved simultaneity is:

```
g_achieved(e) = Sum(r_i * Q_max_i) / Sum(Q_max_i)    for all i downstream of e
```

This is the **weighted average** of partial load ratios, weighted by maximum demand.

### 2.3 Target

Match the target simultaneity function `g(n)` as closely as possible at relevant edges:

```
g_achieved(e) ≈ g_target(n_e)
```

---

## 3. Scope of Optimization

### 3.1 Ignore Leaf Edges

Edges directly connecting to a single consumer (leaf edges, `n = 1`) should be **excluded** from the optimization:

- At leaf edges, `g(1) = 1.0` by definition
- This would force `r_i = 1.0` for all consumers, making simultaneity impossible at upstream edges
- Leaf edges represent local connections where simultaneity is not physically meaningful

### 3.2 Focus on Main/Source Edges

The optimization should prioritize edges closer to the source:

- These edges serve many consumers (`n` is large)
- Simultaneity has the greatest impact on network sizing at these locations
- Deviation from target simultaneity is more acceptable at smaller branch edges

**Suggested weighting:** Edges with more downstream buildings receive higher weight in the objective function.

---

## 4. Optimization Problem

### 4.1 Objective Function

Minimize weighted deviation from target simultaneity:

```
minimize F(r_1, ..., r_N) = Sum over edges e (n_e > 1): w_e * (g_achieved(e) - g_target(n_e))^2
```

where `w_e` is a weight factor, e.g.:
- `w_e = n_e` (proportional to downstream buildings)
- `w_e = n_e^2` (stronger emphasis on main edges)
- `w_e = 1` if `n_e > threshold`, else `0` (only consider main edges)

### 4.2 Constraints

```
0 < r_i <= 1    for all consumers i
```

Optional additional constraints:
- `r_i >= r_min` (minimum load ratio, e.g., 0.3)
- Building-type-specific bounds

---

## 5. Non-Uniqueness of Solutions

### 5.1 Why Many Solutions Exist

The system is typically **underdetermined**:

| Network Topology | Non-leaf Edges | Variables (N consumers) | Result |
|------------------|----------------|-------------------------|--------|
| Star (all branch from one point) | 1 | N | Highly underdetermined |
| Binary tree | ~N/2 | N | Underdetermined |
| Linear chain | N-1 | N | Nearly unique |

Most real networks have branching structures, leading to **infinitely many valid solutions**.

### 5.2 Selecting Among Solutions

To obtain a unique, meaningful solution, add a **regularization term** to the objective:

**Option A: Minimize variance (prefer uniform distribution)**
```
F_reg = lambda * Sum((r_i - r_mean)^2)
```

**Option B: Maximize minimum load (avoid extreme values)**
```
F_reg = -lambda * min(r_i)
```

**Option C: Prefer loads proportional to building size**
```
F_reg = lambda * Sum((r_i - r_target_i)^2)
```
where `r_target_i` could be based on building type or size.

The regularization parameter `lambda` controls the trade-off between matching simultaneity and achieving a "nice" distribution of partial loads.

---

## 6. Algorithm Recommendations

Given the problem structure (quadratic objective, box constraints, underdetermined), suitable algorithms include:

1. **Quadratic Programming (QP)** - If formulated as weighted least squares with regularization
2. **Gradient Descent with Projection** - Simple, handles box constraints
3. **Sequential Least Squares** - Solve for simultaneity match, then adjust for regularization

### Suggested Approach

1. Formulate as regularized least squares:
   ```
   minimize ||A*r - b||^2 + lambda * ||r - r_prior||^2
   subject to: 0 < r <= 1
   ```
   where:
   - `A` encodes the weighted contribution of each consumer to each edge's simultaneity
   - `b` contains the target simultaneity values
   - `r_prior` is the preferred distribution (e.g., all ones, or uniform)

2. Solve using standard QP solver or iterative projection method

---

## 7. Implementation Outline

```
Input:
- Network topology (edges, downstream building counts)
- Consumer maximum demands Q_max_i
- Target simultaneity function g(n)
- Weighting scheme for edges
- Regularization parameter lambda

Algorithm:
1. Identify non-leaf edges (n_e > 1)
2. Build matrix A: A[e,i] = w_e * Q_max_i / Sum(Q_max_j) for j downstream of e
3. Build vector b: b[e] = w_e * g_target(n_e)
4. Solve regularized least squares with box constraints
5. Return partial load ratios r_i

Output:
- Partial load ratios for each consumer
- Achieved simultaneity at each edge
- Deviation report (target vs. achieved)
```

---

## 8. Deliverables

1. Implementation of the steady-state partial load optimization
2. Integration with existing network data structures
3. Comparison of achieved vs. target simultaneity across edges
4. Sensitivity analysis: effect of regularization parameter on solution quality
5. Documentation of chosen weighting scheme and rationale

---

## 9. Validation

- Verify that main edges (high `n_e`) achieve simultaneity close to target
- Check that partial load ratios are physically plausible (not too extreme)
- Compare total network demand with and without simultaneity correction
- Test on networks of varying topology (star, tree, mixed)
