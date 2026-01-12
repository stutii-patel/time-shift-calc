# Task: Implementing Load Diversity through Time-Shifted Consumer Profiles

## 1. Background

### 1.1 Thermal Networks and Consumer Demand

In district heating network simulations, each consumer (building) connected to the network has a **heat demand profile** over time. These profiles describe how much thermal power a building requires at each point in time (e.g., hourly values over one year).

### 1.2 Current Implementation

In the current implementation, all consumer heat demand profiles are **identical in shape** - they only differ by a scaling factor that accounts for different building sizes or heat demands. This means:

- All consumers reach their **peak demand at exactly the same moment** in time
- The aggregated demand of multiple consumers is simply the sum of their individual demands
- This represents a **worst-case scenario** that does not reflect reality

### 1.3 The Concept of Simultaneity

In reality, consumers do not demand heat at exactly the same time. One household may heat intensively in the morning, another in the evening. This temporal diversity is captured by the **simultaneity factor** (German: *Gleichzeitigkeitsfaktor*).

The simultaneity factor `g(n)` describes the ratio between the actual aggregated peak demand of `n` consumers and the theoretical maximum (sum of all individual peaks):

```
g(n) = P_aggregated_peak(n) / Sum(P_individual_peak_i)    for i = 1..n
```

Properties of the simultaneity factor:
- For a single consumer: `g(1) = 1.0`
- As `n` increases, `g(n)` typically **decreases**
- Example: `g(100) = 0.5` means that 100 consumers together have a peak demand that is only 50% of the sum of their individual peaks

The current implementation uses a default simultaneity function based on **Winter et al., Euroheat & Power 2001**:

```
g(n) = a + b * exp(-n/c)
```

where `a = 0.4497`, `b = 0.5512`, `c = 53.84`.

---

## 2. Network Topology and Edge-Specific Simultaneity

### 2.1 Tree Structure of Networks

District heating networks typically have a **tree-like topology**:

```
                [Source]
                   |
              [Edge A] ─────── serves 100 buildings downstream
                   |
           ┌──────┴──────┐
      [Edge B]       [Edge C] ─── serves 60 buildings    ─── serves 40 buildings
           |             |
        ┌──┴──┐       ┌──┴──┐
    [E_D] [E_E]   [E_F] [E_G]
      |     |       |     |
     10    50      25    15   (buildings downstream)
```

### 2.2 Edge-Specific Number of Downstream Buildings

Each edge (pipe segment) in the network serves a different number of **downstream buildings**:
- Edges close to the source serve many buildings → lower simultaneity factor
- Edges close to individual buildings serve few buildings → higher simultaneity factor (closer to 1.0)

The simultaneity factor must be applied **per edge** based on how many consumers are connected downstream of that edge.

### 2.3 Current Behavior (Reference: `VICUS_Network.cpp`)

The current code applies simultaneity as a **scaling factor** to the nominal power of each edge:

```cpp
// Line ~1557 in VICUS_Network.cpp
if (m_considerSimultaneity)
    origEdge.m_nominalPower = itQmax->second * m_simultaneity.value(itNumB->second);
```

Here, `itNumB->second` is the number of downstream buildings for that edge.

**Problem:** This approach only scales the peak value but does not actually shift the time series. The temporal profile remains identical for all consumers.

---

## 3. Task Description

### 3.1 Objective

Develop an **optimization algorithm** that calculates individual **time shifts** for each consumer's heat demand profile, such that the resulting aggregated demand at each edge in the network matches the target simultaneity function.

### 3.2 Mathematical Formulation

**Given:**
- `N` consumers, each with a heat demand time series `Q_i(t)` (identical shape, individually scaled)
- A target simultaneity function `g(n)` defined as `m_simultaneity` in `VICUS_Network`
- Network topology with edges, where each edge `e` has a known number of downstream buildings `n_e`

**Decision Variables:**
- Time shifts `dt_1, dt_2, ..., dt_N` for each consumer

**Shifted Profiles:**
```
Q_i_shifted(t) = Q_i(t + dt_i)
```

Note: Time series are cyclic (e.g., one year), so shifts wrap around.

**Aggregated Demand at Edge `e`:**
```
Q_e(t) = Sum of Q_i_shifted(t) for all consumers i downstream of edge e
```

**Simultaneity at Edge `e`:**
```
g_actual(e) = max(Q_e(t)) / Sum(max(Q_i(t))) for all consumers i downstream of edge e
```

**Objective Function:**
Minimize the deviation between actual and target simultaneity across all edges:

```
minimize F(dt_1, ..., dt_N) = Sum over all edges e of: w_e * (g_actual(e) - g_target(n_e))^2
```

where `w_e` could be a weighting factor (e.g., based on edge importance or number of downstream buildings).

### 3.3 Constraints

1. **Time shift bounds:** Each shift should be within reasonable limits
   - For daily patterns: `|dt_i| <= 12 hours`
   - For annual patterns: `|dt_i| <= 6 months` (or as specified)

2. **Cyclic boundary:** Time series wrap around, so `Q(t + T) = Q(t)` where `T` is the period

3. **Reference consumer:** One consumer can be fixed (`dt_1 = 0`) as a reference to remove redundant degrees of freedom

### 3.4 Algorithm Selection

Consider the following optimization approaches:

1. **Gradient-based methods** (e.g., L-BFGS, Gradient Descent)
   - Pro: Fast convergence for smooth problems
   - Con: May get stuck in local minima

2. **Evolutionary/Genetic Algorithms**
   - Pro: Global search, handles non-convex problems
   - Con: Slower convergence, many function evaluations

3. **Simulated Annealing**
   - Pro: Can escape local minima
   - Con: Requires careful parameter tuning

4. **Particle Swarm Optimization**
   - Pro: Good for continuous optimization, parallelizable
   - Con: May converge prematurely

5. **Heuristic Initialization + Local Search**
   - Start with analytically derived shifts (e.g., uniform distribution)
   - Refine with local optimization

---

## 4. Implementation Steps

### Step 1: Understand the Data Structures
- Study `VICUS_Network`, `VICUS_NetworkEdge`, and `VICUS_NetworkNode`
- Understand how downstream buildings are counted per edge
- Locate the simultaneity spline `m_simultaneity`

### Step 2: Implement the Objective Function
- For a given set of time shifts, compute the resulting simultaneity at each edge
- Calculate the total deviation from the target simultaneity function

### Step 3: Implement the Optimization Algorithm
- Choose an appropriate algorithm
- Implement parameter bounds and constraints
- Handle the cyclic nature of time series

### Step 4: Validate Results
- Verify that optimized shifts produce the desired simultaneity behavior
- Visualize aggregated load curves at different network levels
- Compare against theoretical expectations

### Step 5: Integration
- Integrate the algorithm into the existing codebase
- Ensure compatibility with the VICUS data model

---

## 5. Deliverables

1. **Mathematical documentation** of the optimization problem formulation
2. **Implementation** of the optimization algorithm in C++
3. **Validation results** showing that the target simultaneity is achieved
4. **Visualization** of:
   - Original vs. shifted consumer profiles
   - Aggregated demand at selected edges
   - Achieved vs. target simultaneity across the network
5. **Code documentation** following project conventions

---

## 6. References

- `externals/Vicus/src/VICUS_Network.h` - Network class definition
- `externals/Vicus/src/VICUS_Network.cpp` - Implementation with simultaneity handling
- `Network::setDefaultSimultaneity()` - Default simultaneity function (Winter et al., 2001)
- `Network::calculateEdgeNominalHeatingDemands()` - Current edge demand calculation

---

## 7. Hints and Considerations

1. **Start small:** Begin with a simple network (e.g., 10 consumers, linear topology) to verify your approach before scaling up.

2. **Multiple valid solutions:** The problem likely has many valid solutions (different shift combinations can produce the same simultaneity). Document how your algorithm handles this.

3. **Computational efficiency:** For large networks with many consumers, the objective function evaluation may be expensive. Consider:
   - Efficient data structures for downstream building aggregation
   - Parallelization of objective function evaluation
   - Smart initialization to reduce iterations

4. **Physical plausibility:** Ensure that the resulting shifted profiles remain physically meaningful (e.g., heating demand should still correlate with outdoor temperature patterns).

5. **Edge cases:**
   - Single consumer at an edge: `g = 1.0` (no optimization needed)
   - All consumers at one edge: Full network simultaneity applies

---

## 8. Questions to Consider

1. Should the optimization run once during project setup, or dynamically during simulation?
2. How should conflicts be handled when an edge cannot achieve its target simultaneity due to constraints from other edges?
3. Is it acceptable to have different simultaneity values than the target, as long as they are close?
4. Should the algorithm provide feedback about which edges are most constrained?
