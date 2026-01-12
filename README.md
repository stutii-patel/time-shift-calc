# Time Shift Optimization Logic

This project implements an algorithm to optimize the **Simultaneity Factor** in a district heating network by applying individual **Time Shifts** to consumer heat demand profiles.

## Usage

```bash
python3 time_shift_calculations.py [path/to/vicus_file.vicus]
```
If no file is provided, it runs on a synthetic network.

---

## Code Structure (`time_shift_calculations.py`)

The script is a standalone Python implementation requiring no heavy external libraries (only `networkx` for graph parsing).

### 1. Helper Functions
Replacements for NumPy operations to keep dependencies minimal.
- `vec_sum`, `vec_roll`: Perform element-wise addition and cyclic shifting of list arrays (profiles).

### 2. Class `LinearSpline`
Implements linear interpolation logic to match VICUS C++ `LinearSpline::value()`.
- **Purpose**: Calculates the **Target Simultaneity** based on the number of consumers ($n$).
- **Logic**: Reads $X$ (count) and $Y$ (factor) values from the VICUS XML and interpolates for any given $n$.

### 3. Class `Consumer`
Represents a building with a heat demand.
- **`peak_load`**: The maximum power demand.
- **`time_shift`**: The variable we are optimizing (in hours).
- **`get_shifted_profile()`**: Returns the base 24h profile shifted cyclically by the `time_shift` amount.

### 4. Class `Edge`
Represents a pipe segment in the network.
- **`downstream_consumers`**: List of all consumers supplied by this edge.
- **`calculate_current_simultaneity()`**: Calculates the **Actual** simultaneity factor:
  $$ g_{actual} = \frac{\max(\sum P_{shifted})}{\sum (\max P_{individual})} $$
- **`get_target_simultaneity()`**: Queries the `LinearSpline` to find the **Target** factor for the number of downstream consumers.

### 5. Class `Network`
Container for Consumers and Edges. Maps the topology so the optimizer knows which edges are affected when a specific consumer is shifted.

---

## Optimization Logic: "Fast Hill Climbing"

The function `run_hill_climbing_optimization` performs the magic.

### How Time Shift is Calculated
1.  **Initialization**: All consumers start with `shift = 0`.
2.  **Iteration**: The algorithm loops through every consumer randomly.
3.  **Trial & Error**: For each consumer, it tentatively tries moving the shift by specific steps (e.g., +1h, -1h, +3h, etc.).
4.  **Evaluation**: It recalculates the **Error** (see below) for the local edges affected by this consumer.
5.  **Selection**: If a shift reduces the error, it is **accepted** and applied immediately ("Hill Climbing").
6.  **Convergence**: This repeats until no more improvements can be found or `max_iterations` is reached.

### Terminology in Logs

#### "Target" vs "Actual"
-   **Target**: The goal simultaneity factor for an edge.
    -   *Example*: A pipe serving 15 buildings might have a target $g = 0.9485$ (meaning the peak flow should be ~95% of the total connected capacity).
    -   *Source*: Defined by the `Simultaneity` spline in the VICUS XML.
-   **Actual**: The current simultaneity factor calculated from the specific time-shifted profiles.
    -   *Goal*: We want **Actual** to be as close to **Target** as possible (but not necessarily strictly lower, just matching).

#### "Fast Hill Climbing for 15 consumers"
This indicates the scale of the problem.
-   **15 consumers**: The algorithm found 15 optimizable nodes (SubStations) in the loaded graph.

#### "Iterations"
An **Iteration** is one full pass through all consumers.
-   *Iteration 1*: The algorithm looked at every consumer once and tried to optimize them.
-   *Updates*: How many times it successfully found a better shift during that pass.

#### "Error"
The **Error** denotes the **Sum of Squared Differences** between the Actual and Target simultaneity across all edges.

$$ Error = \sum_{edges} (g_{actual} - g_{target})^2 $$

-   **High Error (e.g., 0.088)**: The Actual curves are far from the Target (usually everyone is peaking at the same time, so $g_{actual} = 1.0$).
-   **Low Error (e.g., 0.002)**: The Actual curves closely match the Targets.

---

## Example Output Explained

```text
75->24: Cons=15 Target=0.9485, Actual=1.0000
```
> **Start**: The edge `75->24` serves 15 consumers. It SHOULD have a factor of **0.9485**, but currently everyone peaks together, so it is **1.0000**.

```text
Iteration 1: Error = 0.005508
```
> **Progress**: After moving some consumers' start times, the total disagreement (Error) dropped massively.

```text
75->24: Cons=15 Target=0.9485, Actual=0.9645
```
> **End**: The edge is now at **0.9645**, which is very close to the target. The peak load has been reduced.
