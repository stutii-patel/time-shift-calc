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


## Outline 
### The Core Objective: Load Balancing

The goal of this script is to adjust the timing of heat consumption for various buildings (consumers) so that the total load on the network pipes (edges) matches a theoretical ideal called the Target Simultaneity Factor.

- The Problem: If all consumers use heat at the exact same time, the pipes must be massive and expensive.

- The Solution: By slightly shifting when a consumer uses heat (e.g., 30 minutes earlier or later), we can "smooth out" the aggregate peak, making the network more efficient.

### Key Data Models

To understand the logic, we must look at the two primary objects in the code:

  - The Consumer: Represented by a 24-hour load profile and a peak_load. Each consumer has a time_shift variable, which acts as a "tuning knob" for the algorithm.

  - The Edge (Pipe): An edge knows which consumers are downstream of it. It maintains an Aggregated Profile, which is the sum of all its downstream consumers' shifted profiles.

### The Math: Target vs. Actual Simultaneity

The algorithm lives or dies by its "Error Function." It tries to minimize the difference between two values at every pipe in the network:

  - Actual Simultaneity (gact): The highest peak of the combined profiles divided by the sum of individual peak loads.

  - Target Simultaneity (gtgt): Calculated using the Winter et al. (2001) formula:
    g(n)=a+1+(cn)db

  - Where n is the number of consumers. This formula tells us how much "smoothing" we should theoretically see as more consumers are added to a pipe.

### The Algorithm: Multi-Start Adaptive Hill Climbing

This is a three-layered optimization strategy designed to find the "Global Minimum" (the best possible set of shifts).

Layer 1: Multi-Start (Avoiding Local Minima) Optimization often gets stuck in "local traps"—solutions that look good but aren't the best. To solve this, the script runs the entire process multiple times (default: 5 restarts).

  - Restart 1: Starts with all shifts at zero.

  - Restarts 2-5: Starts with every consumer assigned a random shift within a ±3-hour window.

Layer 2: Adaptive Phases (Coarse to Fine) The algorithm doesn't just guess randomly. It searches in three distinct phases of increasing precision:

  - Phase 1 (Coarse): Tries moving consumers by large jumps (±3 to ±4 hours) to find the right "neighborhood".

  - Phase 2 (Medium): Tries moderate jumps (±1.5 to ±2 hours).

  - Phase 3 (Fine): Tries tiny adjustments (±15 to ±30 minutes) to perfectly align the peaks.

Layer 3: Coordinate Descent (One-by-One Optimization) Instead of trying to change every consumer at once (which is computationally impossible), it uses Coordinate Descent:

  -  It picks one consumer at a time (in a random order).

  -  It tests various shifts for only that consumer.

  -  It calculates the error only for the pipes downstream of that specific consumer.

  - If a shift reduces the error, it keeps it. If not, it reverts.

### Efficiency: The update_profile Trick

Re-calculating the entire network load every time a consumer moves would be incredibly slow. The script uses a "delta update" logic:

- When a consumer shifts, the Edge simply subtracts that consumer's old profile and adds their new shifted profile to the current aggregate.

- This makes the math extremely fast, allowing the algorithm to handle hundreds of consumers in seconds.

### Summary of Execution Flow

Load Data: Parse the network structure and load the 24-hour heat profiles.

Loop Restarts: For each restart, randomize the starting positions.

Iterate Phases: Run Coarse → Medium → Fine-tuning.

Save Best: Track which restart produced the lowest total squared error across the network.