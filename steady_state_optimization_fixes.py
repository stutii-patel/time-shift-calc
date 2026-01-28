"""
Recommended fixes for steady-state optimization issues
This file contains code snippets to improve accuracy at n_consumers ≈ 50
"""

# ============================================================================
# FIX 1: Improved weighting scheme to balance edge sizes
# ============================================================================

def calculate_adaptive_weight_v1(n_consumers):
    """
    Logarithmic weighting: reduces dominance of very large edges
    while still giving them more importance than small edges
    """
    import math
    return math.log(n_consumers + 1) ** 2


def calculate_adaptive_weight_v2(n_consumers):
    """
    Capped quadratic: keeps quadratic scaling but caps at a maximum
    """
    return min(n_consumers ** 2, 10000)


def calculate_adaptive_weight_v3(n_consumers):
    """
    Tiered weighting: different scaling for different size ranges
    Gives extra boost to medium-sized edges (50-200 consumers)
    """
    if n_consumers <= 50:
        return n_consumers ** 2 * 2.0  # 2x boost for small edges
    elif n_consumers <= 200:
        return n_consumers ** 2 * 1.5  # 1.5x boost for medium edges
    elif n_consumers <= 500:
        return n_consumers ** 2 * 1.0  # Normal weight
    else:
        return n_consumers ** 2 * 0.5  # Reduce for very large edges


def calculate_adaptive_weight_v4(n_consumers):
    """
    Square root scaling: gentler than quadratic
    """
    import math
    return (math.sqrt(n_consumers) * 10) ** 2  # Scale factor to normalize


# ============================================================================
# FIX 2: Modified optimization loop with improved weighting
# ============================================================================

def optimize_steady_state_with_improved_weighting(
    network, relevant_edges, iterations=2000, 
    learning_rate=0.01, lambda_reg=0.1, weight_scheme='v3'
):
    """
    Modified optimization function with improved edge weighting
    
    Parameters:
    -----------
    weight_scheme : str
        'v1' = logarithmic, 'v2' = capped, 'v3' = tiered, 'v4' = sqrt
    """
    
    # Weight calculation functions
    weight_functions = {
        'v1': calculate_adaptive_weight_v1,
        'v2': calculate_adaptive_weight_v2,
        'v3': calculate_adaptive_weight_v3,
        'v4': calculate_adaptive_weight_v4,
    }
    
    calc_weight = weight_functions.get(weight_scheme, calculate_adaptive_weight_v3)
    
    # Initialize all consumers to r=1
    for c in network.consumers:
        c.partial_load_ratio = 1.0
        
    # Calculate initial error
    initial_error = 0
    for edge in relevant_edges:
        if edge.n_consumers <= 1:
            continue
        g_act = edge.calculate_steady_state_simultaneity()
        g_tgt = edge.get_target_simultaneity()
        diff = g_act - g_tgt
        w_e = calc_weight(edge.n_consumers)  # Use adaptive weight
        initial_error += w_e * diff**2
    
    print(f"Initial weighted error (using {weight_scheme}): {initial_error:.6f}")
    
    # Calculate average target for regularization
    target_sims = [e.get_target_simultaneity() for e in relevant_edges if e.n_consumers > 1]
    avg_target_sim = sum(target_sims) / len(target_sims) if target_sims else 0.7
    print(f"Average target simultaneity: {avg_target_sim:.4f}")
    
    best_obj = initial_error
    best_ratios = [c.partial_load_ratio for c in network.consumers]
    no_improvement_count = 0
    
    for it in range(iterations):
        gradients = {c.id: 0.0 for c in network.consumers}
        current_error = 0
        
        # 1. Calculate gradients with adaptive weighting
        edges_processed = 0
        for edge in relevant_edges:
            if edge.n_consumers <= 1:
                continue
            edges_processed += 1
            
            g_act = edge.calculate_steady_state_simultaneity()
            g_tgt = edge.get_target_simultaneity()
            diff = g_act - g_tgt
            
            # USE ADAPTIVE WEIGHT HERE
            w_e = calc_weight(edge.n_consumers)
            
            current_error += w_e * diff**2
            
            denominator = sum(c.peak_load for c in edge.downstream_consumers)
            if denominator == 0:
                continue
            
            factor = 2 * w_e * diff / denominator
            
            for c in edge.downstream_consumers:
                gradients[c.id] += factor * c.peak_load
        
        # 2. Add regularization
        reg_error = 0
        for c in network.consumers:
            deviation = c.partial_load_ratio - avg_target_sim
            regularization = 2 * lambda_reg * deviation
            gradients[c.id] += regularization
            reg_error += lambda_reg * deviation**2
        
        total_obj = current_error + reg_error
        
        # 3. Adaptive learning rate
        grad_values = [abs(g) for g in gradients.values()]
        if grad_values:
            rms_grad = (sum(g**2 for g in grad_values) / len(grad_values)) ** 0.5
            if rms_grad > 1e-10:
                grad_scale = 1.0 / rms_grad
                adaptive_lr = learning_rate * grad_scale
            else:
                adaptive_lr = learning_rate
        else:
            adaptive_lr = learning_rate
        
        if it % 200 == 0:
            print(f"  Iter {it}: Obj={total_obj:.6f} (Fit: {current_error:.6f}, Reg: {reg_error:.6f}) | LR: {adaptive_lr:.6f}")
        
        # 4. Update ratios with constraints
        max_change = 0
        for c in network.consumers:
            old_r = c.partial_load_ratio
            new_r = old_r - adaptive_lr * gradients[c.id]
            new_r = max(0.05, min(1.0, new_r))  # Constrain to [0.05, 1.0]
            c.partial_load_ratio = new_r
            max_change = max(max_change, abs(new_r - old_r))
        
        # 5. Track best solution
        if total_obj < best_obj:
            best_obj = total_obj
            best_ratios = [c.partial_load_ratio for c in network.consumers]
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Early stopping
        if no_improvement_count > 100:
            print(f"  Early stopping at iteration {it}")
            break
        
        if max_change < 1e-6:
            print(f"  Converged at iteration {it}")
            break
    
    # Apply best solution
    for i, c in enumerate(network.consumers):
        c.partial_load_ratio = best_ratios[i]
    
    print(f"Final Error: {best_obj:.6f}")
    return total_obj


# ============================================================================
# FIX 3: Improved results printing (exclude leaf edges)
# ============================================================================

def print_optimization_results_improved(relevant_edges):
    """
    Print results with clear separation of optimized vs. non-optimized edges
    """
    print(f"\n{'='*80}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*80}\n")
    
    # Separate edges by type
    optimized_edges = [e for e in relevant_edges if e.n_consumers > 1]
    leaf_edges = [e for e in relevant_edges if e.n_consumers == 1]
    
    # Print optimized edges
    print(f"OPTIMIZED EDGES (n_consumers > 1): {len(optimized_edges)} edges")
    print(f"{'-'*80}")
    print(f"{'Edge ID':<20} | {'n_e':<5} | {'Target g(n)':<12} | {'Achieved g':<12} | {'Deviation':<10}")
    print(f"{'-'*80}")
    
    sorted_edges = sorted(optimized_edges, key=lambda e: e.n_consumers, reverse=True)
    
    deviations = []
    for edge in sorted_edges:
        g_act = edge.calculate_steady_state_simultaneity()
        g_tgt = edge.get_target_simultaneity()
        dev = g_act - g_tgt
        deviations.append(abs(dev))
        
        print(f"{edge.id:<20} | {edge.n_consumers:<5} | {g_tgt:<12.4f} | {g_act:<12.4f} | {dev:<10.4f}")
    
    # Statistics
    mean_abs_dev = sum(deviations) / len(deviations) if deviations else 0
    max_abs_dev = max(deviations) if deviations else 0
    
    print(f"{'-'*80}")
    print(f"Mean |Deviation|: {mean_abs_dev:.4f}")
    print(f"Max  |Deviation|: {max_abs_dev:.4f}")
    print(f"{'='*80}\n")
    
    # Print leaf edges summary
    if leaf_edges:
        print(f"LEAF EDGES (n_consumers = 1): {len(leaf_edges)} edges")
        print(f"{'-'*80}")
        print("These edges were NOT optimized. By definition:")
        print("  - Target simultaneity g(n) = 1.0")
        print("  - Achieved simultaneity = 1.0")
        print("  - Deviation = 0.0")
        print(f"{'='*80}\n")


# ============================================================================
# FIX 4: Diagnostic function to analyze deviation patterns
# ============================================================================

def analyze_deviation_by_edge_size(relevant_edges):
    """
    Create bins of edge sizes and analyze deviation patterns
    """
    import statistics
    
    # Create bins
    bins = [
        (1, 1, "Single consumer (not optimized)"),
        (2, 10, "Very small (2-10)"),
        (11, 50, "Small (11-50)"),
        (51, 100, "Medium (51-100)"),
        (101, 200, "Large (101-200)"),
        (201, 500, "Very large (201-500)"),
        (501, float('inf'), "Huge (500+)")
    ]
    
    print(f"\n{'='*80}")
    print("DEVIATION ANALYSIS BY EDGE SIZE")
    print(f"{'='*80}\n")
    
    for min_n, max_n, label in bins:
        edges_in_bin = [
            e for e in relevant_edges 
            if min_n <= e.n_consumers <= max_n
        ]
        
        if not edges_in_bin:
            continue
        
        deviations = []
        for edge in edges_in_bin:
            g_act = edge.calculate_steady_state_simultaneity()
            g_tgt = edge.get_target_simultaneity()
            dev = g_act - g_tgt
            deviations.append(dev)
        
        abs_deviations = [abs(d) for d in deviations]
        
        print(f"{label}:")
        print(f"  Count: {len(edges_in_bin)}")
        print(f"  Mean deviation: {statistics.mean(deviations):.4f}")
        print(f"  Mean |deviation|: {statistics.mean(abs_deviations):.4f}")
        print(f"  Max |deviation|: {max(abs_deviations):.4f}")
        print(f"  Std dev: {statistics.stdev(deviations) if len(deviations) > 1 else 0:.4f}")
        print()
    
    print(f"{'='*80}\n")


# ============================================================================
# FIX 5: Enhanced visualization with annotations
# ============================================================================

def plot_enhanced_verification(relevant_edges, output_path):
    """
    Enhanced plotting with better annotations and separate handling of leaf edges
    """
    import matplotlib.pyplot as plt
    import os
    
    # Filter edges
    opt_edges = [e for e in relevant_edges if e.n_consumers > 1]
    leaf_edges = [e for e in relevant_edges if e.n_consumers == 1]
    
    if not opt_edges:
        print("No edges to plot.")
        return
    
    # Collect data
    n_consumers = []
    target_sim = []
    achieved_sim = []
    edge_weights = []
    
    for edge in opt_edges:
        n_consumers.append(edge.n_consumers)
        target_sim.append(edge.get_target_simultaneity())
        achieved_sim.append(edge.calculate_steady_state_simultaneity())
        edge_weights.append(edge.n_consumers ** 2)
    # Determine network size and set appropriate problem range
    max_n = max(n_consumers)
    min_n = min(n_consumers)
    
    if max_n > 200:
        problem_start, problem_end = 40, 60
        problem_label = 'Problem Range (n≈50)'
    elif max_n > 50:
        problem_start, problem_end = 5, 15
        problem_label = 'Problem Range (n=5-15)'
    else:
        problem_start, problem_end = 5, 10
        problem_label = 'Problem Range (n=5-10)'

    
    print(f"Network size range: {min_n} to {max_n} consumers")
    print(f"Using problem range: {problem_start}-{problem_end}")
    
    # Normalize weights
    max_weight = max(edge_weights) if edge_weights else 1
    marker_sizes = [50 + 200 * (w / max_weight) for w in edge_weights]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot 1: Target vs Achieved
    scatter1 = ax1.scatter(
        target_sim, achieved_sim, s=marker_sizes, alpha=0.6,
        c=n_consumers, cmap='viridis', edgecolors='black', linewidth=1
    )
    
    # Perfect match line
    min_val = min(min(target_sim), min(achieved_sim))
    max_val = max(max(target_sim), max(achieved_sim))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
             label='Perfect Match', alpha=0.7)
    
    ax1.set_xlabel('Target Simultaneity g(n)', fontsize=12)
    ax1.set_ylabel('Achieved Simultaneity', fontsize=12)
    ax1.set_title('Steady-State Optimization: Target vs Achieved\\n(Marker size = edge weight, n=1 excluded)',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Colorbar
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Number of Consumers (n_e)', fontsize=10)
    
    # Plot 2: Deviation by size with problem areas highlighted
    deviations = [a - t for a, t in zip(achieved_sim, target_sim)]
    
    scatter2 = ax2.scatter(
        n_consumers, deviations, s=marker_sizes, alpha=0.6,
        c=n_consumers, cmap='viridis', edgecolors='black', linewidth=1
    )
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Deviation', alpha=0.7)
    
    # Highlight problem area (n ≈ 50)
    ax2.axvspan(problem_start, problem_end, alpha=0.15, color='red', 
            label=problem_label)
    ax2.set_xlabel('Number of Downstream Consumers (n_e)', fontsize=12)
    ax2.set_ylabel('Deviation (Achieved - Target)', fontsize=12)
    ax2.set_title('Deviation from Target by Edge Size\\n(Positive = above target, n=1 excluded)',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Statistics
    mean_abs_dev = sum(abs(d) for d in deviations) / len(deviations)
    max_abs_dev = max(abs(d) for d in deviations)
    stats_text = (
        f'Mean |Deviation|: {mean_abs_dev:.4f}\\n'
        f'Max |Deviation|: {max_abs_dev:.4f}\\n'
        f'Leaf edges (n=1): {len(leaf_edges)} (not shown)'
    )
    ax2.text(
        0.02, 0.98, stats_text, transform=ax2.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    )
    
    plt.tight_layout()
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved enhanced verification plot to {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close()


# ============================================================================
# USAGE MAIN
# ============================================================================

def main():
    import sys
    import os
    import time
    # Import the loader from the main script
    # Ensure the script can find time_shift_calculations.py in the current directory
    sys.path.append(os.getcwd())
    try:
        from time_shift_calculations import load_network_from_vicus
    except ImportError:
        print("Error: Could not import 'load_network_from_vicus' from 'time_shift_calculations.py'.")
        print("Make sure both files are in the same directory.")
        return

    start_time = time.time()
    
    if len(sys.argv) < 2:
        print("Usage: python3 optimization_fixes.py <vicus_file> [output_dir] [weight_scheme]")
        print("Weight schemes: v1 (log), v2 (capped), v3 (tiered), v4 (sqrt)")
        return
        
    filepath = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output_improved"
    weight_scheme = sys.argv[3] if len(sys.argv) > 3 else "v3"
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    case_name = os.path.splitext(os.path.basename(filepath))[0]
    print(f"Loading network for case '{case_name}' from {filepath}...")
    
    try:
        net, relevant_edges = load_network_from_vicus(filepath)
    except Exception as e:
        print(f"Failed to load network: {e}")
        return
    
    if not relevant_edges:
        print("No relevant edges found.")
        return

    # Create output directory unique to this case
    case_output_dir = os.path.join(output_dir, case_name, f"steady_state_{weight_scheme}")
    os.makedirs(case_output_dir, exist_ok=True)
    
    print(f"\n--- Starting Optimization using Scheme '{weight_scheme}' for {case_name} ---")
    
    # Run optimization
    # We use a slightly lower learning rate for stability with these new weights
    final_error = optimize_steady_state_with_improved_weighting(
        net, relevant_edges, 
        iterations=2000, 
        learning_rate=0.005, 
        lambda_reg=0.05,
        weight_scheme=weight_scheme
    )
    
    # Print results
    print_optimization_results_improved(relevant_edges)
    
    # Analyze deviations
    analyze_deviation_by_edge_size(relevant_edges)
    
    # Generate plot
    plot_filename = f"verification_{case_name}_{weight_scheme}.png"
    plot_path = os.path.join(case_output_dir, plot_filename)
    plot_enhanced_verification(relevant_edges, plot_path)
    
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")
    print(f"Results saved to: {case_output_dir}")

if __name__ == "__main__":
    main()
