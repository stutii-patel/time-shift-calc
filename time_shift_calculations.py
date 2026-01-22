
import math
import random
import time
import sys
import os
import networkx as nx
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# Import the user's graph creator
try:
    from create_graph import create_graph_from_xml
except ImportError:
    print("Warning: create_graph.py not found. Only synthetic mode available.")
    create_graph_from_xml = None

# CONSTANTS
STEPS_PER_HOUR = 4  # 15-minute intervals

# --- Helper Functions ---

def vec_sum(v1, v2):
    return [x + y for x, y in zip(v1, v2)]

def vec_roll(v, shift):
    n = len(v)
    shift = shift % n
    return v[-shift:] + v[:-shift]

def vec_scale(v, scalar):
    return [x * scalar for x in v]

def vec_zeros(n):
    return [0.0] * n

def resample_vector(v, expansion_factor):
    """Linearly interpolate vector v by expansion_factor."""
    if expansion_factor <= 1:
        return v
    
    new_len = len(v) * expansion_factor
    new_v = []
    
    for i in range(new_len):
        # Map new index i to old index (float)
        old_idx = i / expansion_factor
        
        # Get indices for interpolation
        idx0 = int(math.floor(old_idx))
        idx1 = (idx0 + 1) % len(v) # Cyclic boundary for 24h profiles
        
        fraction = old_idx - idx0
        
        val = v[idx0] * (1 - fraction) + v[idx1] * fraction
        new_v.append(val)
        
    return new_v

def plot_peak_day(v, output_path):
    """Plot the extracted peak day profile for verification."""
    plt.figure(figsize=(10, 5))
    hours = [i / STEPS_PER_HOUR for i in range(len(v))]
    plt.plot(hours, v, marker='o', linestyle='-', markersize=2)
    plt.title("Extracted Peak Day Profile (Normalized)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Normalized Load")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 24)
    plt.ylim(0, 1.1)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Peak day profile plot saved to {output_path}")

# --- Linear Spline Class ---

class LinearSpline:
    def __init__(self, x_values, y_values):
        self.x = x_values
        self.y = y_values
        self.slope = []
        
        # Calculate slopes for interpolation
        if len(self.x) > 1:
            for i in range(len(self.x) - 1):
                dx = self.x[i+1] - self.x[i]
                dy = self.y[i+1] - self.y[i]
                self.slope.append(dy / dx if dx != 0 else 0)
        
        self.x_min = self.x[0] if self.x else 0
        self.x_max = self.x[-1] if self.x else 0

    def value(self, x):
        """Linearly interpolate y for a given x."""
        if not self.x: return 1.0 # Default fallback
        
        # Handle single point
        if len(self.x) == 1:
            return self.y[0]

        # Extrapolation (Constant method assumed based on C++ snippet context or simple clamp)
        # The C++ snippet checks extrapolation method. Defaulting to constant/clamping for safety here.
        if x <= self.x_min:
            return self.y[0]
        if x >= self.x_max:
            return self.y[-1]

        # Find interval (Binary search / bisect could differ, implementing linear scan for simplicity/correctness check)
        # C++ uses std::lower_bound.
        import bisect
        it = bisect.bisect_left(self.x, x)
        
        # 'it' is where x could be inserted while maintaining order. 
        # Elements to the left are < x.
        # Index of lower limit i is it - 1
        
        if it == 0: return self.y[0]
        i = it - 1
        if i >= len(self.slope): i = len(self.slope) - 1 # Safety
        
        dx = x - self.x[i]
        val = self.y[i] + self.slope[i] * dx
        return val

# --- Data Structures ---

class Consumer:
    SHIFT_RESOLUTION = 100  # Internal resolution for fractional shifts

    def __init__(self, id, peak_load, base_profile):
        self.id = id
        self.peak_load = peak_load
        self.base_profile = base_profile
        self.time_shift = 0 
        
        # Pre-calculate high-resolution scaled profile for efficient fractional shifts
        # This handles the "one time calculation" requirement.
        scaled_profile = vec_scale(self.base_profile, self.peak_load)
        self.n_base = len(scaled_profile)
        
        # Use linear interpolation to expand the profile
        self.high_res_profile = []
        for i in range(self.n_base * self.SHIFT_RESOLUTION):
            pos = i / self.SHIFT_RESOLUTION
            idx0 = int(math.floor(pos))
            idx1 = (idx0 + 1) % self.n_base
            frac = pos - idx0
            # val = frac×(b−a) + a
            val = scaled_profile[idx0] * (1 - frac) + scaled_profile[idx1] * frac
            self.high_res_profile.append(val)
            
        # Double the profile to handle cyclic shifts with simple slicing
        self.high_res_profile_doubled = self.high_res_profile + self.high_res_profile
        
        self._cached_profile = None
        self._cached_shift = None

    def get_shifted_profile(self):
        # Extremely fast lookup using pre-calculated high-res profile
        if self._cached_profile is None or self._cached_shift != self.time_shift:
            # Calculate the starting index in the high-res profile
            # Python's % handles negative shifts correctly
            total_steps = self.n_base * self.SHIFT_RESOLUTION
            
            # The shift is in 'base' timesteps. Convert to high-res steps.
            # We want to 'pull' values from the PAST for a positive shift (delay)
            # which is equivalent to rolling forward. 
            # In our vec_roll(v, 1) [1,2,3] -> [3,1,2]. 
            # This means index 0 now contains what was at index -1.
            
            # To match vec_roll(v, shift) logic:
            # shifted_v[i] = v[(i - shift) % n]
            
            shift_steps = int(round(self.time_shift * self.SHIFT_RESOLUTION))
            
            start_offset = (total_steps - (shift_steps % total_steps)) % total_steps
            
            # Slice with step = SHIFT_RESOLUTION to get 'n_base' points
            self._cached_profile = self.high_res_profile_doubled[start_offset : start_offset + total_steps : self.SHIFT_RESOLUTION]
            self._cached_shift = self.time_shift
            
        return self._cached_profile

class Edge:
    def __init__(self, id, downstream_consumers, simultaneity_spline=None):
        self.id = id
        self.downstream_consumers = downstream_consumers
        self.n_consumers = len(downstream_consumers)
        self.simultaneity_spline = simultaneity_spline
        
        peaks = [c.peak_load * max(c.base_profile) for c in downstream_consumers]
        self.sum_individual_peaks = sum(peaks) if peaks else 0
        
        self.current_aggregated_profile = vec_zeros(len(downstream_consumers[0].base_profile)) if downstream_consumers else []
        self.initialize_aggregation()

    def initialize_aggregation(self):
        if not self.downstream_consumers: return
        self.current_aggregated_profile = vec_zeros(len(self.current_aggregated_profile))
        for c in self.downstream_consumers:
            p = c.get_shifted_profile()
            self.current_aggregated_profile = vec_sum(self.current_aggregated_profile, p)

    def calculate_current_simultaneity(self):
        if self.n_consumers <= 1 or self.sum_individual_peaks == 0:
            return 1.0
        peak_agg = max(self.current_aggregated_profile)
        return peak_agg / self.sum_individual_peaks

    def update_profile(self, old_consumer_profile, new_consumer_profile):
        if not self.downstream_consumers: return
        self.current_aggregated_profile = [
            a - o + n 
            for a, o, n in zip(self.current_aggregated_profile, old_consumer_profile, new_consumer_profile)
        ]

    def get_target_simultaneity(self):
        if self.n_consumers <= 1:
            return 1.0
        
        # Winter et al., Euroheat & Power 2001
        # Confirmed constants:
        a = 0.449677646267461
        b = 0.551234688
        c = 53.84382392
        d = 1.762743268
        
        # User rule: "if any number of nodes greater than 295 we should take the last value"
        # The C++ snippet generated up to nmax=300 and clamped/stopped.
        # We will use N=300 as the clamp limit (approx 295-300 range where the curve flattens).
        # Actually user said > 295 takes last value (0.475865).
        # The formula at 295 is approx equal to that.
        
        x = min(self.n_consumers, 295) # Clamp input to 295
        
        val = a + (b / (1 + (x/c)**d))
        return min(1.0, val)

class Network:
    def __init__(self):
        self.consumers = []
        self.edges = []
        self.consumer_to_edges = {} 
        self.simultaneity_spline = None
        self.pos = {}
        self.G_nx = None # Store full graph for visualization

    def add_consumer(self, consumer):
        self.consumers.append(consumer)
        self.consumer_to_edges[consumer.id] = []

    def add_edge(self, edge):
        self.edges.append(edge)
        for c in edge.downstream_consumers:
            if c.id not in self.consumer_to_edges:
                 self.consumer_to_edges[c.id] = []
            self.consumer_to_edges[c.id].append(edge)

# --- Optimization Logic ---

def calculate_local_error(edges):
    total_error = 0
    for edge in edges:
        g_act = edge.calculate_current_simultaneity()
        g_tgt = edge.get_target_simultaneity()
        total_error += (g_act - g_tgt)**2
    return total_error


def multi_start_adaptive_hill_climbing(network, relevant_edges, num_restarts=5, max_iterations_per_phase=20):
    """
    Multi-start hill climbing with adaptive step sizes and coordinate descent.
    """
    print(f"\n{'='*80}")
    print(f"MULTI-START ADAPTIVE HILL CLIMBING")
    print(f"{'='*80}")
    print(f"Consumers: {len(network.consumers)}")
    print(f"Edges: {len(relevant_edges)}")
    print(f"Restarts: {num_restarts}")
    print(f"{'='*80}\n")
    
    profile_length = len(network.consumers[0].base_profile)
    max_shift_timesteps = 3 * STEPS_PER_HOUR  # Limit shifts to ±3 hours (realistic range)
    
    print(f"Profile length: {profile_length} timesteps ({profile_length/STEPS_PER_HOUR}h)")
    print(f"Max shift: ±{max_shift_timesteps} timesteps (±{max_shift_timesteps/STEPS_PER_HOUR}h)\n")
    
    best_global_error = float('inf')
    best_global_shifts = None
    
    for restart in range(num_restarts):
        print(f"\n{'─'*80}")
        print(f"RESTART {restart + 1}/{num_restarts}")
        print(f"{'─'*80}")
        
        # Random initialization (except first restart)
        if restart == 0:
            print("Initialization: Zero shifts (baseline)")
            for consumer in network.consumers:
                consumer.time_shift = 0
        else:
            print(f"Initialization: Random shifts in range ±{max_shift_timesteps/STEPS_PER_HOUR}h")
            for consumer in network.consumers:
                # Spread shifts across entire period
                consumer.time_shift = random.randint(-max_shift_timesteps, max_shift_timesteps)
        
        # Reinitialize all edges
        for edge in relevant_edges:
            edge.initialize_aggregation()
        
        initial_error = calculate_local_error(relevant_edges)
        print(f"Initial error: {initial_error:.6f}")
        
        # Phase 1: Coarse search (large steps)
        print(f"\n  Phase 1: Coarse search (±{profile_length//8} timesteps)")
        shifts_coarse = [
            max_shift_timesteps * 0.75, # 75% of max
            -max_shift_timesteps * 0.75,
            max_shift_timesteps * 0.5, # 50% of max
            -max_shift_timesteps * 0.5,
        ]
        run_coordinate_descent_phase(network, relevant_edges, shifts_coarse, max_shift_timesteps, max_iterations_per_phase)
        
        # Phase 2: Medium search
        print(f"\n  Phase 2: Medium search (±{profile_length//16} timesteps)")
        shifts_medium = [
            max_shift_timesteps * 0.4, # 40% of max
            -max_shift_timesteps * 0.4,
            max_shift_timesteps * 0.25, # 25% of max
            -max_shift_timesteps * 0.25,
        ]
        run_coordinate_descent_phase(network, relevant_edges, shifts_medium, max_shift_timesteps, max_iterations_per_phase)
        
        # Phase 3: Fine search
        print(f"\n  Phase 3: Fine-tuning (±1.0 timesteps = ±0.25h)")
        shifts_fine = [
            1.0,                   # ±15 min (0.25h)
            -1.0,
            2.0,                   # ±30 min (0.50h)
            -2.0,
        ]
        run_coordinate_descent_phase(network, relevant_edges, shifts_fine, max_shift_timesteps, max_iterations_per_phase)
        
        final_error = calculate_local_error(relevant_edges)
        print(f"\n  Final error: {final_error:.6f} (improvement: {initial_error - final_error:.6f})")
        
        if final_error < best_global_error:
            best_global_error = final_error
            best_global_shifts = [c.time_shift for c in network.consumers]
            print(f"  ★ NEW BEST SOLUTION ★")
    
    # Apply best solution
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Best error: {best_global_error:.6f}")
    print(f"Applying best solution...")
    
    for i, consumer in enumerate(network.consumers):
        consumer.time_shift = best_global_shifts[i]
    
    for edge in relevant_edges:
        edge.initialize_aggregation()
    
    return best_global_error

def run_coordinate_descent_phase(network, relevant_edges, shift_increments, max_shift, max_iterations):
    """
    Coordinate descent: optimize one consumer at a time.
    """
    for iteration in range(max_iterations):
        improved = False
        
        # Shuffle to avoid bias
        indices = list(range(len(network.consumers)))
        random.shuffle(indices)
        
        for i in indices:
            consumer = network.consumers[i]
            original_shift = consumer.time_shift
            original_profile = consumer.get_shifted_profile()
            
            best_local_shift = original_shift
            best_local_error = float('inf')
            
            affected_edges = network.consumer_to_edges[consumer.id]
            base_error = calculate_local_error(affected_edges)
            
            # Try all shift increments
            for step in shift_increments:
                new_shift = original_shift + step
                
                # Enforce bounds
                if abs(new_shift) > max_shift:
                    continue
                
                # Temporarily apply shift
                consumer.time_shift = new_shift
                new_profile = consumer.get_shifted_profile()
                
                # Calculate hypothetical error without modifying edges
                hypothetical_error = 0
                for edge in affected_edges:

                    # current total load, old time shift and new time shift
                    hypothetical_agg = [
                        c - o + n 
                        for c, o, n in zip(edge.current_aggregated_profile, original_profile, new_profile)
                    ]
                    
                    peak_agg = max(hypothetical_agg)
                    if edge.sum_individual_peaks > 0:
                        sim = peak_agg / edge.sum_individual_peaks
                    else:
                        sim = 1.0
                    
                    target = edge.get_target_simultaneity()
                    hypothetical_error += (sim - target)**2
                
                if hypothetical_error < best_local_error:
                    best_local_error = hypothetical_error
                    best_local_shift = new_shift
            
            # Revert to original
            consumer.time_shift = original_shift
            consumer._cached_shift = original_shift
            consumer._cached_profile = original_profile
            
            # Apply best move if improvement found
            if best_local_error < base_error - 1e-9:
                consumer.time_shift = best_local_shift
                new_profile = consumer.get_shifted_profile()
                for edge in affected_edges:
                    edge.update_profile(original_profile, new_profile)
                improved = True
        
        current_error = calculate_local_error(relevant_edges)
        print(f"    Iteration {iteration+1}: Error = {current_error:.6f}")
        
        if not improved:
            print(f"    Converged (no improvement)")
            break
def visualize_network(net, relevant_edges, output_path, title="Network Simultaneity"):
    print(f"Generating visualization: {output_path}")
    
    # Decide which graph to plot: Full graph if available, else build from relevant edges
    if net.G_nx:
        G_vis = net.G_nx
    else:
        G_vis = nx.DiGraph()
        for edge in relevant_edges:
            try:
                if "->" in edge.id:
                    u, v = edge.id.split("->")
                    G_vis.add_edge(u, v)
            except ValueError:
                pass
    
    if len(G_vis.nodes()) == 0:
        print("No edges to visualize.")
        return

    # Map relevant edges to their simultaneity
    edge_obj_map = {e.id: e for e in relevant_edges}
    
    # Collect simultaneity values for color mapping
    sim_values = []
    for u, v in G_vis.edges():
        eid = f"{u}->{v}"
        if eid in edge_obj_map:
            sim = edge_obj_map[eid].calculate_current_simultaneity()
            sim_values.append(sim)
            
    vmin = min(sim_values) if sim_values else 0.9
    vmax = max(sim_values) if sim_values else 1.0
    # Ensure we have a reasonable range
    if vmax - vmin < 0.01:
        vmin = vmin - 0.05
        vmax = vmax + 0.05
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.coolwarm
    
    # Prepare edge colors and widths
    full_edge_colors = []
    full_edge_widths = []
    for u, v in G_vis.edges():
        eid = f"{u}->{v}"
        if eid in edge_obj_map:
            sim = edge_obj_map[eid].calculate_current_simultaneity()
            full_edge_colors.append(cmap(norm(sim)))
            full_edge_widths.append(5.0)  # Bolder edges
        else:
            full_edge_colors.append((0.5, 0.5, 0.5, 0.9))  # Gray for pipes
            full_edge_widths.append(3.0)  # Bolder pipes too

    # Setup Positions - Use Kamada-Kawai for better spacing
    pos = net.pos
    relevant_pos = {n: pos[n] for n in G_vis.nodes() if n in pos}
    
    if len(relevant_pos) < len(G_vis.nodes()) or len(relevant_pos) == 0:
        # Use Kamada-Kawai layout for better spacing
        relevant_pos = nx.kamada_kawai_layout(G_vis, scale=2.0)
    else:
        # Scale existing positions for better spacing
        xs = [p[0] for p in relevant_pos.values()]
        ys = [p[1] for p in relevant_pos.values()]
        x_range = max(xs) - min(xs) if xs else 1
        y_range = max(ys) - min(ys) if ys else 1
        scale_factor = 2.0 / max(x_range, y_range, 1)
        relevant_pos = {n: (p[0] * scale_factor, p[1] * scale_factor) for n, p in relevant_pos.items()}
        
    # Create figure with more space
    plt.figure(figsize=(16, 12))
    
    # Draw edges (no arrows - flow direction is implicit from source)
    nx.draw_networkx_edges(
        G_vis, 
        relevant_pos, 
        edge_color=full_edge_colors, 
        width=full_edge_widths,
        arrows=False,
        alpha=0.9
    )
    
    # Colorbar
    if sim_values:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), label='Simultaneity Factor', shrink=0.8)
        cbar.ax.tick_params(labelsize=10)

    # Edge Labels - show ALL edges (including single-consumer edges)
    edge_labels = {}
    consumer_map = {str(c.id): c for c in net.consumers}
    
    for u, v in G_vis.edges():
        eid = f"{u}->{v}"
        if eid in edge_obj_map:
            # Edge is in relevant_edges (has optimization data)
            e = edge_obj_map[eid]
            sim = e.calculate_current_simultaneity()
            n_cons = e.n_consumers
            edge_labels[(u, v)] = f"g={sim:.3f}\nn={n_cons}"
        else:
            # Check if this edge connects a consumer to a mixer (n=1 case)
            # u is the consumer node in "consumer->mixer" edges
            if str(u) in consumer_map:
                edge_labels[(u, v)] = f"g=1.000\nn=1"
            
    nx.draw_networkx_edge_labels(
        G_vis, relevant_pos, edge_labels=edge_labels, 
        font_size=8, font_color='darkblue',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9)
    )

    # Prepare node properties
    consumer_map = {str(c.id): c for c in net.consumers}
    node_labels = {}
    node_colors = []
    node_sizes = []
    
    for n in G_vis.nodes():
        n_type = G_vis.nodes[n].get('type', 'Unknown')
        
        if n in consumer_map:
            # Consumer node
            shift = consumer_map[n].time_shift
            shift_hours = shift / STEPS_PER_HOUR
            node_labels[n] = f"{n}\n{shift_hours:.2f}h"
            node_colors.append('#4ECDC4')  # Teal for consumers
            node_sizes.append(800)
        elif n_type == 'Source':
            # Source node - LARGEST
            node_labels[n] = f"SOURCE\n{n}"
            node_colors.append('#FF6B6B')  # Red-orange for source
            node_sizes.append(1500)
        elif n_type == 'Mixer':
            # Mixer node - SMALLEST
            node_labels[n] = f"{n}"
            node_colors.append('#95A5A6')  # Gray for mixers
            node_sizes.append(300)
        else:
            node_labels[n] = f"{n}"
            node_colors.append('#BDC3C7')
            node_sizes.append(400)

    # Draw nodes with variable sizes
    nx.draw_networkx_nodes(
        G_vis, relevant_pos, 
        node_size=node_sizes, 
        node_color=node_colors,
        edgecolors='black',
        linewidths=1.5
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G_vis, relevant_pos, 
        labels=node_labels, 
        font_size=8, 
        font_weight='bold'
    )

    # Title and legend
    plt.title(f"{title}\nEdge Labels: g=Simultaneity, n=Downstream Consumers", fontsize=14)
    
    # Add legend for node types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', markersize=15, label='Source'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', markersize=12, label='Consumer (ID + shift)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#95A5A6', markersize=8, label='Mixer'),
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.axis('off')
    plt.tight_layout()
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print("Saved.")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close()

def parse_simultaneity_spline(filepath):
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        # Namespace map often needed for VICUS/IBK
        # The user example had <IBK:LinearSpline name="Simultaneity">
        # We need to handle namespaces carefully or just search by tag name if possible?
        # ET handles namespaces with {uri}tag syntax.
        
        # Brute force search for any LinearSpline with name="Simultaneity"
        # We look for all elements and check attributes/tags
        spline_x = []
        spline_y = []
        
        # Searching...
        found = False
        for elem in root.iter():
            if 'LinearSpline' in elem.tag and elem.get('name') == 'Simultaneity':
                # Found it
                for child in elem:
                    if 'X' in child.tag:
                        spline_x = [float(val) for val in child.text.split()]
                    if 'Y' in child.tag:
                        spline_y = [float(val) for val in child.text.split()]
                found = True
                break
        
        if found and spline_x and spline_y:
            print(f"Parsed Simultaneity Spline: {len(spline_x)} points.")
            return LinearSpline(spline_x, spline_y)
        else:
            print("Simultaneity Spline not found in XML. Using default.")
            return None
            
    except Exception as e:
        print(f"Error parsing XML for spline: {e}")
        return None

def load_real_profiles(directory):
    """Load TSV profiles - always hourly resolution."""
    profiles = []
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []
        
    for filename in os.listdir(directory):
        if filename.endswith(".tsv"):
            filepath = os.path.join(directory, filename)
            
            try:
                vals = []
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    for line in lines[1:]:  # Skip header
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            try:
                                val = float(parts[1])
                                vals.append(val)
                            except ValueError:
                                pass
                
                if vals:
                    peak = max(vals) if vals else 1.0
                    if peak == 0: peak = 1.0
                    normalized = [v / peak for v in vals]
                    profiles.append(normalized)
                    print(f"Loaded: {filename} ({len(normalized)} hourly values)")
                    
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return profiles


def load_network_from_vicus(filepath):
    print(f"Loading VICUS file: {filepath}")
    
    # 1. Parse Spline
    sim_spline = parse_simultaneity_spline(filepath)
    
    # 2. Parse Graph
    G_nx, sources, consumer_ids, pos = create_graph_from_xml(filepath)
    if not sources: raise ValueError("No Source node found.")
    
    root = sources[0]
    try:
        T_directed = nx.bfs_tree(G_nx, root)
    except Exception as e:
        print(f"Error determining flow direction: {e}")
        return None, []

    net = Network()
    net.G_nx = G_nx # Store full graph
    net.simultaneity_spline = sim_spline
    net.pos = pos
    
    # Load Real Profiles
    real_profiles = []
    
    target_filename = "Residential_SingleFamily_HeatingLoad.tsv"

    specific_path = os.path.join("times-series", target_filename)
    if os.path.exists(specific_path):
        vals = []
        try:
             with open(specific_path, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        try:
                            vals.append(float(parts[1]))
                        except ValueError: pass
        except Exception as e:
            print(f"Error loading {target_filename}: {e}")
            
        if vals:
            # Extract a single representative day (24 hours) from annual data
            # Find the day with the MAXIMUM peak heating demand using RAW values
            num_days = len(vals) // 24
            max_peak = -1
            peak_day_index = 0
            
            for day in range(num_days):
                day_start = day * 24
                day_end = day_start + 24
                day_peak = max(vals[day_start:day_end])
                if day_peak > max_peak:
                    max_peak = day_peak
                    peak_day_index = day
            
            start_hour = peak_day_index * 24
            end_hour = start_hour + 24
            
            if len(vals) >= end_hour:
                daily_profile_raw = vals[start_hour:end_hour]
                print(f"Found peak day: day {peak_day_index+1} (hours {start_hour}-{end_hour-1}) with peak={max_peak:.4f}")
            else:
                # Fallback: take first 24 hours if file is shorter than expected
                daily_profile_raw = vals[:24] if len(vals) >= 24 else vals
                max_peak = max(daily_profile_raw) if daily_profile_raw else 1.0
                print(f"Using first 24 hours as representative day")
            
            # NOW normalize the representative day
            normalized_daily = [v / max_peak if max_peak > 0 else 0 for v in daily_profile_raw]
            
            # Resample for the simulation resolution
            resampled = resample_vector(normalized_daily, STEPS_PER_HOUR)
            real_profiles = [resampled]
            print(f"Force-selected profile: {target_filename} (peak day extraction, resampled to {len(real_profiles[0])} steps)")
            
            # Save a plot of the peak day profile for verification
            plot_peak_day(resampled, "output/peak_day_profile.png")
    
    # Fallback if specific file failed
    if not real_profiles:
        print("Specific profile not found/loaded. Using synthetic base profile.")
        exit()
    
    node_to_consumer = {}
    node_attrs = nx.get_node_attributes(G_nx, 'heating_demand')
    
    for c_id in consumer_ids:
        demand = node_attrs.get(c_id, 0)
        if demand <= 0: demand = 10.0
        
        # Randomly assign a profile
        assigned_profile = random.choice(real_profiles)
        
        c = Consumer(c_id, demand, assigned_profile)
        net.add_consumer(c)
        node_to_consumer[c_id] = c
        
    reachable_consumers = {}
    dfs_nodes = list(nx.dfs_postorder_nodes(T_directed, root))
    
    for n in dfs_nodes:
        consumers_here = []
        if n in node_to_consumer:
            consumers_here.append(node_to_consumer[n])
        for child in T_directed.successors(n):
            if child in reachable_consumers:
                consumers_here.extend(reachable_consumers[child])
        reachable_consumers[n] = consumers_here
        
    relevant_edges = []
    count = 0
    for u, v in T_directed.edges():
        edge_id = f"{u}->{v}"
        downstream = reachable_consumers.get(v, [])
        if downstream:
            edge = Edge(edge_id, downstream, simultaneity_spline=sim_spline)
            net.add_edge(edge)
            relevant_edges.append(edge)
            count += 1
            
    print(f"Created {len(net.consumers)} consumers and {count} relevant optimization edges.")
    return net, relevant_edges

# --- Main ---

def plot_simultaneity_curve(relevant_edges, output_path, title="Simultaneity Factor vs Downstream Consumers"):
    print(f"Generating simultaneity curve: {output_path}")
    
    n_consumers = []
    actual_sim = []
    target_sim = []
    
    for edge in relevant_edges:
        n = edge.n_consumers
        if n > 0:
            n_consumers.append(n)
            actual_sim.append(edge.calculate_current_simultaneity())
            target_sim.append(edge.get_target_simultaneity())
            
    if not n_consumers:
        print("No data to plot for simultaneity curve.")
        return

    plt.figure(figsize=(10, 6))
    
    # Plot Actual
    plt.scatter(n_consumers, actual_sim, color='blue', label='Actual Optimization Result', alpha=0.7)
    
    # Plot Target (Sort for line plot)
    sorted_pairs = sorted(zip(n_consumers, target_sim))
    sorted_n, sorted_t = zip(*sorted_pairs)
    
    # Generate a smooth target curve for reference if possible
    # We can use the spline from the first edge if available, or just the points we have
    # For better visualization, let's query the target function over a range
    
    # Check if we have a spline
    spline = relevant_edges[0].simultaneity_spline if relevant_edges else None
    if spline:
        x_range = range(min(n_consumers), max(n_consumers) + 5)
        y_range = [spline.value(x) for x in x_range]
        plt.plot(x_range, y_range, color='green', linestyle='--', label='Target Curve (Spline)')
    else:
        # Fallback to scatter/line of points we have
         plt.plot(sorted_n, sorted_t, color='green', linestyle='--', label='Target Curve (Points)')
         
    plt.xlabel("Number of Downstream Consumers")
    plt.ylabel("Simultaneity Factor")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print("Saved.")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close()

def plot_peak_load_comparison(consumers, output_path):
    """
    Plot Peak Load vs Consumer ID (Original vs Shifted) 
    to verify that interpolation preserves the peak magnitude.
    """
    print(f"Generating peak load verification plot: {output_path}")
    
    # Sort consumers by ID for a consistent x-axis
    sorted_consumers = sorted(consumers, key=lambda x: str(x.id))
    
    c_ids = [str(c.id) for c in sorted_consumers]
    original_peaks = []
    shifted_peaks = []
    
    for c in sorted_consumers:
        # Original peak (scaled by peak_load)
        orig_p = c.peak_load * max(c.base_profile)
        # Shifted peak (from the high-res pre-calculated profile)
        shift_p = max(c.get_shifted_profile())
        
        original_peaks.append(orig_p)
        shifted_peaks.append(shift_p)
        
    plt.figure(figsize=(12, 6))
    
    x = range(len(c_ids))
    plt.plot(x, original_peaks, 'o-', color='blue', label='Original Peak Load', alpha=0.6, markersize=4)
    plt.plot(x, shifted_peaks, 'x--', color='red', label='Shifted Peak Load (Interpolated)', alpha=0.9, markersize=4)
    
    plt.xticks(x, c_ids, rotation=45, fontsize=8)
    plt.xlabel("Consumer ID")
    plt.ylabel("Peak Load [W or scaled]")
    plt.title("Peak Load Verification: Original vs. Interpolated Shift")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=300)
        print("Saved.")
    except Exception as e:
        print(f"Error saving verification plot: {e}")
    finally:
        plt.close()

def plot_profile_verification(consumers, output_path, num_to_plot=3):
    """
    Plot the 24h profile of selected consumers before and after shifting 
    to verify shape preservation and correct horizontal shifting.
    """
    print(f"Generating profile shift verification plot: {output_path}")
    
    # Pick a few representative consumers (first few by ID)
    selected = sorted(consumers, key=lambda x: str(x.id))[:num_to_plot]
    
    if not selected: return
    
    fig, axes = plt.subplots(len(selected), 1, figsize=(12, 5 * len(selected)), sharex=True)
    if len(selected) == 1: axes = [axes]
    
    profile_len = len(selected[0].base_profile)
    hours = [i / STEPS_PER_HOUR for i in range(profile_len)]
    
    for i, c in enumerate(selected):
        orig = vec_scale(c.base_profile, c.peak_load)
        shifted = c.get_shifted_profile()
        
        # Find peak positions for markers
        orig_peak_idx = orig.index(max(orig))
        shifted_peak_idx = shifted.index(max(shifted))
        
        orig_peak_h = orig_peak_idx / STEPS_PER_HOUR
        shifted_peak_h = shifted_peak_idx / STEPS_PER_HOUR
        
        ax = axes[i]
        ax.plot(hours, orig, label=f"Original Profile", color='gray', alpha=0.5, linestyle='--')
        ax.plot(hours, shifted, label=f"Shifted Profile", color='blue', alpha=0.9, linewidth=2)
        
        # Vertical markers for peak visual verification
        ax.axvline(x=orig_peak_h, color='gray', linestyle=':', alpha=0.6, label=f"Orig Peak @ {orig_peak_h:.2f}h")
        ax.axvline(x=shifted_peak_h, color='red', linestyle='--', alpha=0.8, label=f"Shifted Peak @ {shifted_peak_h:.2f}h")
        
        ax.set_ylabel("Load [W]")
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.2)
        
        # Explicit title showing the delta
        calc_shift = shifted_peak_h - orig_peak_h
        # Handle wrap around calculation for the title display
        if calc_shift > 12: calc_shift -= 24
        if calc_shift < -12: calc_shift += 24
        
        ax.set_title(f"Consumer {c.id}: Shift = {c.time_shift/STEPS_PER_HOUR:+.2f}h (Visual Peak Move: {calc_shift:+.2f}h)")
        
    axes[-1].set_xlabel("Hour of Day")
    plt.tight_layout()
    
    try:
        plt.savefig(output_path, dpi=200)
        print("Saved.")
    except Exception as e:
        print(f"Error saving profile verification plot: {e}")
    finally:
        plt.close()

def main():
    start_time = time.time()
    filepath = sys.argv[1] if len(sys.argv) > 1 else None
    # Allow optional second argument for output directory
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    if filepath and os.path.exists(filepath):
        net, relevant_edges = load_network_from_vicus(filepath)
    else:
       print("Usage: python3 time_shift_calculations.py <vicus_file> [output_dir]")
       return
    
    if not relevant_edges: return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_name = os.path.basename(filepath)
    base_name_no_ext = os.path.splitext(base_name)[0]

    # Initial Visualization
    output_vis_initial = os.path.join(output_dir, f"visualization_{base_name_no_ext}_initial.png")
    visualize_network(net, relevant_edges, output_vis_initial, title="Initial Network State")

    print("\n--- BEFORE ---")
    print_edges = relevant_edges if len(relevant_edges) < 10 else relevant_edges[:10]
    for edge in print_edges:
        print(f"{edge.id}: Cons={edge.n_consumers} Target={edge.get_target_simultaneity():.4f}, Actual={edge.calculate_current_simultaneity():.4f}")
        
    multi_start_adaptive_hill_climbing(net, relevant_edges, num_restarts=5, max_iterations_per_phase=15)
    
    print("\n--- AFTER ---")
    for edge in print_edges:
        print(f"{edge.id}: Cons={edge.n_consumers} Target={edge.get_target_simultaneity():.4f}, Actual={edge.calculate_current_simultaneity():.4f}")

    shifts = [c.time_shift for c in net.consumers]
    if shifts:
        print(f"\nShifts: Min={min(shifts)/STEPS_PER_HOUR}h, Max={max(shifts)/STEPS_PER_HOUR}h")
        print("\n--- INDIVIDUAL SHIFTS ---")
        print(f"{'Consumer ID':<15} | {'Shift (h)':<10}")
        print("-" * 65)
        sorted_consumers = sorted(net.consumers, key=lambda x: int(x.id) if str(x.id).isdigit() else str(x.id))
        
        for c in sorted_consumers:
            print(f"{str(c.id):<15} | {c.time_shift/STEPS_PER_HOUR:<10.2f}")

    print(f"\nElapsed time: {time.time() - start_time:.2f}s")
    
    output_vis_final = os.path.join(output_dir, f"visualization_{base_name_no_ext}_final.png")
    visualize_network(net, relevant_edges, output_vis_final, title="Optimized Network State")
    
    # Plot Simultaneity Curve
    output_curve = os.path.join(output_dir, f"simultaneity_curve_{base_name_no_ext}.png")
    plot_simultaneity_curve(relevant_edges, output_curve)

    # Plot Peak Load Verification (requested by user)
    output_verify = os.path.join(output_dir, f"peak_verification_{base_name_no_ext}.png")
    plot_peak_load_comparison(net.consumers, output_verify)

    # Plot Profile Shift Verification (requested by user)
    output_shift_verify = os.path.join(output_dir, f"profile_shift_verification_{base_name_no_ext}.png")
    plot_profile_verification(net.consumers, output_shift_verify)

if __name__ == "__main__":
    main()
