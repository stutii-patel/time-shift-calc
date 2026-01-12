
import math
import random
import time
import sys
import os
import networkx as nx
import xml.etree.ElementTree as ET

# Import the user's graph creator
try:
    from create_graph import create_graph_from_xml
except ImportError:
    print("Warning: create_graph.py not found. Only synthetic mode available.")
    create_graph_from_xml = None

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
    def __init__(self, id, peak_load, base_profile):
        self.id = id
        self.peak_load = peak_load
        self.base_profile = base_profile
        self.time_shift = 0 
        self._cached_profile = None
        self._cached_shift = None

    def get_shifted_profile(self):
        shift = int(round(self.time_shift))
        if self._cached_profile is None or self._cached_shift != shift:
             scaled = vec_scale(self.base_profile, self.peak_load)
             self._cached_profile = vec_roll(scaled, shift)
             self._cached_shift = shift
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
        
        if self.simultaneity_spline:
            return self.simultaneity_spline.value(self.n_consumers)
        
        # Fallback Formula
        a = 0.4497
        b = 0.5512
        c = 53.84
        return a + b * math.exp(-self.n_consumers / c)

class Network:
    def __init__(self):
        self.consumers = []
        self.edges = []
        self.consumer_to_edges = {} 
        self.simultaneity_spline = None

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

def run_hill_climbing_optimization(network, relevant_edges, max_iterations=20):
    print(f"Starting Fast Hill Climbing for {len(network.consumers)} consumers...")
    shifts_to_try = [1, -1, 3, -3, 6, -6, 12, -12] 
    
    for iteration in range(max_iterations):
        improved = False
        indices = list(range(len(network.consumers)))
        random.shuffle(indices)
        
        for i in indices:
            consumer = network.consumers[i]
            original_shift = consumer.time_shift
            original_profile = consumer.get_shifted_profile()
            
            best_local_shift = original_shift
            best_local_improvement = 0
            
            affected_edges = network.consumer_to_edges[consumer.id]
            base_edge_error = calculate_local_error(affected_edges)
            
            for step in shifts_to_try:
                new_shift = original_shift + step
                if abs(new_shift) > 12: continue
                
                consumer.time_shift = new_shift
                new_profile = consumer.get_shifted_profile()
                
                for edge in affected_edges:
                    edge.update_profile(original_profile, new_profile)
                    
                new_edge_error = calculate_local_error(affected_edges)
                improvement = base_edge_error - new_edge_error
                
                if improvement > best_local_improvement + 1e-6:
                    best_local_improvement = improvement
                    best_local_shift = new_shift
                
                for edge in affected_edges:
                    edge.update_profile(new_profile, original_profile)
                    
                consumer.time_shift = original_shift
                consumer._cached_shift = int(round(original_shift))
                consumer._cached_profile = original_profile

            if best_local_improvement > 0:
                consumer.time_shift = best_local_shift
                new_profile = consumer.get_shifted_profile()
                for edge in affected_edges:
                    edge.update_profile(original_profile, new_profile)
                improved = True
        
        current_total_error = calculate_local_error(relevant_edges)
        print(f"Iteration {iteration+1}: Error = {current_total_error:.6f}")
        if not improved:
            print("Converged.")
            break
            
    return calculate_local_error(relevant_edges)

# --- Graph Import Logic ---

def create_base_profile():
    base_day = []
    for h in range(24):
        val = math.exp(-(h - 12)**2 / (2 * 4)) 
        base_day.append(val)
    return base_day * 365

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
    net.simultaneity_spline = sim_spline
    base_profile_year = create_base_profile()
    
    node_to_consumer = {}
    node_attrs = nx.get_node_attributes(G_nx, 'heating_demand')
    
    for c_id in consumer_ids:
        demand = node_attrs.get(c_id, 0)
        if demand <= 0: demand = 10.0
        c = Consumer(c_id, demand, base_profile_year)
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

def create_synthetic_network():
    print("Creating synthetic network...")
    net = Network()
    base_profile_year = create_base_profile()
    # No spline in synthetic
    
    all_consumers = []
    for i in range(100):
        peak = 10 + random.uniform(-2, 2)
        c = Consumer(i, peak, base_profile_year)
        net.add_consumer(c)
        all_consumers.append(c)
        
    edge_Main = Edge("Edge_Main", all_consumers, None)
    net.add_edge(edge_Main)
    
    return net, [edge_Main]

def main():
    start_time = time.time()
    filepath = sys.argv[1] if len(sys.argv) > 1 else None
    
    if filepath and os.path.exists(filepath):
        net, relevant_edges = load_network_from_vicus(filepath)
    else:
        net, relevant_edges = create_synthetic_network()
    
    if not relevant_edges: return

    print("\n--- BEFORE ---")
    print_edges = relevant_edges if len(relevant_edges) < 10 else relevant_edges[:10]
    for edge in print_edges:
        print(f"{edge.id}: Cons={edge.n_consumers} Target={edge.get_target_simultaneity():.4f}, Actual={edge.calculate_current_simultaneity():.4f}")
        
    run_hill_climbing_optimization(net, relevant_edges)
    
    print("\n--- AFTER ---")
    for edge in print_edges:
        print(f"{edge.id}: Cons={edge.n_consumers} Target={edge.get_target_simultaneity():.4f}, Actual={edge.calculate_current_simultaneity():.4f}")

    shifts = [c.time_shift for c in net.consumers]
    if shifts:
        print(f"\nShifts: Min={min(shifts)}, Max={max(shifts)}")
        print("\n--- INDIVIDUAL SHIFTS ---")
        print(f"{'Consumer ID':<15} | {'Shift (Hours)':<15}")
        print("-" * 35)
        sorted_consumers = sorted(net.consumers, key=lambda x: int(x.id) if str(x.id).isdigit() else str(x.id))
        
        for c in sorted_consumers:
             print(f"{str(c.id):<15} | {int(c.time_shift):<15}")

    print(f"\nElapsed time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
