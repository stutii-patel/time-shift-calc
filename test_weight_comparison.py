#!/usr/bin/env python3
"""
Test different edge weighting schemes on complex scenarios.
Tests: w_e = 1, w_e = n, w_e = n²
"""

import os
import sys
import subprocess
import shutil

# Test scenarios
SCENARIOS = [
    "data/test_scenarios/complex/NetzHast.vicus",
    "data/test_scenarios/complex/demoVerbundnetz_SimulationNeu.vicus",
    "data/test_scenarios/complex/test3.vicus",
]

# Weight configurations: (name, code_snippet)
WEIGHTS = [
    ("w_1", "w_e = 1"),
    ("w_n", "w_e = edge.n_consumers"),
    ("w_n2", "w_e = edge.n_consumers**2"),
]

def modify_weight(weight_code):
    """Modify the weight in time_shift_calculations.py"""
    with open("time_shift_calculations.py", "r") as f:
        content = f.read()
    
    # Replace the weight line - match both occurrences (with and without TODO comment)
    import re
    # This regex matches "w_e = edge.n_consumers**2" optionally followed by comment
    pattern = r'w_e = edge\.n_consumers\*\*2( # TODO)?'
    replacement = f'{weight_code} # TEMP_TEST'
    content = re.sub(pattern, replacement, content)
    
    # Also replace if already modified (from previous run)
    pattern2 = r'w_e = .* # TEMP_TEST'
    content = re.sub(pattern2, replacement, content)
    
    with open("time_shift_calculations.py", "w") as f:
        f.write(content)

def restore_original():
    """Restore original weight (n²)"""
    with open("time_shift_calculations.py", "r") as f:
        content = f.read()
    
    import re
    pattern = r'w_e = .* # TEMP_TEST'
    # We need to be careful here. The first occurrence had "# TODO", the second didn't.
    # But since we replaced both with the same TEMP_TEST line, we can't distinguish them easily.
    # However, replacing both with "w_e = edge.n_consumers**2 # TODO" is harmless for the second line, just adds a comment.
    # Or better: Check context? No, simpler to just restore them to n^2.
    replacement = 'w_e = edge.n_consumers**2 # TODO'
    content = re.sub(pattern, replacement, content)
    
    with open("time_shift_calculations.py", "w") as f:
        f.write(content)

def run_test(scenario_path, weight_name, weight_code):
    """Run optimization with specific weight"""
    basename = os.path.basename(scenario_path).replace(".vicus", "")
    output_dir = f"output/weight_comparison/{basename}/{weight_name}"
    
    print(f"\n{'='*60}")
    print(f"Scenario: {basename}")
    print(f"Weight: {weight_name} ({weight_code})")
    print(f"{'='*60}")
    
    # Modify weight
    modify_weight(weight_code)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run optimization
    log_file = os.path.join(output_dir, "logs.log")
    cmd = ["python3", "time_shift_calculations.py", scenario_path, output_dir]
    
    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    if result.returncode == 0:
        print(f"✓ Success! Results in: {output_dir}")
    else:
        print(f"✗ Failed! Check logs: {log_file}")
    
    return result.returncode == 0

def main():
    print("="*60)
    print("EDGE WEIGHT COMPARISON TEST")
    print("="*60)
    print(f"Scenarios: {len(SCENARIOS)}")
    print(f"Weights: {len(WEIGHTS)}")
    print(f"Total tests: {len(SCENARIOS) * len(WEIGHTS)}")
    print("="*60)
    
    results = {}
    
    for scenario in SCENARIOS:
        basename = os.path.basename(scenario).replace(".vicus", "")
        results[basename] = {}
        
        for weight_name, weight_code in WEIGHTS:
            success = run_test(scenario, weight_name, weight_code)
            results[basename][weight_name] = success
    
    # Restore original
    restore_original()
    print(f"\n{'='*60}")
    print("Restored original weight (n²)")
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    for scenario, weights in results.items():
        print(f"\n{scenario}:")
        for weight, success in weights.items():
            status = "✓" if success else "✗"
            print(f"  {status} {weight}")
    
    print(f"\n{'='*60}")
    print("All results saved to: output/weight_comparison/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
