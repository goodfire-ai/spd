#!/usr/bin/env python3
"""
Split error cases into discovery and test sets for the graph-informed SFT experiment.

Discovery set: Used to generate attribution graphs and inform training data creation
Test set: Held out for final evaluation
"""

import json
import random

# Set seed for reproducibility
random.seed(42)

def main():
    # Load error cases
    with open("data/twohop_error_cases.json") as f:
        all_cases = json.load(f)

    # Filter to only "outputs_country" cases
    country_errors = [c for c in all_cases if c["error_type"] == "outputs_country"]

    print(f"Total error cases: {len(all_cases)}")
    print(f"Clear 'outputs_country' errors: {len(country_errors)}")

    # Select discovery set: diverse regions, mix of capital probabilities
    # Goal: 8 cases for discovery, rest for test

    # Manually select diverse discovery cases:
    discovery_landmarks = [
        "the Great Barrier Reef",    # Australia, capital prob 27%
        "the Taj Mahal",              # India, capital prob 13%
        "Red Square",                 # Russia, capital prob 30%
        "the Atomium",                # Belgium, capital prob 27%
        "Angkor Wat",                 # Cambodia, capital prob 15%
        "Christ the Redeemer",        # Brazil, capital prob 0.3%
        "Mount Kilimanjaro",          # Tanzania, capital prob 31%
        "the Matterhorn",             # Switzerland, capital prob 17%
    ]

    discovery_set = []
    test_set = []

    for case in country_errors:
        if case["landmark"] in discovery_landmarks:
            discovery_set.append(case)
        else:
            test_set.append(case)

    print(f"\nDiscovery set: {len(discovery_set)} cases")
    print(f"Test set: {len(test_set)} cases")

    # Save splits
    with open("data/discovery_set.json", "w") as f:
        json.dump(discovery_set, f, indent=2)

    with open("data/test_set.json", "w") as f:
        json.dump(test_set, f, indent=2)

    # Print discovery set details
    print("\n" + "=" * 80)
    print("DISCOVERY SET (for graph generation and training data creation):")
    print("-" * 80)
    for case in discovery_set:
        print(f"  {case['landmark']}")
        print(f"    Country: {case['country']} ({case['country_prob']:.1%})")
        print(f"    Capital: {case['capital']} ({case['capital_prob']:.1%})")
        print()

    print("=" * 80)
    print("TEST SET (held out for evaluation):")
    print("-" * 80)
    for case in test_set:
        print(f"  {case['landmark']}: {case['country']} → {case['capital']}")

    print("\nSaved to data/discovery_set.json and data/test_set.json")

if __name__ == "__main__":
    main()
