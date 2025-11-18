# Quick Node Test Guide

## TL;DR

```bash
source .venv/bin/activate

# 1. Set config to use fewer steps
python set_test_steps.py 300 --backup

# 2. Check which nodes are available
python check_nodes.py

# 3. Run the test (adjust nodes if needed)
python test_node_reliability.py --n-runs 10 --steps 300

# 4. Wait for jobs to complete (monitor with squeue)
squeue --me

# 5. Analyze results (use job IDs from step 3)
python analyze_node_test_results.py <suspect_job_id> <control_job_id>

# 6. Restore config
python set_test_steps.py --restore
```

## What This Does

- Runs the same experiment 10 times on a suspect node and 10 times on a control node
- Each run is short (300 steps ≈ 2-5 minutes)
- Compares failure rates to determine if the suspect node is problematic
- Provides statistical evidence of node reliability issues

## Understanding Results

- **Strong Evidence**: Suspect has ≥2 failures, control has 0 → Node is likely faulty
- **Moderate Evidence**: Suspect has >2x failures → Node may be faulty
- **Weak Evidence**: Small difference → Results inconclusive, run more tests
- **No Evidence**: Equal rates → Node appears fine

## Files Created

- `test_node_reliability.py` - Main test script
- `analyze_node_test_results.py` - Results analysis
- `check_nodes.py` - Check node availability
- `set_test_steps.py` - Helper to modify config
- `NODE_TEST_README.md` - Full documentation

## Key Options

### test_node_reliability.py
- `--n-runs N` - Runs per node (default: 10)
- `--steps N` - Training steps per run (default: 300)
- `--suspect-node NAME` - Node to test
- `--control-node NAME` - Comparison node
- `--dp N` - GPUs for data parallelism (default: 8)

### spd-run (now supports nodelist)
```bash
# Run on specific node(s)
spd-run --experiments ss_gpt2_simple --dp 8 --nodelist h200-reserved-145-005

# Avoid problematic node
spd-run --experiments ss_gpt2_simple --dp 8 --nodelist h200-reserved-145-005,h200-reserved-145-007
```
