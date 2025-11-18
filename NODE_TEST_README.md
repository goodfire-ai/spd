# Node Reliability Testing

This directory contains scripts to test whether specific SLURM nodes are causing job failures.

## Problem

Jobs running `spd-run --experiments ss_gpt2_simple --dp=8` have been failing intermittently, with suspicion that node `h200-dev-145-040` is the culprit.

**Note**: As of 2025-11-17, node `h200-dev-145-040` is DRAINED (reason: "GPU diags running"). This suggests the admins may have already identified issues with it. You can still test other nodes for reliability.

## Solution

We run the same experiment multiple times on both the suspect node and a control node, then compare failure rates.

## Setup

### 1. Update the config to use fewer steps

Use the helper script to set the number of training steps:

```bash
source .venv/bin/activate
python set_test_steps.py 300 --backup
```

This will:
- Update `spd/experiments/lm/ss_gpt2_simple_config.yaml` to use 300 steps
- Create a backup of the original config (use `--restore` to revert later)

Alternatively, manually edit `spd/experiments/lm/ss_gpt2_simple_config.yaml` and change `steps: 200000` to `steps: 300`.

This ensures each test run completes quickly (~2-5 minutes instead of hours).

### 2. Check available nodes (optional)

```bash
source .venv/bin/activate
python check_nodes.py
```

This will show you which nodes are available and recommend good candidates for testing.

### 3. Run the test

```bash
source .venv/bin/activate
python test_node_reliability.py --n-runs 10 --steps 300
```

Options:
- `--n-runs N`: Number of runs on each node (default: 10)
- `--steps N`: Number of training steps per run (default: 300)
- `--suspect-node NAME`: Node to test (default: h200-dev-145-040)
- `--control-node NAME`: Control node (default: h200-reserved-145-005)
- `--project NAME`: W&B project name (default: spd-node-test)
- `--dp N`: Number of GPUs for data parallelism (default: 8)

Example with custom settings:
```bash
python test_node_reliability.py --n-runs 5 --steps 200 --suspect-node h200-dev-145-040 --control-node h200-reserved-145-007
```

### 3. Monitor the jobs

The script will output job IDs and monitoring commands. You can watch progress with:

```bash
squeue -j <suspect_job_id>,<control_job_id>
```

Or check logs:
```bash
tail -f ~/slurm_logs/slurm-<job_id>_*.out
```

### 4. Analyze the results

Once all jobs complete (check with `squeue`), analyze the results:

```bash
python analyze_node_test_results.py <suspect_job_id> <control_job_id>
```

This will:
- Count successes and failures for each node
- Calculate success rates
- Provide a verdict on whether the suspect node is problematic
- Suggest next steps for investigation

## Example Output

```
════════════════════════════════════════════════════════════════════
ANALYSIS
════════════════════════════════════════════════════════════════════
⚠️  Suspect node has 60.0% lower success rate than control node!
Suspect: 4/10 success vs Control: 10/10 success

🔴 STRONG EVIDENCE: Suspect node has multiple failures while control has none!
```

## Interpretation

- **Strong Evidence** (suspect has ≥2 failures, control has 0): Node is likely faulty
- **Moderate Evidence** (suspect has >2x failures): Node may be faulty
- **Weak Evidence** (small difference): Results are inconclusive, run more tests
- **No Evidence** (equal rates): Node appears fine

## Using the --nodelist flag in spd-run

The `spd-run` command now supports restricting jobs to specific nodes:

```bash
# Run on specific node
spd-run --experiments ss_gpt2_simple --dp 8 --nodelist h200-reserved-145-005

# Run on multiple nodes
spd-run --experiments ss_gpt2_simple --dp 8 --nodelist h200-reserved-145-005,h200-reserved-145-007
```

This is useful for:
- Avoiding problematic nodes
- Testing specific hardware
- Debugging node-specific issues

## Cleanup

After testing, restore the original config:

```bash
python set_test_steps.py --restore
```

Or manually change `steps:` back to `200000` in `spd/experiments/lm/ss_gpt2_simple_config.yaml`.

## Notes

- The suspect node (h200-dev-145-040) is currently **drained** in SLURM (reason: "GPU diags running"), which suggests admins may have already identified issues
- Each test run uses the same configuration and code (via git snapshot) to ensure fair comparison
- All runs are logged to W&B for detailed inspection
- Consider running multiple test cycles to increase statistical confidence
- The node test creates git snapshot branches prefixed with `node-test-*` - you can delete these after testing
