# Experiment Log

Log of experiments run on neuron attribution graphs. Add new entries at the top.

---

## Template

```
### YYYY-MM-DD: [Experiment Name]

**Goal**: What are we trying to learn?

**Method**:
- Script/command used
- Key parameters

**Results**:
- Key findings
- Artifacts generated (link to ARTIFACTS.md section)

**Next Steps**: What to do with these results
```

---

## Experiments

### 2026-01-22: Multiplicative Neuron Steering for Resveratrol-Longevity Unlearning

**Goal**: Test position-specific (multiplicative) in-MLP steering to selectively unlearn the resveratrol-longevity association while preserving other longevity knowledge.

**Method**:
- Target: L8/N11963 (resveratrol/longevity intervention detector)
- Multiplicative steering: Only steer at token positions where the neuron is active (vs additive which steers all positions)
- Training data: 8 prompts linking resveratrol to longevity/sirtuins/aging
- Evaluation: 4 target prompts (should lose longevity), 4 control prompts (metformin, rapamycin, senolytics, CR mimetics - should keep longevity), 12 indirect elicitation prompts

**Implementation**:
- Modified `goodfire-train/goodfire_train/config.py` to allow `steer_on_active_only` with neuron steering
- Updated `goodfire-train/goodfire_train/trainer.py` to use `compute_neuron_position_mask`
- Added `--steer-on-active-only` and `--activation-threshold` to `resveratrol_unlearning.py`
- Created `scripts/launch_sweep_slurm.py` with `--multiplicative` flag

**Sweep Results**:

| Alpha | Epochs | Target Unlearned | Controls Preserved | Notes |
|-------|--------|------------------|-------------------|-------|
| ±10 | 5 | 31% (1-2/4) | 100% | Initial baseline |
| ±30 | 5 | 54% (2-3/4) | 100% | Higher alpha helps |
| ±50 | 5 | 54% (2-3/4) | 100% | Diminishing returns |
| +10 | 30 | 62% (2-3/4) | 100% | More epochs effective |
| -10 | 30 | 54% | 75% | Negative alpha damages controls |
| **+30** | **10** | **62%** | **100%** | **★ BEST CONFIG** |
| +20 | 30 | 62% | 100% | Also good |
| +50 | 20 | 85% | 90% | Max unlearning but loses 1 control |
| +100 | 30 | 85% | 90% | Overkill |

**Key Findings**:
1. **Positive alpha is safer**: Negative alphas damage controls at high epochs
2. **Best config**: α=+30, 10 epochs → 62% target unlearning, 100% controls
3. **Model coherence maintained**: Responses remain fully coherent even at α=100, 30 epochs
4. **Directional unlearning**: Successfully broke resveratrol→longevity but NOT longevity→resveratrol
   - Indirect elicitation: 0/12 success (model still mentions resveratrol when asked about longevity compounds)

**Circuit Analysis** (RelP comparison baseline vs trained):
- Neuron L8/N11963 activations reduced 10-20% on resveratrol prompts
- "rol" token activation: 1.21 → 1.00 (↓18%)
- Successfully unlearned prompts no longer mention longevity in completions
- Some prompts (especially sirtuin-related) still trigger longevity associations

**Artifacts**:
- Sweep outputs: `goodfire-train/outputs/sweep_mult_20260122_*/`
- Comparison script: `goodfire-train/scripts/compare_relp_circuits.py`
- Best checkpoint: `resv_mult_a30_e10.pt`

**Next Steps**:
- Try bidirectional training (both resveratrol→longevity AND longevity→resveratrol prompts)
- Test on other neurons/associations
- Investigate why sirtuin-related prompts resist unlearning

---

### 2025-01-14: Dashboard HTML Generator Agent

**Goal**: Build an agent to transform neuron investigation JSON into beautiful Distill.pub-style HTML pages.

**Method**:
- Created `neuron_scientist/dashboard_agent.py` using Claude Agent SDK
- Agent reads dashboard JSON, generates narrative content, writes HTML
- Two tools: `get_dashboard_data`, `write_html`

**Results**:
- Agent successfully generates HTML with neuron links and expandable experiments
- Test output: `frontend/reports/L4_N10555_test.html`
- Refactored after code review from alpha agent

**Next Steps**: Run on batch of investigations, compare quality with manual HTML

---

### 2025-01-13: Batch Neuron Investigations

**Goal**: Run neuron scientist agent on multiple neurons from v6 edge stats.

**Method**:
- `scripts/run_neuron_scientist.py` with neurons from `medical_edge_stats_v6_enriched.json`
- Max 100 experiments per neuron

**Results**:
- 20+ investigations completed in `outputs/investigations/`
- Dashboard and investigation JSON pairs generated
- See ARTIFACTS.md for full list

**Next Steps**: Generate HTML reports, analyze patterns across neurons

---

### 2025-01-12: Medical Edge Stats v6

**Goal**: Build comprehensive edge statistics from medical corpus with enriched neuron data.

**Method**:
- Generated RelP graphs for 579 substantive medical prompts
- Aggregated edge statistics across all graphs
- Enriched with DER (Direct Effect Ratio) and output projections

**Results**:
- `data/medical_edge_stats_v6.json` - raw edge stats
- `data/medical_edge_stats_v6_enriched.json` - with DER, projections, labels
- ~2000 neurons with significant activity

**Next Steps**: Use for neuron scientist investigations, SFT experiments

---

### 2025-01-11: Interactive Labeling Session

**Goal**: Manually label high-DER neurons to build ground truth.

**Method**:
- `scripts/interactive_labeling.py` with neurons sorted by DER
- Two-pass labeling: input function, then output function

**Results**:
- `data/interactive_labels.json` - manually verified labels
- Improved understanding of neuron function patterns

**Next Steps**: Use as training data for automated labeling

---

*Add new experiments above this line*
