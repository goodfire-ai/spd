# Postprocess plan for s-55ea3f9b

```
spd-postprocess spd/postprocess/s-55ea3f9b.yaml --dependency 311644_1
```

## Job chain

```
311644_1 (Dan's training job, ~3h remaining)
└── harvest array [8 × GPU]
    └── harvest merge [CPU, 200G]
        ├── autointerp interpret [CPU, 2 GPU]
        ├── attribution array [8 × GPU]
        │   └── attribution merge [CPU, 200G]
        │       └── graph-interp [CPU, 240G, also depends on harvest merge]
```

## Job details

### 1. Harvest array (8 workers)

| Param | Value |
|-------|-------|
| GPUs | 1 per worker × 8 workers |
| Time | 12:00:00 |
| Dependency | afterok:311644_1 |
| n_batches | 20,000 |
| batch_size | 32 |
| activation_examples_per_component | 1,000 |
| pmi_token_top_k | 40 |

### 2. Harvest merge

| Param | Value |
|-------|-------|
| GPUs | 0 (CPU only) |
| Memory | 200G |
| Time | 04:00:00 |
| Dependency | afterok:<harvest_array> |

### 3. Autointerp interpret

| Param | Value |
|-------|-------|
| GPUs | 2 |
| Time | 12:00:00 |
| Dependency | afterok:<harvest_merge> |
| Model | google/gemini-3-flash-preview |
| Reasoning effort | low |
| Strategy | compact_skeptical |
| forbidden_words | [] (Pile model, not SimpleStories) |
| label_max_words | 5 |
| cost_limit_usd | 100 |
| Evals | none (skipped) |

### 4. Attribution array (8 workers)

| Param | Value |
|-------|-------|
| GPUs | 1 per worker × 8 workers |
| Time | 48:00:00 |
| Dependency | afterok:<harvest_merge> |
| n_batches | 10,000 |
| batch_size | 32 |
| ci_threshold | 0.0 |

### 5. Attribution merge

| Param | Value |
|-------|-------|
| GPUs | 0 (CPU only) |
| Memory | 200G |
| Time | 01:00:00 |
| Dependency | afterok:<attr_array> |

### 6. Graph interp

| Param | Value |
|-------|-------|
| GPUs | 0 (CPU only) |
| CPUs | 16 |
| Memory | 240G |
| Time | 24:00:00 |
| Dependency | afterok:<harvest_merge>:<attr_merge> |
| Model | google/gemini-3-flash-preview |
| Reasoning effort | low |
| attr_metric | attr_abs |
| top_k_attributed | 8 |
| max_examples | 20 |
| label_max_words | 8 |

## Total GPU-hours (worst case)

- Harvest: 8 GPU × 12h = 96 GPU-h
- Autointerp: 2 GPU × 12h = 24 GPU-h
- Attributions: 8 GPU × 48h = 384 GPU-h
- Graph interp: 0 GPU
- **Total: 504 GPU-h max** (actual will be much less — time limits are generous)

## Notes

- Never more than 8 GPUs running concurrently (harvest finishes before attributions start)
- Autointerp and attributions run in parallel (both depend on harvest merge only)
- Graph interp waits for both harvest merge and attribution merge
- All jobs use a shared git snapshot for code consistency
