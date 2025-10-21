# CI Decision Tree Visualization Plan

## High-Level Architecture

**Export:** Python creates one JSON per tree → **Display:** HTML/JS loads JSON and renders visualizations

## Data to Export (per tree)

### 1. Tree Metadata
- Layer index, target component, module key
- Metrics: AP, accuracy, balanced accuracy, prevalence
- Tree stats: depth, n_leaves

### 2. Tree Structure
- sklearn tree serialized to JSON (children_left, children_right, feature, threshold, value)
- Feature names mapped (e.g., "blocks.0.mlp.W_in:17")
- Which input components are used in splits

### 3. Activation Data
- **True activations:** continuous values + binary (after threshold) + histogram
- **Predicted activations:** probabilities + binary (threshold 0.5) + histogram
- Confusion matrix counts (TP, TN, FP, FN)

### 4. Token Visualizations
- Select ~5-10 example samples (random, or best/worst, or one per confusion category)
- For each sample:
  - Token strings (decoded from IDs)
  - True activation per token position
  - Predicted probability per token position
  - Which position has max activation

## Key Design Decisions

### 1. Sample Selection for Token Viz
Options:
- **Random sample** - Simple, representative
- **Stratified by confusion matrix** - One TP, TN, FP, FN example each
- **Ranked by error** - Worst predictions, most informative failures
- **User configurable?**

### 2. Histogram Binning
- How many bins? (50? 100? Configurable?)
- Linear or log scale?
- Separate histograms for true vs predicted, or overlay?

### 3. Tree Rendering
Options:
- **Text representation** - Simple indented tree as string
- **Interactive D3 tree** - Nicer but more complex
- **Hybrid** - Text by default, D3 optional

### 4. File Organization
```
ci_dt_vis/
├── index.html                 # Main viewer
├── data/
│   ├── tree_1_0.json          # One JSON per tree
│   ├── tree_1_1.json
│   └── metadata.json          # Index of all trees
├── js/
│   ├── sparklines.js          # Already exists
│   ├── token-display.js       # Already exists (modify)
│   ├── tree-display.js        # NEW
│   └── viewer.js              # NEW - main app
```

### 5. Tokenizer Access
- Need tokenizer to decode token IDs → strings
- Options:
  - Pass tokenizer to export function
  - Save tokenizer path in JSON, load in Python
  - Pre-decode all tokens during export

## Implementation Flow

### Python Side (`export.py`)

```python
def export_tree_json(
    tree: LayerModel,
    target_idx: int,
    X: np.ndarray,           # input features
    Y_true: np.ndarray,       # true activations (binary)
    Y_prob: np.ndarray,       # predicted probs
    tokens_data: dict,        # sample tokens + activations
    output_path: str
) -> None:
    """Export single tree to JSON."""

def export_all_trees(
    models: list[LayerModel],
    layers_true: list[np.ndarray],
    per_layer_stats: list[dict],
    batch_data: dict,         # Need tokens from dataloader
    tokenizer,                # For decoding
    output_dir: str
) -> None:
    """Main export function - call from run.py."""
```

**Key functions to implement:**
- `tree_to_dict()` - Serialize sklearn tree
- `compute_histogram()` - Bin activations for sparklines
- `extract_sample_tokens()` - Get token strings + activations for N samples
- `get_input_features()` - Map tree features back to layer/component names

### JS Side (`viewer.js`)

```javascript
// Main viewer logic
async function loadTree(layer, targetIdx) {
    const data = await fetch(`data/tree_${layer}_${targetIdx}.json`);

    displayMetadata(data.metadata);
    displayHistograms(data.activations);      // Use sparklines.js
    displayTokens(data.tokens);               // Use token-display.js (modified)
    displayTree(data.tree);                   // NEW tree-display.js
    displayInputFeatures(data.input_features);
}
```

**Key functions to implement:**
- `displayHistograms()` - Call sparkbars() with true (blue) and predicted (red)
- `displayTokens()` - Modified token-display.js to overlay true (blue) + pred (red)
- `displayTree()` - Render tree structure (text or D3)

### Modified `token-display.js`

Need new function:
```javascript
function createDualActivationVisualization(tokens, trueActs, predProbs) {
    // Overlay blue (true) and red (predicted)
    // Perfect prediction = purple
    // False negative = blue only
    // False positive = red only
}
```

## JSON Schema Example

```json
{
  "metadata": {
    "layer_index": 1,
    "target_idx": 5,
    "module_key": "blocks.0.mlp.W_gate",
    "metrics": {"ap": 0.85, "accuracy": 0.92}
  },
  "tree": {
    "structure": {
      "children_left": [1, -1, 3, ...],
      "feature": [7, -2, 12, ...],
      "feature_names": ["blocks.0.attn.W_Q:3", ...]
    }
  },
  "activations": {
    "true": {
      "histogram": {"bins": [...], "counts": [...]}
    },
    "predicted": {
      "histogram": {"bins": [...], "counts": [...]}
    }
  },
  "tokens": {
    "data": [
      {
        "tokens": ["The", "cat", "sat", ...],
        "true_activations": [0.0, 0.0, 0.8, ...],
        "predicted_probabilities": [0.05, 0.1, 0.85, ...]
      }
    ]
  },
  "input_features": {
    "blocks.0.attn.W_Q": [3, 17, 42],  // Which components used
    "blocks.0.mlp.W_in": [5, 12]
  }
}
```

## Open Questions

1. **How many token samples to show?** 5? 10? 50?
2. **Sample selection strategy?** Random vs stratified vs worst-predictions?
3. **Tree display?** Text vs D3.js interactive?
4. **Single big JSON vs many small files?** (Currently: many small files)
5. **Need continuous activations or just binary?** (Binary sufficient for token viz?)
6. **Histogram bins?** How many, linear or log scale?
