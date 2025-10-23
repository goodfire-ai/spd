# CI Decision Tree Visualization Plan

## Overview

This document outlines the complete visualization strategy for causal importance decision trees, including static plots (matplotlib/PDF) and interactive visualizations (HTML/JS).

---

## Part 1: Static Plot Improvements

### 1.1 Layer Metrics - Distribution Plots

**Current:** Bar charts for mean AP, accuracy, balanced accuracy per layer

**New:** Scatter plots with horizontal jitter showing full distribution per layer

**Implementation:**
- Replace `plot_layer_metrics()` bar charts with jittered scatter plots
- For each layer, show all target component metrics as points with random horizontal jitter
- Add mean/median line overlays
- Better titles explaining metrics in terms of confusion matrix:

```python
# Accuracy title
r"Accuracy per Target Component\n" +
r"$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$"

# Balanced Accuracy title
r"Balanced Accuracy per Target Component\n" +
r"$\text{Balanced Acc} = \frac{1}{2}\left(\frac{TP}{TP+FN} + \frac{TN}{TN+FP}\right)$"

# Average Precision title
r"Average Precision per Target Component\n" +
r"$\text{AP} = \sum_n (R_n - R_{n-1}) P_n$" + "\n" +
r"where $P_n = \frac{TP}{TP+FP}$ (precision), $R_n = \frac{TP}{TP+FN}$ (recall)"
```

**Rationale:** Shows full distribution of performance across targets, not just means. More informative about variance in tree quality.

---

### 1.2 AP vs Prevalence Plot

**Current:** Simple scatter plot with alpha=0.6

**New Improvements:**
1. **Log x-axis** for prevalence (many rare components)
2. **No marker edges** (set `edgecolors='none'`)
3. **Color by tree depth** using viridis colormap
4. **Enhanced title:**
   ```python
   r"Average Precision vs Component Prevalence\n" +
   r"Prevalence = $\frac{n_\text{active samples}}{n_\text{total samples}}$"
   ```

**Additional:** Add heatmap version (see 1.3 below)

---

### 1.3 Tree Statistics - New Heatmaps

**Current:** Has depth vs accuracy, leaf count vs accuracy, depth vs leaf count heatmaps

**New Addition:** AP vs Prevalence heatmap

**Implementation:**
- Add new heatmap to `plot_tree_statistics()`:
  - x-axis: prevalence bins (log scale, e.g. [0.001, 0.01, 0.1, 0.5, 1.0])
  - y-axis: AP bins (linear, 0 to 1)
  - color: log10(count + 1) as in existing heatmaps
  - title:
    ```python
    r"Tree Performance vs Component Prevalence\n" +
    r"AP = Average Precision, Prev = $\frac{n_\text{active}}{n_\text{total}}$"
    ```

**Rationale:** Complements the scatter plot; easier to see density patterns.

---

### 1.4 Global Title Improvements

**Rules:**
- Use LaTeX notation via raw strings: `r"$\text{TP}$"` not unicode "TP"
- Use `\n` for line breaks in long titles
- Explain abbreviations and formulas
- Be explicit about what's plotted

**Examples:**

```python
# Before
"Covariance of components (all layers)"

# After
r"Component Coactivation Matrix\n" +
r"$\text{Cov}(i,j) = \mathbb{E}[(A_i - \mu_i)(A_j - \mu_j)]$\n" +
r"where $A_i$ is binary activation of component $i$"

# Before
"Tree depth"

# After
r"Distribution of Decision Tree Depths\n" +
r"(Depth = longest path from root to leaf)"

# Before
"Activations (True)"

# After
r"True Binary Activations\n" +
r"$A_{ij} = \mathbb{1}[\text{activation}_{ij} > \theta]$, $\theta = $" + f"{config.activation_threshold}"
```

---

### 1.5 Activations Plot - Sorting and Diff

**Current:** Two subplots (true, predicted) with no ordering

**New Architecture:**

```
plot_activations_unsorted(...)  # Original style with layer boundaries
plot_activations_sorted(...)    # New sorted version with diff
```

#### 1.5.1 Unsorted Version (Enhanced)

**Changes:**
- Add layer boundary lines and labels (borrow from `spd/clustering/plotting/activations.py:add_component_labeling()`)
- Show module names on y-axis (component dimension)
- Keep samples unsorted on x-axis
- Two subplots: true, predicted

**Implementation:**
```python
def plot_activations_unsorted(
    layers_true: list[np.ndarray],
    layers_pred: list[np.ndarray],
    module_keys: list[str],  # NEW: need module names
) -> None:
    """Show true and predicted activations with layer boundaries."""
    # Concatenate
    A_true = np.concatenate(layers_true, axis=1)
    A_pred = np.concatenate(layers_pred, axis=1)

    # Create component labels like "blocks.0.attn:0", "blocks.0.attn:1", ...
    component_labels = []
    for module_key, layer in zip(module_keys, layers_true):
        n_components = layer.shape[1]
        component_labels.extend([f"{module_key}:{i}" for i in range(n_components)])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot
    ax1.imshow(A_true.T, aspect="auto", interpolation="nearest", cmap="Blues")
    ax2.imshow(A_pred.T, aspect="auto", interpolation="nearest", cmap="Reds")

    # Add layer boundaries (adapt from spd/clustering/plotting/activations.py)
    add_component_labeling(ax1, component_labels, axis='y')
    add_component_labeling(ax2, component_labels, axis='y')

    # Titles
    ax1.set_title(r"True Binary Activations (Unsorted)\n" +
                  r"$A_{ij} = \mathbb{1}[\text{act}_{ij} > \theta]$")
    ax2.set_title(r"Predicted Binary Activations (Unsorted)\n" +
                  r"$\hat{A}_{ij} = \mathbb{1}[P(A_{ij}=1) > 0.5]$")
```

#### 1.5.2 Sorted Version (New)

**Sorting Strategy:**

1. **Sample Sorting (Greedy):**
   - Compute sample similarity matrix (cosine similarity on true activations)
   - Greedy ordering: start from most central sample, add nearest neighbor iteratively
   - Apply **same ordering** to predicted activations (so we can compare)
   - Reference implementation already exists in `spd/clustering/plotting/activations.py:120-162`

2. **Component Sorting (Greedy):**
   - Compute component similarity matrix (cosine similarity on true activations)
   - Same greedy algorithm but on columns instead of rows
   - Apply same ordering to both true and predicted

**Three Subplots:**
1. True activations (samples sorted, components sorted)
2. Predicted activations (same ordering)
3. **Diff plot:** `predicted - true` with RdBu colormap
   - Red = False Positive (predicted 1, true 0)
   - Blue = False Negative (predicted 0, true 1)
   - White = Correct

**Implementation:**
```python
def plot_activations_sorted(
    layers_true: list[np.ndarray],
    layers_pred: list[np.ndarray],
    module_keys: list[str],
) -> None:
    """Show sorted activations with diff plot."""
    A_true = np.concatenate(layers_true, axis=1).astype(float)
    A_pred = np.concatenate(layers_pred, axis=1).astype(float)

    # Sort samples (greedy on rows)
    sample_order = greedy_sort(A_true, axis=0)  # Returns indices
    A_true_sorted_samples = A_true[sample_order, :]
    A_pred_sorted_samples = A_pred[sample_order, :]

    # Sort components (greedy on columns)
    component_order = greedy_sort(A_true_sorted_samples, axis=1)
    A_true_sorted = A_true_sorted_samples[:, component_order]
    A_pred_sorted = A_pred_sorted_samples[:, component_order]

    # Diff
    A_diff = A_pred_sorted - A_true_sorted  # Range: [-1, 0, 1]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    ax1.imshow(A_true_sorted.T, aspect="auto", interpolation="nearest", cmap="Blues")
    ax1.set_title(r"True Activations (Sorted)\n" +
                  r"Samples and components sorted by similarity")

    ax2.imshow(A_pred_sorted.T, aspect="auto", interpolation="nearest", cmap="Reds")
    ax2.set_title(r"Predicted Activations (Sorted)\n" +
                  r"Same ordering as true activations")

    # Diff plot with centered colormap
    im3 = ax3.imshow(A_diff.T, aspect="auto", interpolation="nearest",
                     cmap="RdBu_r", vmin=-1, vmax=1)
    ax3.set_title(r"Prediction Errors (Predicted - True)\n" +
                  r"Red = FP ($\hat{A}=1, A=0$), Blue = FN ($\hat{A}=0, A=1$), White = Correct")
    plt.colorbar(im3, ax=ax3)

    fig.tight_layout()
```

**Helper Function:**
```python
def greedy_sort(A: np.ndarray, axis: int) -> np.ndarray:
    """Greedy ordering by similarity.

    Args:
        A: 2D array
        axis: 0 for rows, 1 for columns

    Returns:
        Indices in sorted order
    """
    # Transpose if sorting columns
    if axis == 1:
        A = A.T

    # Compute cosine similarity
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    A_normalized = A / norms
    similarity = A_normalized @ A_normalized.T

    # Greedy ordering (same as in activations.py)
    n = similarity.shape[0]
    avg_sim = similarity.mean(axis=1)
    start_idx = int(np.argmax(avg_sim))

    ordered = [start_idx]
    remaining = set(range(n))
    remaining.remove(start_idx)
    current = start_idx

    while remaining:
        sims = [(i, similarity[current, i]) for i in remaining]
        best_idx = max(sims, key=lambda x: x[1])[0]
        ordered.append(best_idx)
        remaining.remove(best_idx)
        current = best_idx

    return np.array(ordered)
```

---

### 1.6 Covariance Matrix - Sorted Version

**Current:** Single unsorted covariance plot

**New:** Two versions
1. **Unsorted** with layer boundaries (like activations unsorted)
2. **Sorted** using same component ordering from activations

**Implementation:**
```python
def plot_covariance_unsorted(
    layers_true: list[np.ndarray],
    module_keys: list[str],
) -> None:
    """Covariance with layer boundaries."""
    A = np.concatenate(layers_true, axis=1).astype(float)
    C = np.cov(A, rowvar=False)

    component_labels = [...]  # Same as activations

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(C, aspect="auto", interpolation="nearest", cmap="RdBu_r")

    # Add layer boundaries on both axes
    add_component_labeling(ax, component_labels, axis='x')
    add_component_labeling(ax, component_labels, axis='y')

    ax.set_title(r"Component Covariance Matrix (Unsorted)\n" +
                 r"$\text{Cov}(i,j) = \mathbb{E}[(A_i - \mu_i)(A_j - \mu_j)]$")
    plt.colorbar(im)

def plot_covariance_sorted(
    layers_true: list[np.ndarray],
    component_order: np.ndarray,  # Pass in from activations
) -> None:
    """Covariance with sorted components."""
    A = np.concatenate(layers_true, axis=1).astype(float)
    A_sorted = A[:, component_order]
    C_sorted = np.cov(A_sorted, rowvar=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(C_sorted, aspect="auto", interpolation="nearest", cmap="RdBu_r")
    ax.set_title(r"Component Covariance Matrix (Sorted)\n" +
                 r"Components ordered by similarity")
    plt.colorbar(im)
```

---

## Part 2: Interactive Tree Visualization (HTML/JS)

### 2.1 High-Level Architecture

**Export:** Python creates one JSON per tree → **Display:** HTML/JS loads JSON and renders visualizations

### 2.2 Data to Export (per tree)

#### Tree Metadata
```json
{
  "layer_index": 1,
  "target_component_idx": 5,
  "module_key": "blocks.0.mlp.W_gate",
  "metrics": {
    "ap": 0.85,
    "accuracy": 0.92,
    "balanced_accuracy": 0.88,
    "prevalence": 0.023,
    "n_samples": 200,
    "n_positive": 46,
    "n_negative": 154,
    "confusion_matrix": {
      "TP": 40,
      "TN": 144,
      "FP": 10,
      "FN": 6
    }
  },
  "tree_stats": {
    "max_depth": 5,
    "n_leaves": 12,
    "n_nodes": 23
  }
}
```

#### Tree Structure
```json
{
  "structure": {
    "children_left": [1, -1, 3, 4, -1, ...],
    "children_right": [2, -1, 5, 6, -1, ...],
    "feature": [7, -2, 12, 3, -2, ...],
    "threshold": [0.5, -2, 0.5, 0.5, -2, ...],
    "value": [[30, 20], [5, 15], ...],  // [n_negative, n_positive] per node
    "n_node_samples": [200, 50, 150, ...]
  },
  "feature_names": [
    "blocks.0.attn.W_Q:3 (prev=0.15, AP=0.82)",
    "blocks.0.attn.W_Q:17 (prev=0.08, AP=0.91)",
    "blocks.0.mlp.W_in:5 (prev=0.23, AP=0.76)",
    ...
  ]
}
```

#### Activation Histograms
```json
{
  "true_activations": {
    "histogram": {
      "bins": [0.0, 0.01, 0.02, ...],  // Bin edges
      "counts": [120, 45, 23, ...]
    }
  },
  "predicted_probabilities": {
    "histogram": {
      "bins": [0.0, 0.1, 0.2, ...],
      "counts": [80, 30, 40, ...]
    }
  }
}
```

#### Token-Level Samples

**Sample Selection Strategies:**

1. **Stratified by confusion matrix** (recommended):
   - 2 True Positives (high confidence, low confidence)
   - 2 True Negatives (high confidence, low confidence)
   - 2 False Positives (worst errors)
   - 2 False Negatives (worst errors)
   - Total: 8 samples

2. **Fallback if categories insufficient:**
   - Random samples from each category
   - Fill missing categories with "N/A"

**Data Structure:**
```json
{
  "samples": [
    {
      "sample_idx": 42,
      "category": "TP",  // TP, TN, FP, or FN
      "confidence": 0.95,  // abs(predicted_prob - 0.5)
      "tokens": ["The", "cat", "sat", "on", "the", "mat"],
      "true_activations": [0.0, 0.0, 0.82, 0.91, 0.0, 0.0],  // Continuous values
      "predicted_probabilities": [0.05, 0.1, 0.88, 0.94, 0.02, 0.01],
      "true_binary": [0, 0, 1, 1, 0, 0],
      "predicted_binary": [0, 0, 1, 1, 0, 0],
      "max_true_pos": 2,  // Index of max activation in true
      "max_pred_pos": 3   // Index of max activation in predicted
    },
    // ... 7 more samples
  ]
}
```

#### Input Features Summary
```json
{
  "input_features_by_module": {
    "blocks.0.attn.W_Q": [3, 17, 42],  // Component indices used in tree
    "blocks.0.mlp.W_in": [5, 12]
  },
  "n_input_features_total": 5,
  "n_components_total": 256  // All components in layer 0
}
```

### 2.3 Python Export Implementation

**New File:** `spd/clustering/ci_dt/export.py`

```python
"""Export decision tree data to JSON for interactive visualization."""

from pathlib import Path
from typing import Any
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from spd.clustering.ci_dt.core import LayerModel


def export_tree_json(
    tree: DecisionTreeClassifier,
    layer_idx: int,
    target_idx: int,
    module_key: str,
    X: np.ndarray,  # Input features (all layer 0 components)
    Y_true: np.ndarray,  # True binary activations for this target
    Y_prob: np.ndarray,  # Predicted probabilities
    tokens_batch: list[list[str]],  # Decoded tokens for all samples
    feature_names: list[str],
    output_path: Path,
) -> None:
    """Export single tree to JSON."""

    # 1. Compute metrics
    Y_pred = (Y_prob >= 0.5).astype(int)
    metrics = compute_tree_metrics(Y_true, Y_pred, Y_prob)

    # 2. Serialize tree structure
    tree_dict = serialize_tree(tree, feature_names)

    # 3. Create activation histograms
    histograms = create_histograms(Y_true, Y_prob)

    # 4. Select and export token samples
    samples = select_token_samples(
        Y_true, Y_prob, Y_pred, tokens_batch
    )

    # 5. Identify which input features are used
    input_features = extract_input_features(tree, module_key)

    # 6. Combine into single JSON
    data = {
        "metadata": {
            "layer_index": layer_idx,
            "target_component_idx": target_idx,
            "module_key": module_key,
            "metrics": metrics,
            "tree_stats": {
                "max_depth": int(tree.tree_.max_depth),
                "n_leaves": int(tree.tree_.n_leaves),
                "n_nodes": int(tree.tree_.node_count),
            }
        },
        "tree": tree_dict,
        "histograms": histograms,
        "samples": samples,
        "input_features": input_features,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def export_all_trees(
    models: list[LayerModel],
    layers_true: list[np.ndarray],
    per_layer_stats: list[dict],
    component_acts: dict[str, Tensor],  # Original activations (continuous)
    batch_data: dict,  # From dataloader (has token IDs)
    tokenizer,  # HuggingFace tokenizer
    feature_names: list[list[str]],
    output_dir: Path,
) -> None:
    """Export all trees and create index."""

    # Decode all tokens once
    tokens_batch = decode_all_tokens(batch_data, tokenizer)

    # Export each tree
    tree_index = []
    for layer_idx, model in enumerate(models):
        module_key = list(component_acts.keys())[layer_idx]
        X = layers_true[0]  # Always predict from layer 0
        Y_true = layers_true[layer_idx + 1]  # Target layer

        for target_idx, estimator in enumerate(model.model.estimators_):
            # Get predictions for this target
            Y_prob = estimator.predict_proba(X)[:, 1]

            # Get feature names for this layer's inputs
            feat_names = feature_names[layer_idx] if feature_names else None

            # Export
            tree_path = output_dir / "data" / f"tree_{layer_idx}_{target_idx}.json"
            export_tree_json(
                tree=estimator,
                layer_idx=layer_idx,
                target_idx=target_idx,
                module_key=module_key,
                X=X,
                Y_true=Y_true[:, target_idx],
                Y_prob=Y_prob,
                tokens_batch=tokens_batch,
                feature_names=feat_names,
                output_path=tree_path,
            )

            # Add to index
            tree_index.append({
                "layer": layer_idx,
                "target": target_idx,
                "module_key": module_key,
                "ap": per_layer_stats[layer_idx]["ap"][target_idx],
                "file": f"data/tree_{layer_idx}_{target_idx}.json"
            })

    # Write index
    index_path = output_dir / "data" / "index.json"
    with open(index_path, 'w') as f:
        json.dump(tree_index, f, indent=2)


def select_token_samples(
    Y_true: np.ndarray,
    Y_prob: np.ndarray,
    Y_pred: np.ndarray,
    tokens_batch: list[list[str]],
    n_per_category: int = 2,
) -> list[dict]:
    """Select stratified samples from confusion matrix categories."""

    # Categorize samples
    TP_mask = (Y_true == 1) & (Y_pred == 1)
    TN_mask = (Y_true == 0) & (Y_pred == 0)
    FP_mask = (Y_true == 0) & (Y_pred == 1)
    FN_mask = (Y_true == 1) & (Y_pred == 0)

    # Confidence = distance from decision boundary
    confidence = np.abs(Y_prob - 0.5)

    samples = []

    for category, mask in [("TP", TP_mask), ("TN", TN_mask),
                           ("FP", FP_mask), ("FN", FN_mask)]:
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        # Sort by confidence
        sorted_indices = indices[np.argsort(confidence[indices])[::-1]]

        # Take high and low confidence
        n_take = min(n_per_category, len(sorted_indices))
        if n_take == 2:
            selected = [sorted_indices[0], sorted_indices[-1]]  # High and low
        else:
            selected = sorted_indices[:n_take]

        for idx in selected:
            samples.append({
                "sample_idx": int(idx),
                "category": category,
                "confidence": float(confidence[idx]),
                "tokens": tokens_batch[idx],
                "true_activations": Y_true[idx].tolist(),  # Would need continuous version
                "predicted_probabilities": [float(Y_prob[idx])] * len(tokens_batch[idx]),
                "true_binary": int(Y_true[idx]),
                "predicted_binary": int(Y_pred[idx]),
            })

    return samples
```

**Integration in `run.py`:**

```python
# After computing metrics (line ~121)
from spd.clustering.ci_dt.export import export_all_trees

export_output_dir = Path("./ci_dt_vis")
export_all_trees(
    models=models,
    layers_true=layers_true,
    per_layer_stats=per_layer_stats,
    component_acts=component_acts_concat,
    batch_data=next(iter(dataloader)),  # Need to save earlier
    tokenizer=cfg.task_config.tokenizer,
    feature_names=feature_names,
    output_dir=export_output_dir,
)
print(f"Exported tree visualizations to {export_output_dir}")
```

### 2.4 HTML/JS Viewer Implementation

**File Structure:**
```
ci_dt_vis/
├── index.html              # Main viewer
├── data/
│   ├── index.json          # Tree index
│   ├── tree_1_0.json       # Individual trees
│   ├── tree_1_1.json
│   └── ...
├── js/
│   ├── viewer.js           # Main app logic
│   ├── tree-display.js     # Tree visualization
│   ├── token-display.js    # Token highlighting
│   └── sparklines.js       # Histograms
└── css/
    └── style.css
```

**`index.html`:**
```html
<!DOCTYPE html>
<html>
<head>
    <title>CI Decision Tree Viewer</title>
    <link rel="stylesheet" href="css/style.css">
    <script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body>
    <div id="app">
        <div id="tree-selector">
            <h2>Select Tree</h2>
            <label>Layer: <select id="layer-select"></select></label>
            <label>Target: <select id="target-select"></select></label>
        </div>

        <div id="tree-info">
            <h2>Tree Metrics</h2>
            <div id="metrics"></div>
        </div>

        <div id="tree-viz">
            <h2>Decision Tree Structure</h2>
            <div id="tree-svg"></div>
        </div>

        <div id="activation-histograms">
            <h2>Activation Distributions</h2>
            <canvas id="hist-canvas"></canvas>
        </div>

        <div id="token-samples">
            <h2>Example Samples</h2>
            <div id="samples-container"></div>
        </div>
    </div>

    <script src="js/sparklines.js"></script>
    <script src="js/token-display.js"></script>
    <script src="js/tree-display.js"></script>
    <script src="js/viewer.js"></script>
</body>
</html>
```

**`js/viewer.js`:**
```javascript
// Main viewer logic
let treeIndex = [];
let currentTree = null;

async function init() {
    // Load tree index
    const response = await fetch('data/index.json');
    treeIndex = await response.json();

    // Populate layer selector
    const layers = [...new Set(treeIndex.map(t => t.layer))];
    const layerSelect = document.getElementById('layer-select');
    layers.forEach(layer => {
        const option = document.createElement('option');
        option.value = layer;
        option.text = `Layer ${layer}`;
        layerSelect.appendChild(option);
    });

    // Event listeners
    layerSelect.addEventListener('change', onLayerChange);
    document.getElementById('target-select').addEventListener('change', onTargetChange);

    // Load first tree
    if (treeIndex.length > 0) {
        await loadTree(treeIndex[0].layer, treeIndex[0].target);
    }
}

function onLayerChange() {
    const layer = parseInt(document.getElementById('layer-select').value);
    const trees = treeIndex.filter(t => t.layer === layer);

    const targetSelect = document.getElementById('target-select');
    targetSelect.innerHTML = '';
    trees.forEach(tree => {
        const option = document.createElement('option');
        option.value = tree.target;
        option.text = `Target ${tree.target} (AP=${tree.ap.toFixed(3)})`;
        targetSelect.appendChild(option);
    });

    if (trees.length > 0) {
        loadTree(layer, trees[0].target);
    }
}

async function loadTree(layer, target) {
    const response = await fetch(`data/tree_${layer}_${target}.json`);
    currentTree = await response.json();

    displayMetrics(currentTree.metadata);
    displayHistograms(currentTree.histograms);
    displayTree(currentTree.tree);
    displayTokenSamples(currentTree.samples);
}

function displayMetrics(metadata) {
    const m = metadata.metrics;
    const cm = m.confusion_matrix;

    const html = `
        <table>
            <tr><td>AP:</td><td>${m.ap.toFixed(3)}</td></tr>
            <tr><td>Accuracy:</td><td>${m.accuracy.toFixed(3)}</td></tr>
            <tr><td>Balanced Acc:</td><td>${m.balanced_accuracy.toFixed(3)}</td></tr>
            <tr><td>Prevalence:</td><td>${m.prevalence.toFixed(4)}</td></tr>
            <tr><td colspan="2"><strong>Confusion Matrix:</strong></td></tr>
            <tr><td>TP:</td><td>${cm.TP}</td></tr>
            <tr><td>TN:</td><td>${cm.TN}</td></tr>
            <tr><td>FP:</td><td>${cm.FP}</td></tr>
            <tr><td>FN:</td><td>${cm.FN}</td></tr>
        </table>
    `;
    document.getElementById('metrics').innerHTML = html;
}

function displayHistograms(histograms) {
    // Use sparklines.js to render dual histograms
    const canvas = document.getElementById('hist-canvas');
    const ctx = canvas.getContext('2d');

    // Draw true activations (blue) and predicted (red) overlaid
    drawHistogram(ctx, histograms.true_activations, 'blue', 0);
    drawHistogram(ctx, histograms.predicted_probabilities, 'red', 0);
}

function displayTree(treeData) {
    // Use tree-display.js to render D3 tree
    renderDecisionTree('tree-svg', treeData);
}

function displayTokenSamples(samples) {
    const container = document.getElementById('samples-container');
    container.innerHTML = '';

    samples.forEach(sample => {
        const div = document.createElement('div');
        div.className = `sample sample-${sample.category}`;
        div.innerHTML = `
            <h3>${sample.category} (confidence: ${sample.confidence.toFixed(3)})</h3>
            <div class="tokens">${renderTokens(sample)}</div>
        `;
        container.appendChild(div);
    });
}

function renderTokens(sample) {
    // Create dual-color token visualization
    // Blue background = true activation, Red = predicted
    return sample.tokens.map((token, i) => {
        const trueVal = sample.true_activations[i];
        const predVal = sample.predicted_probabilities[i];

        // Dual gradient or side-by-side bars
        return `<span class="token"
                      style="--true-val: ${trueVal}; --pred-val: ${predVal}">
                    ${token}
                </span>`;
    }).join(' ');
}

// Initialize on load
init();
```

**`js/tree-display.js`:**
```javascript
function renderDecisionTree(containerId, treeData) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    // Simple text-based tree for now
    // Can upgrade to D3.js interactive tree later

    const textTree = buildTextTree(treeData.structure, treeData.feature_names);
    const pre = document.createElement('pre');
    pre.textContent = textTree;
    container.appendChild(pre);
}

function buildTextTree(structure, featureNames, nodeIdx = 0, depth = 0) {
    const indent = '  '.repeat(depth);

    if (structure.children_left[nodeIdx] === -1) {
        // Leaf node
        const value = structure.value[nodeIdx];
        const prediction = value[1] > value[0] ? 'ACTIVE' : 'INACTIVE';
        return `${indent}→ ${prediction} (${value[0]}/${value[1]})\n`;
    }

    // Internal node
    const feature = structure.feature[nodeIdx];
    const threshold = structure.threshold[nodeIdx];
    const featureName = featureNames[feature];

    let result = `${indent}${featureName} <= ${threshold}?\n`;
    result += buildTextTree(structure, featureNames, structure.children_left[nodeIdx], depth + 1);
    result += `${indent}else:\n`;
    result += buildTextTree(structure, featureNames, structure.children_right[nodeIdx], depth + 1);

    return result;
}
```

---

## Implementation Checklist

### Phase 1: Static Plot Improvements
- [ ] Update `plot_layer_metrics()`: scatter with jitter instead of bars
- [ ] Add LaTeX titles to all metrics plots (TP/FP/TN/FN formulas)
- [ ] Update AP vs prevalence: log scale, no edges, color by depth
- [ ] Add AP vs prevalence heatmap to `plot_tree_statistics()`
- [ ] Implement `greedy_sort()` helper function
- [ ] Create `plot_activations_unsorted()` with layer boundaries
- [ ] Create `plot_activations_sorted()` with diff plot
- [ ] Create `plot_covariance_unsorted()` with layer boundaries
- [ ] Create `plot_covariance_sorted()`
- [ ] Update all plot titles with LaTeX and newlines
- [ ] Test with existing `run.py` workflow

### Phase 2: Data Export
- [ ] Create `spd/clustering/ci_dt/export.py`
- [ ] Implement `export_tree_json()`
- [ ] Implement `export_all_trees()`
- [ ] Implement `select_token_samples()` with stratified sampling
- [ ] Implement `serialize_tree()`, `compute_tree_metrics()`, etc.
- [ ] Add export call to `run.py`
- [ ] Test JSON output schema

### Phase 3: Interactive Viewer
- [ ] Create `ci_dt_vis/` directory structure
- [ ] Implement `index.html` layout
- [ ] Implement `viewer.js` tree selection and loading
- [ ] Implement `tree-display.js` text rendering (D3 optional)
- [ ] Implement `token-display.js` dual-color visualization
- [ ] Implement histogram rendering (reuse or adapt sparklines.js)
- [ ] Add CSS styling
- [ ] Test end-to-end workflow

### Phase 4: Documentation
- [ ] Update `run.py` docstrings
- [ ] Add README in `ci_dt_vis/` explaining viewer usage
- [ ] Document JSON schema
- [ ] Add example screenshots

---

## Open Questions / Design Decisions

1. **Token samples per tree:** 8 total (2 per category) seems reasonable. Too many?
2. **Histogram bins:** 50 bins for activations, 20 for probabilities?
3. **D3.js tree or text?** Start with text, add D3 if needed
4. **Component sorting:** Should we also show a version with components sorted by layer, then by similarity within layer?
5. **File size:** Each tree JSON might be 50-200KB. With 1000s of trees, total size could be 50-200MB. Acceptable?
6. **Continuous activations for tokens:** Currently we only have binary. Need to save continuous pre-threshold values?

---

## Success Metrics

**Static Plots:**
- Plots are immediately interpretable without prior knowledge
- Titles explain abbreviations and formulas
- Layer boundaries visible in unsorted plots
- Sorting reveals structure (coactivation patterns)
- Diff plot clearly shows FP/FN errors

**Interactive Viewer:**
- Can load and view any tree in <1 second
- Token examples clearly show where component activates
- Confusion matrix category examples are informative
- Tree structure is readable
- Histograms show activation distributions clearly
