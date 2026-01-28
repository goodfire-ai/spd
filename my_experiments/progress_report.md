# Progress Report - Week of Jan 17-24, 2026

## Summary

This week focused on deepening my understanding of component relationships in decomposed networks and exploring alternative sparse representation learning approaches. The main accomplishments were:

1. **Causal analysis of component relationships**: Developed a comprehensive analysis pipeline to understand how input components (fc1) causally influence output components (fc2) in a decomposed MNIST MLP. Key finding: the network uses surprisingly sparse connectivity - just 5-9 input components account for 80% of the causal effect on each output component.

2. **SAE architecture explorations**: Implemented and experimented with several SAE variants including Stochastic SAE (with SPD-style CI functions), AbsTopKSAE (allowing negative activations), JumpReLU SAE (Anthropic's approach), and Multi-Input SAE (using context from multiple locations).

3. **Transcoder implementation**: Built a complete transcoder training pipeline based on BatchTopK to imitate MLP layers, supporting multiple activation types (relu, topk, batchtopk). This is groundwork for comparing what transcoders find versus what SPD finds.

4. **Shapes/Colors dataset**: Created a synthetic dataset with multiple independent attributes (shape, color, size) to test whether SPD can discover atomic features that combine compositionally.

5. **Conv2d decomposition exploration**: Attempted to decompose a full CNN (conv + fc layers) using SPD. Discovered that while SPD's core algorithm supports Conv2d via `Conv2dComponents`, the metrics/visualization pipeline needs updates to handle spatial dimensions.

## Main Experiments

### 1. Causal Analysis of Component Relationships

Applied a series of increasingly sophisticated analyses to understand how the ~28 alive input components relate to the ~10 alive output components in a decomposed MNIST MLP:

**Methods explored:**
- **Lasso regression**: Found top-2 predictive inputs per output (R² = 0.6-0.87)
- **Stepwise forward regression**: Greedy feature selection showing ~80% R² ceiling even with 10 features
- **Weight path analysis**: Computed U1 @ V2 to see structural connectivity
- **Causal ablation**: For each (i,j) pair, measured E[output_j | input_i ON] - E[output_j | input_i OFF], filtered to samples where CI(j) > 0.1

**Key findings:**
- At 80% threshold, most outputs only need 5-9 input components
- Strong excitatory effects: C369 → C89 (+3.13), C289 → C423 (+2.56)
- Strong inhibitory effects: C2 → C61 (-3.78), C398 → C435 (-2.04)
- Interestingly, some weight-based predictions don't match causal effects (sign flips), suggesting ReLU gating creates complex interactions

![Causal Graph](output/mnist_experiment_v2/causal_graph_sparse.png)

### 2. SAE Architecture Explorations

Implemented several SAE variants to compare against SPD:

**Stochastic SAE** (`stochastic_sae.py`)
- Uses SPD-style CI functions to predict feature importance
- Stochastic masking based on CI values
- Key learning: Must use SPD's gradient trick (noise-based differentiable sampling) rather than `torch.bernoulli()` which kills gradients

**AbsTopKSAE** (`abs_topk_sae.py`)
- TopK on absolute values instead of post-ReLU
- Allows negative feature activations
- Per-sample K sampling (log-uniform distribution) for varied sparsity

**JumpReLU SAE** (`anthropic_jumprelu_sae.py`)
- Implementation of Anthropic's approach
- Learnable threshold per feature
- Added BatchTopK comparison mode

**Multi-Input SAE** (`multi_input_sae.py`)
- Encoder receives inputs from multiple locations
- Tests whether cross-layer context improves feature discovery

### 3. Transcoder Implementation

Built a complete transcoder training pipeline (`mnist_transcoder.py`) based on [BatchTopK](https://github.com/bartbussmann/BatchTopK):

**Architecture:**
- Encoder: d_input (784) → n_features (sparse hidden)
- Decoder: n_features → d_output (128, matching MLP hidden size)
- Supports three activation types: `relu`, `topk`, `batchtopk`

**Key features:**
- Input unit normalization (optional)
- Dead feature revival via auxiliary loss
- Decoder weight normalization to unit norm
- Gradient projection orthogonal to decoder weights

**Training setup:**
- Reconstruction loss: L2 on normalized activations
- L1 sparsity penalty (for relu mode)
- Auxiliary loss for dead feature revival
- Visualization of feature directions and digit-feature heatmaps

This is groundwork for comparing what transcoders find versus what SPD finds on the same model.

### 4. Shapes/Colors Dataset

Created `shapes_colours_dataset.py` - a synthetic dataset where each image has:
- Shape (circle, square, triangle)
- Color (red, green, blue)
- Size (small, medium, large)

The goal is to test whether SPD discovers 9 atomic features (3 shapes + 3 colors + 3 sizes) that combine compositionally, rather than 27 features for each combination. Built interactive Gradio dashboard for visualization.

### 5. SPD Enhancements for MNIST

Added several improvements to the MNIST experiment:
- **Weight decay/L1/L2 regularization**: Custom loss terms on V and U matrices
- **Component directions visualization**: Shows top-N most active components as 28x28 images during training
- **CI heatmap improvements**: Raw mean CI values instead of normalized

### 6. Conv2d Decomposition (Attempted)

Explored decomposing convolutional layers in SPD for the shapes/colors CNN:

**What I tried:**
- Created `ShapesCNNWrapper` to wrap the full CNN (3 conv + 2 fc layers) with single tensor output
- Modified `shapes_decomposition.py` to pass raw images as input (not just MLP features)
- Added conv1, conv2, conv3 to the `module_info` alongside fc1, fc2

**What SPD already has:**
SPD has a complete `Conv2dComponents` class (`spd/models/components.py:322`) that:
- Treats convolution as a linear transformation at each spatial location
- V: (in_channels * kH * kW, C) maps input patches to components
- U: (C, out_channels) maps components to output channels
- Uses `F.conv2d` internally for efficient CUDA-optimized computation
- Supports per-location masks: (batch, H_out, W_out, C)

**What broke:**
Several evaluation metrics and visualizations crashed with conv layers:
- `CIHistogramsConfig` - expects flat component activations
- `ComponentActivationDensityConfig` - spatial dimension mismatch
- `UVPlotsConfig` - designed for 2D weight matrices
- `StochasticAccuracyLayerwiseConfig` - layer output shape assumptions

**Current status:**
Reverted to fc-only decomposition for now. The core SPD algorithm does support Conv2d, but the metrics/visualization pipeline needs updates to handle spatial dimensions properly. This is a good candidate for a future PR to make SPD fully support CNN decomposition.

## Open Questions / Next Steps

1. **Why do causal effects sometimes contradict weight predictions?** The sign flips between U1@V2 weights and measured causal effects suggest complex interactions through the ReLU. Worth investigating whether this is meaningful or a measurement artifact.

2. **Transcoder vs SPD comparison**: Now that the transcoder is implemented, systematically compare what it finds versus what SPD finds on the same MNIST MLP. Do they discover the same features?

3. **Shapes/Colors results**: Need to run full SPD decomposition and analyze whether it finds atomic vs. compositional features.

4. **Transcoder + stochastic ablation**: Could training a transcoder with a stochastic ablation loss (like SPD's importance minimality loss) solve feature splitting?

5. **Pre-decomposed vs. post-hoc**: Why do these give different decompositions? Which is more interpretable or useful?

6. **Conv2d metric support**: The core SPD algorithm supports Conv2d, but metrics/visualizations need updates. Could contribute a PR to make the full pipeline work with CNNs.

7. **Project proposal**: Crystallize research questions into a focused proposal.

## Files Created This Week

```
my_experiments/
├── analyze_feature_relationships.py  # Causal analysis pipeline
├── stochastic_sae.py                 # SPD-style SAE
├── abs_topk_sae.py                   # AbsTopK SAE variant
├── anthropic_jumprelu_sae.py         # JumpReLU implementation
├── multi_input_sae.py                # Multi-location encoder
├── shapes_colours_dataset.py         # Synthetic dataset
├── shapes_decomposition.py           # SPD on shapes
├── shapes_component_dashboard.py     # Gradio visualization
├── mnist_transcoder.py               # Transcoder implementation
└── mnist_experiment.py               # Enhanced with L1/L2, visualizations
```

## Visualizations Generated

- `causal_effect_matrix.png` - Full causal effect heatmap
- `causal_effect_matrix_sparse.png` - Sparse version (80% threshold)
- `causal_graph_sparse.png` - Bipartite graph with minimized crossings
- `weight_path_matrix.png` - U1 @ V2 structural connectivity
- `ci_correlation_matrix.png` - CI score correlations
- `feature_connectivity_matrix.png` - Lasso-based connectivity
- `stepwise_regression_results.png` - Stepwise R² curves
- `transcoder_features.png` - Transcoder feature directions as 28x28 images
- `transcoder_digit_heatmap.png` - Feature activations by digit class
