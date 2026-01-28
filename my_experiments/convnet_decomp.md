# Convolutional Layer Decomposition in SPD

This document explains how we extend Stochastic Parameter Decomposition (SPD) to convolutional layers.


## Convolutions as Linear Operations

A 2D convolution can be understood as a linear operation applied at each spatial location. Consider a conv layer with:
- Input: (batch, in_channels, H, W)
- Kernel size: (kH, kW)
- Output channels: out_channels

At each spatial position, the convolution extracts a patch of size (in_channels, kH, kW), flattens it to a vector of size (in_channels * kH * kW), and applies a linear transformation to produce out_channels outputs.

So conceptually, the conv weight of shape (out_channels, in_channels, kH, kW) can be viewed as a matrix of shape (out_channels, in_channels * kH * kW) that operates on flattened patches.

## The Decomposition

We decompose this "flattened" view of the convolution weight:

```
W_flat = V @ U
```

Where:
- V has shape (in_channels * kH * kW, C) - maps input patches to C component activations
- U has shape (C, out_channels) - maps component activations to output channels

The interpretation:
- **V** learns to detect features in input patches and produce a component activation for each
- **U** learns how to combine these component activations into the final output channels

## Efficient Implementation

Rather than explicitly extracting patches and doing matrix multiplication (which would be slow), we implement this using native convolution operations:

### Step 1: Compute component activations via convolution

We reshape V from (in_channels * kH * kW, C) into C convolutional filters of shape (C, in_channels, kH, kW). Applying these filters to the input gives us component activations at each spatial location:

```
component_acts = conv2d(input, V_as_filters)
# Result: (batch, C, H_out, W_out)
```

This is mathematically equivalent to: for each spatial location, extract the patch, flatten it, and multiply by V.

### Step 2: Apply U via 1x1 convolution

We reshape U from (C, out_channels) into 1x1 convolutional filters of shape (out_channels, C, 1, 1). Applying this to the component activations:

```
output = conv2d(component_acts, U_as_1x1)
# Result: (batch, out_channels, H_out, W_out)
```

A 1x1 convolution is equivalent to applying a linear transformation independently at each spatial location - exactly what we need to map the C component activations to out_channels outputs.

## Stochastic Masking

During SPD training, we apply stochastic masks to the component activations to encourage sparse usage:

```
masked_component_acts = component_acts * mask
```

The mask can be:
- Per-sample: shape (batch, C) - same mask at all spatial locations
- Per-location: shape (batch, H_out, W_out, C) - different mask at each position

This allows us to analyze which components are causally important for the model's predictions.

## Reconstructing the Original Weight

To verify the decomposition or to use the decomposed model without masking, we can reconstruct the original conv weight:

```
W_reconstructed = (V @ U).T.reshape(out_channels, in_channels, kH, kW)
```

## Summary

The convolution decomposition works by:
1. Viewing the conv weight as a linear transformation on flattened patches
2. Factorizing this into V (patch features to components) and U (components to outputs)
3. Implementing efficiently using native conv2d operations
4. Enabling the same component analysis as linear layers through stochastic masking

This allows us to analyze convolutional networks using the same SPD framework we use for fully connected networks.
