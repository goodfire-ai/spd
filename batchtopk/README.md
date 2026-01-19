# BatchTopK Sparse Autoencoders

BatchTopK is a novel approach to Sparse Autoencoders (SAEs) that offers an alternative to TopK SAEs as introduced by OpenAI. This repository contains the implementation and experiments for BatchTopK SAEs, as described in our preliminary findings.

### What is the BatchTopK activation function?

BatchTopK modifies the TopK activation in SAEs in the following way:

1. Instead of applying TopK to each sample independently, it flattens all feature activations across the batch.
2. It then takes the top (K * batch_size) activations.
3. Finally, it reshapes the result back to the original batch shape.

## Usage

```bash
git clone https://github.com/bartbussmann/BatchTopK.git
cd BatchTopK
pip install transformer_lens
python main.py
```

## Acknowledgments
The training code is heavily inspired and basically a stripped-down version of [SAELens](https://github.com/jbloomAus/SAELens). Thanks to the SAELens team for their foundational work in this area!
