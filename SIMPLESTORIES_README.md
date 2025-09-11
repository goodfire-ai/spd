# SimpleStories Model Scripts

This directory contains scripts to interact with the SimpleStories model for text generation.

## Scripts Available

### 1. `simplestories_demo.py` (Recommended)
A comprehensive demo script with the following features:
- Automatic CUDA/CPU detection
- Error handling and troubleshooting tips
- Interactive mode for custom prompts
- Detailed logging and progress indicators
- Model parameter count display

**Usage:**
```bash
python simplestories_demo.py
```

### 2. `simplestories_simple.py`
A minimal script that closely follows the original code with basic error handling:
- Simple and straightforward
- Minimal modifications to the original code
- Basic device detection (CUDA/CPU)

**Usage:**
```bash
python simplestories_simple.py
```

## Requirements

The scripts require the following dependencies, which are already included in this project:
- `torch` (>=2.6)
- `transformers`

## Model Information

- **Model**: SimpleStories-1.25M
- **Parameters**: ~1.25 million
- **Model Type**: LlamaForCausalLM
- **Source**: Hugging Face Hub (`SimpleStories/SimpleStories-1.25M`)

## Features

### Automatic Device Detection
- The scripts automatically detect if CUDA is available
- Falls back to CPU if CUDA is not available
- Displays device information and CUDA version

### Text Generation
- **Default prompt**: "The curious cat looked at the"
- **Max tokens**: 400 (demo) / 200 (interactive)
- **Temperature**: 0.7
- **Sampling**: Enabled with EOS token handling

### Error Handling
- Model loading error handling
- Generation error handling
- Helpful troubleshooting messages
- Graceful fallbacks
- Proper attention mask and pad token handling (no warnings)

## Example Output

```
============================================================
SimpleStories Model Demo
============================================================
CUDA is available! Using device: NVIDIA H100 80GB HBM3
CUDA version: 12.6
Loading model from: SimpleStories/SimpleStories-1.25M
This may take a moment...
✓ Tokenizer loaded successfully
✓ Pad token set to EOS token
✓ Model loaded successfully
✓ Model moved to cuda

Model loaded successfully on cuda
Model parameters: 1,245,824

============================================================
GENERATION EXAMPLE
============================================================

Generating text with prompt: 'The curious cat looked at the'
Generating...

Generated text:
the curious cat looked at the shiny object. " what is that? " alex wondered. he wondered, " is it a great treasure chest? " " maybe treasure is gold, " he replied. the cat chuckled, " but it is not gold. it is a shiny thing. " the cat smiled. " good luck is everywhere! " " what? " alex asked. the cat thought for a moment. " maybe it was the magic stone! " said the cat. the cat nodded. the cat ' s eyes lit up, and he knew he had to go home.
```

## Troubleshooting

### Common Issues

1. **Model Download Issues**
   - Ensure you have an internet connection
   - Check if you have enough disk space
   - Verify the model path is correct

2. **CUDA Issues**
   - The script will automatically fall back to CPU if CUDA is not available
   - Check your PyTorch installation supports CUDA if you expect GPU acceleration

3. **Memory Issues**
   - The 1.25M parameter model is quite small and should work on most systems
   - If you encounter memory issues, try reducing `max_new_tokens`

### Performance Notes

- **GPU**: Much faster generation, especially for longer sequences
- **CPU**: Slower but functional for testing and development
- **Model Size**: The 1.25M parameter model is very lightweight and fast

## Customization

You can easily modify the scripts to:
- Change the model size (e.g., to "3M" or "7M" if available)
- Adjust generation parameters (temperature, max_tokens, etc.)
- Use different prompts
- Add custom post-processing

## Integration

These scripts can be easily integrated into larger projects or used as starting points for more complex text generation applications.
