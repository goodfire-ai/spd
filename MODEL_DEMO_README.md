# Model Demo Scripts - SimpleStories & Pythia

This directory contains scripts to interact with both SimpleStories and Pythia models for text generation.

## 🚀 Quick Start

### Unified Script (Recommended)
```bash
# SimpleStories model
python model_demo.py --model simplestories

# Pythia 70M model
python model_demo.py --model pythia

# Interactive mode with custom prompt
python model_demo.py --model pythia --interactive --prompt "The robot discovered"
```

### Individual Model Scripts
```bash
# SimpleStories only
python simplestories_simple.py
python simplestories_demo.py

# Pythia only
python pythia_demo.py
```

## 📋 Available Scripts

### 1. `model_demo.py` (Unified - Recommended)
A comprehensive script that supports both SimpleStories and Pythia models:
- **Command-line interface** with multiple options
- **Automatic model detection** and loading
- **Interactive mode** for custom prompts
- **Flexible parameters** (temperature, max tokens, etc.)

**Usage Examples:**
```bash
# Basic usage
python model_demo.py --model simplestories
python model_demo.py --model pythia

# With custom parameters
python model_demo.py --model pythia --max-tokens 300 --temperature 0.8

# Interactive mode
python model_demo.py --model simplestories --interactive

# Custom prompt
python model_demo.py --model pythia --prompt "In a distant galaxy"
```

### 2. `pythia_demo.py`
A simple script specifically for the Pythia 70M model:
- Minimal and straightforward
- Default prompt: "Once upon a time"
- Fixed parameters optimized for Pythia

### 3. `simplestories_simple.py` & `simplestories_demo.py`
Original SimpleStories scripts (unchanged)

## 🤖 Supported Models

### SimpleStories Models
- **Model**: SimpleStories-1.25M
- **Parameters**: ~1.25 million
- **Model Type**: LlamaForCausalLM
- **Source**: `SimpleStories/SimpleStories-1.25M`
- **Default Prompt**: "The curious cat looked at the"
- **Best For**: Creative storytelling, children's stories

### Pythia Models
- **Model**: Pythia-70M
- **Parameters**: ~70 million
- **Model Type**: GPTNeoXForCausalLM
- **Source**: `EleutherAI/pythia-70m`
- **Default Prompt**: "Once upon a time"
- **Best For**: General text generation, more coherent longer texts

## ⚙️ Command Line Options

The unified script (`model_demo.py`) supports the following options:

```bash
python model_demo.py [OPTIONS]

Options:
  --model, -m {simplestories,pythia}
                        Model to use (default: simplestories)
  --size, -s SIZE       Model size (e.g., '1.25M' for SimpleStories, '70m' for Pythia)
  --prompt, -p PROMPT   Custom prompt (default: model-specific)
  --max-tokens, -t NUM  Maximum new tokens to generate (default: 400)
  --temperature, -temp FLOAT
                        Sampling temperature (default: 0.7)
  --interactive, -i     Run in interactive mode
  --help, -h            Show help message
```

## 📊 Model Comparison

| Feature | SimpleStories-1.25M | Pythia-70M |
|---------|---------------------|------------|
| **Parameters** | 1.25M | 70M |
| **Model Type** | LlamaForCausalLM | GPTNeoXForCausalLM |
| **Size** | Very Small | Small |
| **Speed** | Very Fast | Fast |
| **Quality** | Good for short stories | Better for longer texts |
| **Use Case** | Creative, whimsical | General purpose |
| **Default Prompt** | "The curious cat looked at the" | "Once upon a time" |

## 🎯 Example Outputs

### SimpleStories Output
```
Prompt: "The curious cat looked at the"
Generated text:
the curious cat looked at the flowers and smiled. as he returned home, the boy felt happy. the garden was full of laughter, and he had learned that true happiness comes from within.
```

### Pythia Output
```
Prompt: "Once upon a time"
Generated text:
Once upon a time, the first time it was possible to find a job, and there was no one to be hired. He was a private life coach who, according to a report from the New York Times, had already done so. As a result, he was not an associate of an associate of a prominent businessman who had been the head of the New York Times' division of trade associations.
```

## 🔧 Technical Details

### Automatic Model Detection
The unified script automatically detects the model type based on:
1. Model path (SimpleStories vs pythia)
2. Model configuration (LlamaForCausalLM vs GPTNeoXForCausalLM)
3. Fallback auto-detection if path doesn't match

### Device Management
- **Automatic CUDA detection** and usage
- **CPU fallback** if CUDA is not available
- **Device information** display (GPU name, CUDA version)

### Token Handling
- **Proper pad token setup** (no warnings)
- **Attention mask handling** for reliable generation
- **EOS token management** for clean text endings

## 🚀 Performance Notes

- **SimpleStories**: Very fast, good for quick experiments
- **Pythia**: Slightly slower but better quality for longer texts
- **GPU Acceleration**: Both models benefit significantly from CUDA
- **Memory Usage**: Both models are lightweight and work well on most systems

## 🔍 Troubleshooting

### Common Issues

1. **Model Download Issues**
   - Ensure internet connection
   - Check disk space
   - Verify model paths are correct

2. **CUDA Issues**
   - Scripts automatically fall back to CPU
   - Check PyTorch CUDA installation if expecting GPU acceleration

3. **Generation Quality**
   - Try different temperatures (0.5-1.0)
   - Adjust max_tokens for desired length
   - Experiment with different prompts

### Getting Help
```bash
python model_demo.py --help
```

## 🎨 Customization

You can easily extend the scripts to:
- Add more model types
- Implement custom generation parameters
- Add post-processing steps
- Integrate with other tools

The unified script is designed to be easily extensible for additional models in the future.
