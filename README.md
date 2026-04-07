# TransSteer

A tool for extracting steering vectors from Large Language Models using the `steering_vectors` library.

## Features

- Load Llama-2-7B (or other models) in 4-bit quantization using BitsAndBytesConfig for efficient memory usage
- Extract steering vectors using Standard Mean aggregation
- Extract steering vectors using Private Mean aggregation with differential privacy (noise_multiplier and clip parameters)
- Save extracted vectors as `.npy` files
- Automatic VRAM cleanup after extraction

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python extraction.py --dataset_path path/to/your/dataset.pkl
```

### Full Options

```bash
python extraction.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --output_dir ./vectors \
    --dataset_path path/to/your/dataset.pkl \
    --layers 10 11 12 13 14 15 \
    --read_token_index -2 \
    --noise_multiplier 1.0 \
    --clip 1.0
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `meta-llama/Llama-2-7b-hf` | HuggingFace model name or path |
| `--output_dir` | `./vectors` | Directory to save extracted vectors |
| `--dataset_path` | `None` | Path to training dataset for extraction |
| `--layers` | `None` (all) | Specific layers to extract vectors from |
| `--read_token_index` | `-2` | Token index to read activations from |
| `--noise_multiplier` | `1.0` | Noise multiplier for private mean (DP) |
| `--clip` | `1.0` | Clipping bound for private mean (DP) |

## Output

The script saves two vector files in the `vectors/` directory:

- `standard_mean_vector.npy` - Steering vector using standard mean aggregation
- `private_mean_vector.npy` - Steering vector using private mean aggregation with differential privacy

## Memory Management

The script automatically:
1. Loads the model with 4-bit quantization to reduce VRAM usage
2. Deletes the model and tokenizer after extraction
3. Empties the CUDA cache to free VRAM

## Requirements

- Python 3.8+
- CUDA-capable GPU with sufficient VRAM
- HuggingFace account with access to Llama-2 models (if using default model)