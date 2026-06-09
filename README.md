# Diffusion Activation Sparsification

Research implementation of activation sparsification techniques for efficient diffusion transformers. This project explores quantile-based approaches to reduce computational costs during training and inference without significant loss in generation quality.

## Overview

Diffusion transformers (DiT, DiffiT) achieve state-of-the-art image generation quality but require substantial computational resources. This work proposes two techniques to address this:

- **Quantile Layer Normalization (QLayerNorm)**: Replaces mean-centering with quantile-based centering, enabling explicit control over activation sparsity while preserving smooth activation functions.
- **Quantile GELU Masking (QGELU)**: Applies binary masks based on quantile thresholds to GELU outputs, creating structured sparsity patterns.

Both methods use **Exponential Moving Average (EMA)** statistics to eliminate runtime overhead after a warmup phase.

## Key Results

**Experiments on ImageNet-1K:**

| Model | Metric | Baseline | QLayerNorm | Improvement |
|-------|--------|----------|-----------|-------------|
| DiT-S/2 | FID | 49.40 | 48.31 | ↓ 1.09 |
| DiffiT-XL/2 | FID | 11.47 | 10.26 | ↓ 1.21 |

- **Training throughput**: Maintained close to baseline
- **Activation sparsity**: Significantly increased while maintaining model quality
- **Hardware acceleration**: Compatible with 2:4 structured sparsity for ~1.5× forward pass speedup
- **VRAM reduction**: Lower intermediate activation storage enables training on smaller GPUs

## Features

### Modifiers
- **QLayerNorm**: Quantile-based layer normalization with learnable quantile parameters
- **SparseGELU**: GELU activation with quantile-based masking and running statistics
- Customizable quantile search modes: global, batchwise, or channelwise
- EMA-based statistics accumulation for efficient inference

### Models
- **DiT**: Scalable Diffusion Models with Transformers (base implementation)
- **DiffiT**: Diffusion Vision Transformer variants
- **ControlNet**: Adapter for adding spatial controls to diffusion models

### Analysis Tools
- **SparsityAnalyser**: Tracks activation sparsity patterns across layers
- **DistributionAnalyser**: Monitors activation distribution changes
- **EffectiveRankAnalyser**: Analyzes representation complexity
- **WeightDriftAnalyser**: Detects weight bias shifts during training

## Installation

### Using Conda
```bash
conda env create -f environment.yml
conda activate diffusion-activation-sparsification
```

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 1.13 with CUDA 11.7
- torchvision
- timm
- diffusers
- accelerate
- tensorflow-gpu (for evaluation metrics)

## Quick Start

### Training a Base Model

```bash
python train.py \
    --model DiT-S/2 \
    --batch-size 256 \
    --results-dir ./results \
    --num-classes 1000
```

### Training with QLayerNorm

```bash
python train.py \
    --model DiT-S/2 \
    --norm-layer qlayernorm \
    --quantile 0.1 \
    --batch-size 256 \
    --results-dir ./results
```

### Training with QGELU Sparsification

```bash
python train.py \
    --model DiT-S/2 \
    --act-layer sparsegelu \
    --sparsity-level 0.3 \
    --batch-size 256 \
    --results-dir ./results
```

### ControlNet Adaptation

```bash
python train.py \
    --model DiT-S/2 \
    --ckpt /path/to/pretrained/dit-s-2.pt \
    --train-controlnet \
    --controlnet-depth 6 \
    --batch-size 128 \
    --results-dir ./results
```

## Project Structure

```
diffusion-activation-sparsification/
├── models/
│   ├── dit.py              # DiT model implementation
│   ├── diffit.py           # DiffiT variants
│   ├── controlnet.py       # ControlNet adapter
│   └── common_layers.py    # Shared layer definitions
├── modifiers/
│   ├── normalization.py    # QLayerNorm and other normalizations
│   ├── activation.py       # SparseGELU and activation functions
│   ├── mlp.py             # MLP layer variants
│   ├── attention.py       # Attention layer variants
│   ├── decorators.py      # Module wrappers for analysis
│   └── utils.py           # Helper utilities
├── diffusion/
│   ├── gaussian_diffusion.py  # Diffusion process implementation
│   ├── diffusion_utils.py     # Utility functions
│   └── respace.py            # Timestep rescheduling
├── analysers/
│   ├── base.py              # Base analyzer class
│   ├── sparsity_analyser.py # Activation sparsity tracking
│   ├── distribution_analyser.py
│   ├── effective_rank_analyser.py
│   └── weight_drift_analyser.py
├── utils/
│   ├── train_utils.py      # Training utilities
│   ├── dataset_utils.py    # Dataset loading and preprocessing
│   └── common.py           # Common utilities
├── train.py                # Main training script
├── evaluator.py            # Model evaluation
├── sample_ddp.py           # Distributed sampling
└── extract_features.py     # Feature extraction utility
```

## Configuration

### Training Arguments

Key hyperparameters for training:

- `--model`: Model architecture (DiT-S/2, DiT-B/2, DiT-L/2, DiT-XL/2, DiffiT-XL/2)
- `--batch-size`: Batch size (default: 256)
- `--results-dir`: Output directory for checkpoints and logs
- `--num-classes`: Number of output classes (default: 1000)
- `--ckpt`: Path to pretrained checkpoint for fine-tuning
- `--lr`: Learning rate (default: 1e-4)
- `--epochs`: Number of training epochs (default: 100)

### Sparsification Parameters

- `--norm-layer`: Layer normalization type (layernorm, qlayernorm)
- `--quantile`: Quantile level for QLayerNorm (0.0-1.0)
- `--act-layer`: Activation function (gelu, sparsegelu)
- `--sparsity-level`: Target sparsity level for QGELU (0.0-1.0)
- `--quantile-search-mode`: How to compute quantiles (global, batchwise, channelwise)

## Experimental Features

### Running Statistics (EMA)

Both QLayerNorm and SparseGELU support exponential moving average (EMA) statistics:

- **Warmup phase**: Track statistics for N batches
- **Inference phase**: Use accumulated statistics without recomputation
- **Momentum**: Controls EMA update rate (default: 0.1)

### Analysis and Monitoring

Run analysis during training to monitor:

```python
from analysers import SparsityAnalyser, DistributionAnalyser, WeightDriftAnalyser

# Track sparsity patterns
sparsity_analyser = SparsityAnalyser(model)
sparsity_report = sparsity_analyser.analyze()

# Monitor activation distributions
dist_analyser = DistributionAnalyser(model)
dist_report = dist_analyser.analyze()

# Detect weight drift
drift_analyser = WeightDriftAnalyser(model)
drift_report = drift_analyser.analyze()
```

## Evaluation

### FID Score Computation

```bash
python evaluator.py \
    --model-path /path/to/checkpoint.pt \
    --model-type DiT-S/2 \
    --batch-size 256 \
    --num-samples 50000
```

### Feature Extraction

Extract intermediate features for analysis:

```bash
python extract_features.py \
    --model-path /path/to/checkpoint.pt \
    --data-path /path/to/imagenet \
    --output-dir ./features
```

## Known Limitations

1. **ControlNet Compatibility**: Changed activation distributions break compatibility with standard zero-convolution initialization. Requires adapted fine-tuning protocols.

2. **Variable-Length Sequences**: EMA-based statistics work best with fixed spatial token structure (as in DiT). May require adjustment for models with variable sequence lengths.

3. **Hardware Dependency**: Actual speedup depends on hardware support for sparse matrix operations. Requires specialized acceleration for 2:4 structured sparsity.

## Citation

If you use this work in your research, please cite:

```bibtex
@thesis{serkov2026thesis,
  author = {Serkov, Aleksandr Maksimovich},
  title = {Исследование и разработка подходов на основе разрежения активаций 
           для создания эффективных диффузионных трансформеров},
  year = {2026},
  school = {National Research University Higher School of Economics},
  note = {Bachelor's Thesis}
}
```

## References

- Peebles et al. (2022). "[Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)" (DiT)
- Hatamizadeh et al. (2024). "[DiffiT: Diffusion Vision Transformers](https://arxiv.org/abs/2401.11577)"
- Hoefler et al. (2021). "[Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks](https://arxiv.org/abs/2102.00554)"
- Wang et al. (2023). "[Structural Sparsity in Models: Pruning, Growth for Inference Acceleration with Minimal Accuracy Loss](https://arxiv.org/abs/2306.08629)"

## License

This project is part of an academic thesis at National Research University Higher School of Economics.

## Contact

For questions or feedback about the implementation, please open an issue on GitHub.

---

**Note**: This implementation is based on the [fast-dit](https://github.com/chuanyangjin/fast-DiT) repository. Modifications focus on activation sparsification and quantile-based techniques.
