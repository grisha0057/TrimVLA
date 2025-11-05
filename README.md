# TrimVLA

An exploratory study of a pruning algorithm for VLA models. This project was originally forked from the [LightVLA](https://github.com/LiAutoAD/LightVLA) repository.

## Hardware Requirements

- 2x NVIDIA RTX 4090 (48GB VRAM each)

## Setup

### 1. Environment Setup

```bash
# Create conda environment
conda create -n trimvla python=3.10 -y
conda activate trimvla

# Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -e .
```

### 2. Install LIBERO (for robot experiments)

```bash
cd experiments/robot/libero
pip install -r libero_requirements.txt
```

## Quick Start

### Training

```bash
# Train on LIBERO spatial tasks
bash train_scripts/train_libero_spatial.sh
```

### Evaluation

```bash
# Evaluate on LIBERO spatial tasks
bash eval_scripts/eval_libero_spatial.sh
```

### Fine-tuning with LoRA

```bash
# Fine-tune the model
python vla-scripts/finetune.py

# Merge LoRA weights
python vla-scripts/merge_lora_weights_and_save.py
```

## Project Structure

```
TrimVLA/
├── prismatic/          # Core VLA model implementation
│   ├── models/         # Model architectures (VLMs, VLAs)
│   ├── vla/            # VLA-specific components
│   └── training/       # Training strategies (DDP, FSDP)
├── experiments/        # Robot experiment scripts
│   └── robot/libero/   # LIBERO benchmark utilities
├── overfit_experiment/ # Overfitting analysis tools
├── train_scripts/      # Training shell scripts
├── eval_scripts/       # Evaluation shell scripts
└── vla-scripts/        # VLA training and deployment scripts
```

## Checkpoints

200, 400, 800-step checkpoints can be found [here](https://huggingface.co/grisha0057).