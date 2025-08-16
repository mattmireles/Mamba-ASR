# Device Optimization Guide - Mamba-ASR Multi-Platform Deployment

## Overview

This guide provides device-specific optimization strategies for Mamba-ASR deployment across CUDA, Apple Silicon MPS, and CPU platforms. Each platform has unique characteristics that require tailored configuration approaches for optimal performance.

## Platform-Specific Optimization Matrix

| Platform | Memory Architecture | Compute Pattern | Optimization Focus |
|----------|-------------------|-----------------|-------------------|
| **CUDA** | Discrete GPU Memory | Parallel Throughput | Batch Size, Mixed Precision |
| **Apple Silicon MPS** | Unified Memory | Memory Bandwidth | Memory Pressure, Operation Support |
| **CPU** | System Memory | Sequential Processing | Thread Count, Memory Usage |

## CUDA Optimization Strategies

### Hardware Requirements
- **Minimum**: RTX 3080 (10GB VRAM) for training small models
- **Recommended**: RTX 4090 (24GB) or A100 (40GB) for large models  
- **Production**: A100 (80GB) or H100 (80GB) for maximum throughput

### CUDA Configuration Template
```yaml
# CUDA-Optimized Configuration
precision: bf16                    # BFloat16 for A100/H100, fp16 for older GPUs
batch_size: 32                     # Large batches for GPU throughput
grad_accumulation_factor: 4        # Effective batch = 128-256
max_grad_norm: 5.0                # Standard gradient clipping
num_workers: 16                    # High parallelism for data loading

# Dynamic batching for memory efficiency
dynamic_batching: True
max_batch_length_train: 850        # Maximize GPU utilization
max_batch_length_val: 100         # Conservative for beam search
num_bucket: 200                    # Fine-grained length bucketing

# Memory optimization
torch.backends.cudnn.benchmark: True  # Optimize for fixed input sizes
torch.backends.cuda.matmul.allow_tf32: True  # Faster matmul on Ampere+
```

### CUDA Performance Tuning

#### Memory Optimization
```python
# Environment variables for CUDA optimization
export CUDA_LAUNCH_BLOCKING=0           # Asynchronous kernel launches
export TORCH_BACKENDS_CUDNN_BENCHMARK=1 # cuDNN optimization
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"  # Memory fragmentation
```

#### Batch Size Scaling
- **RTX 3080 (10GB)**: batch_size=8, grad_accumulation=8  
- **RTX 4090 (24GB)**: batch_size=16, grad_accumulation=4
- **A100 (40GB)**: batch_size=32, grad_accumulation=4
- **A100 (80GB)**: batch_size=64, grad_accumulation=2

#### Mixed Precision Guidelines
```yaml
# Precision selection by GPU generation
precision: bf16    # Ampere+ (A100, RTX 30/40 series)
precision: fp16    # Turing/Volta (V100, RTX 20 series)  
precision: fp32    # Older architectures or debugging
```

## Apple Silicon MPS Optimization

### Hardware Support Matrix
- **M1**: 8-core GPU, 68.25 GB/s memory bandwidth
- **M1 Pro**: 16-core GPU, 200 GB/s memory bandwidth  
- **M1 Max**: 32-core GPU, 400 GB/s memory bandwidth
- **M2**: 10-core GPU, 100 GB/s memory bandwidth
- **M3**: Up to 40-core GPU, 300 GB/s memory bandwidth

### MPS Configuration Template
```yaml
# Apple Silicon MPS-Optimized Configuration
precision: fp32                    # Full precision for MPS stability
batch_size: 8                      # Conservative batch sizes
grad_accumulation_factor: 8        # Compensate with accumulation
max_grad_norm: 2.0                # Lower clipping for stability
num_workers: 4                     # Limited by memory bandwidth

# Memory pressure management
dynamic_batching: True
max_batch_length_train: 400        # Reduced for memory pressure
max_batch_length_val: 50          # Conservative validation batches
num_bucket: 100                    # Coarser bucketing

# MPS-specific optimizations
PYTORCH_MPS_HIGH_WATERMARK_RATIO: 0.7  # Memory pressure threshold
PYTORCH_ENABLE_MPS_FALLBACK: 0    # Disable fallback in production
```

### MPS Performance Considerations

#### Memory Architecture Impact
```python
# Unified memory optimization
# Apple Silicon shares CPU/GPU memory - optimize total system usage
system_memory_gb = 16  # Adjust based on your system
training_memory_limit = system_memory_gb * 0.7  # Leave 30% for system
```

#### Operation Support Matrix
```yaml
# MPS-supported operations (as of PyTorch 2.1)
attention_type: RelPosMHAXL     # Fully supported
activation: !name:torch.nn.GELU # Supported activation
loss_reduction: batchmean       # Supported reduction

# Operations requiring CPU fallback
# - Some advanced indexing patterns
# - Certain sparse operations
# - Complex number operations
```

#### Device Detection and Setup
```python
# MPS device detection and optimization
import torch

def get_optimal_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        # Set memory fraction
        torch.mps.set_per_process_memory_fraction(0.7)
        return device
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

## CPU Optimization Strategies

### CPU Configuration Template
```yaml
# CPU-Optimized Configuration
precision: fp32                    # Full precision required
batch_size: 4                      # Memory-constrained batches
grad_accumulation_factor: 16       # Large accumulation for effective batch
max_grad_norm: 1.0                # Conservative gradient clipping
num_workers: 2                     # Limited by compute capacity

# Simplified batching strategy
dynamic_batching: False            # Fixed batching for predictability
shuffle: True                      # Manual shuffling instead of dynamic

# CPU-specific optimizations
torch.set_num_threads: 8           # Match CPU core count
torch.set_num_interop_threads: 2   # Inter-operation parallelism
```

### CPU Performance Optimization

#### Thread Configuration
```python
# Optimal CPU threading setup
import torch
import os

# Set thread counts based on CPU
cpu_count = os.cpu_count()
torch.set_num_threads(min(cpu_count, 8))      # Intra-op parallelism
torch.set_num_interop_threads(2)              # Inter-op parallelism

# NUMA optimization for multi-socket systems
os.environ["OMP_NUM_THREADS"] = str(min(cpu_count, 8))
os.environ["MKL_NUM_THREADS"] = str(min(cpu_count, 8))
```

#### Memory Optimization
```yaml
# Memory-conscious configuration
max_batch_length_train: 200        # Shorter sequences
max_batch_length_val: 50          # Conservative validation
num_bucket: 50                     # Coarse bucketing
sorting: ascending                 # Predictable memory usage
```

## Device-Specific Architecture Recommendations

### CUDA: Maximum Performance
```yaml
# Conformer Large S2S for CUDA
encoder_module: conformer
decoder_module: transformer
num_encoder_layers: 18
num_decoder_layers: 6
d_model: 512
nhead: 8
batch_size: 32
precision: bf16
```

### Apple Silicon MPS: Balanced Efficiency
```yaml
# ConMamba Large for MPS efficiency
encoder_module: conmamba
decoder_module: transformer  
num_encoder_layers: 12
num_decoder_layers: 6
d_model: 256
d_state: 16
batch_size: 8
precision: fp32
```

### CPU: Resource Conscious
```yaml
# Conformer Small CTC for CPU
encoder_module: conformer
num_encoder_layers: 6
num_decoder_layers: 0  # CTC only
d_model: 256
nhead: 4
batch_size: 4
precision: fp32
```

## Cross-Platform Development Workflow

### Environment Setup
```bash
# CUDA environment
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Apple Silicon environment  
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
export PYTORCH_ENABLE_MPS_FALLBACK=0

# CPU environment
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### Device Detection Script
```python
def setup_training_device():
    """Automatically configure device-optimal settings."""
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        config_updates = {
            "batch_size": 32,
            "precision": "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16",
            "num_workers": 16,
            "grad_accumulation_factor": 4
        }
        print(f"CUDA detected: {torch.cuda.get_device_name()}")
        
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        config_updates = {
            "batch_size": 8, 
            "precision": "fp32",
            "num_workers": 4,
            "grad_accumulation_factor": 8
        }
        print("Apple Silicon MPS detected")
        
    else:
        device = torch.device("cpu")
        config_updates = {
            "batch_size": 4,
            "precision": "fp32", 
            "num_workers": 2,
            "grad_accumulation_factor": 16
        }
        print("CPU training mode")
    
    return device, config_updates
```

## Performance Benchmarking

### Expected Training Times (Conformer Large)

| Platform | Hardware | Training Time | Memory Usage | WER |
|----------|----------|---------------|--------------|-----|
| CUDA | A100 80GB | 3-4 days | 60GB | 2.8% |
| CUDA | RTX 4090 | 5-7 days | 20GB | 2.8% |
| MPS | M1 Max | 8-12 days | 16GB | 3.2% |
| MPS | M2 | 12-18 days | 12GB | 3.2% |
| CPU | 32-core Xeon | 20-30 days | 8GB | 3.5% |

### Inference Performance (Real-time Factor)

| Platform | Hardware | RTF | Memory | Batch Size |
|----------|----------|-----|---------|------------|
| CUDA | A100 | 0.05x | 4GB | 32 |
| CUDA | RTX 4090 | 0.1x | 2GB | 16 |
| MPS | M1 Max | 0.3x | 2GB | 8 |
| MPS | M2 | 0.5x | 1.5GB | 4 |
| CPU | Intel i9 | 2.0x | 1GB | 1 |

## Troubleshooting Device-Specific Issues

### CUDA Issues
```bash
# Out of memory
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"
# Reduce batch_size or enable gradient_checkpointing

# Slow training
nvidia-smi  # Check GPU utilization
# Increase num_workers if GPU util < 90%

# cuDNN errors
export CUDNN_DETERMINISTIC=1
export CUDNN_BENCHMARK=0
```

### MPS Issues  
```bash
# Operation not supported
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Debug mode only

# Memory pressure
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6  # More conservative

# Numerical instability
# Switch to fp32 precision
# Reduce learning rate by 0.5x
```

### CPU Issues
```bash
# Slow training
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)

# Memory issues  
# Reduce batch_size to 1-2
# Disable dynamic_batching
# Use gradient_checkpointing
```

## Production Deployment Considerations

### CUDA Production
- Use TensorRT for inference optimization
- Implement dynamic batching for throughput
- Monitor GPU memory fragmentation
- Use mixed precision in production

### Apple Silicon Production
- Profile MPS operation support regularly
- Implement CPU fallback strategies
- Monitor memory pressure closely
- Test across different Apple Silicon variants

### CPU Production
- Use ONNX for inference optimization
- Implement quantization for mobile deployment
- Consider model distillation for edge devices
- Profile thread count vs. performance

This guide provides comprehensive device-specific optimization strategies for deploying Mamba-ASR across diverse hardware platforms while maintaining the AI-first documentation principles.