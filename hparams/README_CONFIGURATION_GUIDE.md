# Mamba-ASR Configuration Guide - AI-First Documentation

## Overview

This guide provides comprehensive documentation for configuring Mamba-ASR across different architectures, training paradigms, and deployment scenarios. All configurations follow AI-first documentation principles for maximum clarity and maintainability.

## Architecture Selection Matrix

### CTC vs S2S Training Paradigms

| Training Type | Architecture | Use Case | Complexity | Performance |
|---------------|-------------|----------|------------|-------------|
| **CTC** | Encoder-Only | Streaming ASR, Fast Inference | Low | Good |
| **S2S** | Encoder-Decoder | Highest Accuracy, Language Integration | High | Excellent |

### Encoder Architecture Options

| Encoder | Strengths | Best For | Memory | Speed |
|---------|-----------|----------|---------|-------|
| **Conformer** | Audio modeling excellence | Production ASR | Medium | Fast |
| **ConMamba** | Long sequences, efficiency | Research, streaming | Low | Very Fast |
| **Transformer** | Baseline, well-understood | Research, ablation | Medium | Medium |

## Configuration Parameter Flow

```
YAML Configuration
    ↓
SpeechBrain Object Instantiation
    ↓
TransformerASR Class
    ↓
Architecture-Specific Modules
    ↓
Training/Inference Pipeline
```

## Model Scaling Strategies

### Small vs Large Configurations

#### Conformer Scaling
- **Small**: 12 layers, 256 d_model, 4 heads → ~25M params
- **Large**: 18 layers, 512 d_model, 8 heads → ~100M params

#### ConMamba Scaling  
- **Small**: 12 layers, 256 d_model, d_state=8 → ~20M params
- **Large**: 18 layers, 512 d_model, d_state=16 → ~80M params

### Mamba-Specific Parameters

#### Critical Mamba Configuration
```yaml
# State space model parameters
d_state: 16          # Hidden state dimension (8=small, 16=large, 32=research)
expand: 2            # Expansion factor for hidden dimension  
d_conv: 4            # Convolution kernel size for local modeling
bidirectional: True  # Bidirectional processing for better accuracy
```

#### Parameter Tuning Guidelines
- **d_state**: Controls model capacity (16 optimal for most cases)
- **expand**: Memory/compute trade-off (2 = balanced, 4 = high capacity)
- **d_conv**: Local context size (4 = standard speech modeling)
- **bidirectional**: True for offline ASR, False for streaming

## Device Optimization Strategies

### CUDA Optimization
```yaml
precision: bf16           # BFloat16 for A100/H100
batch_size: 32           # Large batches for throughput
grad_accumulation_factor: 4  # Effective batch = 128
num_workers: 16          # High parallelism
```

### Apple Silicon MPS Optimization
```yaml
precision: fp32          # Full precision for stability
batch_size: 8            # Smaller batches for memory
grad_accumulation_factor: 8  # Compensate with accumulation
num_workers: 4           # Limited by memory bandwidth
```

### CPU Optimization
```yaml
precision: fp32          # Full precision required
batch_size: 4            # Memory-constrained
dynamic_batching: False  # Simpler batching strategy
num_workers: 2           # Limited by compute
```

## Training Strategy Selection

### CTC Training (Fast, Simple)
- **Advantages**: Fast convergence, streaming-ready, memory efficient
- **Use Cases**: Production systems, resource constraints, streaming
- **Trade-offs**: Lower accuracy ceiling, limited language modeling

### S2S Training (Accuracy, Complex)
- **Advantages**: Maximum accuracy, language model integration, flexibility
- **Use Cases**: Research, offline ASR, maximum performance requirements
- **Trade-offs**: Complex training, higher memory, slower inference

### Joint CTC/Attention Training
- **Loss Weighting**: `ctc_weight: 0.3, attention_weight: 0.7`
- **Benefits**: Combines alignment robustness with language modeling
- **Complexity**: Requires careful hyperparameter tuning

## Memory and Performance Guidelines

### Training Memory Requirements

| Configuration | GPU Memory | Training Time | Expected WER |
|---------------|------------|---------------|--------------|
| Conformer Small CTC | 8GB | 2-3 days | 4-5% |
| Conformer Large CTC | 12GB | 3-5 days | 3-4% |
| Conformer Large S2S | 16GB | 5-7 days | 2-3% |
| ConMamba Large S2S | 12GB | 4-6 days | 2-3% |

### Inference Performance

| Architecture | Real-time Factor | Memory | Use Case |
|-------------|------------------|---------|----------|
| Conformer CTC | 0.1-0.3x | 2GB | Production |
| Conformer S2S | 0.5-1.0x | 4GB | High accuracy |
| ConMamba CTC | 0.05-0.2x | 1GB | Edge deployment |

## Configuration File Structure

### CTC Configuration Template
```
hparams/CTC/
├── conformer_large.yaml    # Production Conformer CTC
├── conmamba_large.yaml     # Research ConMamba CTC
└── conformer_small.yaml    # Resource-constrained CTC
```

### S2S Configuration Template  
```
hparams/S2S/
├── conformer_large.yaml      # Maximum accuracy Conformer
├── conmamba_large.yaml       # Efficient ConMamba S2S
├── conmambamamba_large.yaml  # ConMamba encoder + Mamba decoder
└── *_small.yaml             # Resource-constrained variants
```

## Architecture-Specific Considerations

### Conformer Architecture
- **Strengths**: Proven audio modeling, streaming support, production ready
- **Configuration Focus**: Kernel sizes, attention types, macaron scaling
- **Optimal Settings**: `kernel_size: 31, attention_type: RelPosMHAXL`

### ConMamba Architecture
- **Strengths**: Linear complexity, memory efficiency, long sequences
- **Configuration Focus**: State space parameters, bidirectionality
- **Optimal Settings**: `d_state: 16, bidirectional: True, d_conv: 4`

### Hybrid Architectures
- **ConMamba + Transformer Decoder**: Best of both worlds
- **ConMamba + Mamba Decoder**: Full state-space model pipeline
- **Configuration Strategy**: Balance complexity vs. performance gains

## Language Model Integration

### External LM Configuration
```yaml
pretrained_lm_tokenizer_path: speechbrain/asr-transformer-transformerlm-librispeech
lm_weight: 0.60              # Strong LM influence
ctc_weight_decode: 0.40      # Balanced CTC during decoding
no_lm: False                 # Enable LM integration
```

### LM Weight Tuning
- **High LM Weight (0.8)**: Better fluency, may sacrifice acoustic accuracy
- **Balanced LM Weight (0.6)**: Optimal trade-off for most applications  
- **Low LM Weight (0.3)**: Acoustic accuracy priority, faster decoding

## Troubleshooting Common Issues

### Training Instability
- **Gradient Explosion**: Reduce `max_grad_norm` to 1.0-2.0
- **Loss Divergence**: Lower learning rate, increase warmup steps
- **Memory Issues**: Reduce batch size, enable gradient checkpointing

### Performance Issues
- **Slow Convergence**: Increase effective batch size via accumulation
- **Poor Accuracy**: Check tokenizer alignment, increase model capacity
- **Memory Pressure**: Use mixed precision, reduce sequence lengths

### Device-Specific Issues
- **MPS Fallbacks**: Enable `PYTORCH_ENABLE_MPS_FALLBACK=1` for debugging
- **CUDA OOM**: Use gradient checkpointing, reduce precision to fp16
- **CPU Performance**: Disable dynamic batching, reduce worker count

## Best Practices

### Configuration Management
1. **Version Control**: Track configuration changes with git
2. **Reproducibility**: Always set seeds and document environment
3. **Documentation**: Comment all non-standard parameter choices
4. **Validation**: Test configurations on small datasets first

### Experimentation Strategy
1. **Baseline First**: Start with known-good configurations
2. **Single Changes**: Modify one parameter at a time
3. **Systematic Scaling**: Follow established scaling laws
4. **Performance Monitoring**: Track both accuracy and efficiency metrics

### Production Deployment
1. **Configuration Freezing**: Lock successful hyperparameters
2. **Model Averaging**: Use checkpoint averaging for stability
3. **Inference Optimization**: Profile and optimize critical paths
4. **Monitoring**: Continuous performance and accuracy tracking

## Configuration Examples

### Quick Start Configurations

#### For Research/Experimentation
```yaml
# Use Conformer Small CTC for fast iteration
encoder_module: conformer
num_encoder_layers: 12
d_model: 256
batch_size: 16
```

#### For Production Deployment
```yaml
# Use Conformer Large S2S for maximum accuracy  
encoder_module: conformer
decoder_module: transformer
num_encoder_layers: 18
d_model: 512
lm_weight: 0.60
```

#### For Resource-Constrained Environments
```yaml
# Use ConMamba for efficiency
encoder_module: conmamba
d_model: 256
d_state: 8
precision: fp16
```

This guide provides the foundation for understanding and configuring Mamba-ASR across different scenarios. Each configuration file contains detailed parameter explanations following AI-first documentation principles.