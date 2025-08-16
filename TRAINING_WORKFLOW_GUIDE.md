# Mamba-ASR Training Workflow Guide

## Overview

This comprehensive guide provides step-by-step instructions for training and deploying Mamba-ASR models from data preparation through production deployment. Following AI-first documentation principles, this guide enables the next developer (who is an AI) to successfully navigate the complete machine learning workflow.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Data Preparation](#data-preparation)
3. [Configuration Selection](#configuration-selection)
4. [Model Training](#model-training)
5. [Model Evaluation](#model-evaluation)
6. [Model Deployment](#model-deployment)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)
8. [Troubleshooting](#troubleshooting)

## Environment Setup

### Prerequisites

```bash
# System Requirements
- Python 3.8+ (native ARM64 for Apple Silicon)
- PyTorch 2.0+ with appropriate backend (CUDA/MPS/CPU)
- SpeechBrain framework
- LibriSpeech dataset (960h for full training)

# Hardware Recommendations
- CUDA: RTX 4090 (24GB) or A100 (40GB+) for optimal training
- Apple Silicon: M1 Max/M2/M3 with 32GB+ unified memory
- CPU: 32+ cores with 64GB+ RAM (training will be very slow)
```

### Installation

```bash
# 1. Clone repository
git clone <mamba-asr-repository>
cd Mamba-ASR

# 2. Create virtual environment
python -m venv mamba_env
source mamba_env/bin/activate  # Linux/macOS
# mamba_env\Scripts\activate  # Windows

# 3. Install dependencies
pip install torch torchaudio speechbrain hyperpyyaml
pip install sentencepiece wandb  # Optional: for advanced features

# 4. Verify device availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

### Environment Configuration

```bash
# CUDA Environment (recommended for production)
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Apple Silicon Environment
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
export PYTORCH_ENABLE_MPS_FALLBACK=0

# CPU Environment (development/testing only)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

## Data Preparation

### LibriSpeech Dataset Setup

```bash
# 1. Download LibriSpeech dataset
wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
wget http://www.openslr.org/resources/12/train-clean-360.tar.gz
wget http://www.openslr.org/resources/12/train-other-500.tar.gz
wget http://www.openslr.org/resources/12/dev-clean.tar.gz
wget http://www.openslr.org/resources/12/test-clean.tar.gz
wget http://www.openslr.org/resources/12/test-other.tar.gz

# 2. Extract all archives to a common directory
mkdir -p datasets/LibriSpeech
cd datasets/LibriSpeech
for file in ../../*.tar.gz; do tar -xzf "$file"; done

# 3. Verify dataset structure
ls -la  # Should show: train-clean-100, train-clean-360, etc.
```

### Data Preparation Execution

```bash
# 1. Navigate to Mamba-ASR directory
cd /path/to/Mamba-ASR

# 2. Run data preparation (automatic during training, or manual)
python -c "
from librispeech_prepare import prepare_librispeech
prepare_librispeech(
    data_folder='/path/to/datasets/LibriSpeech',
    save_folder='./data_prepared',
    tr_splits=['train-clean-100', 'train-clean-360', 'train-other-500'],
    dev_splits=['dev-clean'],
    te_splits=['test-clean', 'test-other'],
    skip_prep=False
)
"

# 3. Verify CSV generation
ls data_prepared/  # Should show: train.csv, dev-clean.csv, test-*.csv
```

### Data Preparation Validation

```python
# Verify data preparation quality
import pandas as pd

# Load and inspect training manifest
train_df = pd.read_csv('data_prepared/train.csv')
print(f"Training samples: {len(train_df)}")
print(f"Total duration: {train_df['duration'].sum() / 3600:.1f} hours")
print(f"Average duration: {train_df['duration'].mean():.2f} seconds")

# Check for missing files
import os
missing_files = []
for _, row in train_df.head(100).iterrows():  # Sample check
    if not os.path.exists(row['wav']):
        missing_files.append(row['wav'])
        
if missing_files:
    print(f"Warning: {len(missing_files)} missing audio files")
else:
    print("✓ All sampled audio files exist")
```

## Configuration Selection

### Training Strategy Decision Matrix

| Use Case | Model Type | Configuration | Expected WER | Training Time |
|----------|------------|---------------|--------------|---------------|
| **Quick Prototyping** | CTC Small | `hparams/CTC/conformer_small.yaml` | 4-5% | 1-2 days |
| **Production Baseline** | CTC Large | `hparams/CTC/conformer_large.yaml` | 3-4% | 3-5 days |
| **Maximum Accuracy** | S2S Large | `hparams/S2S/conformer_large.yaml` | 2-3% | 5-7 days |
| **Research/Efficiency** | ConMamba | `hparams/S2S/conmamba_large.yaml` | 2-3% | 4-6 days |

### Configuration Customization

```yaml
# Example: Customize hparams/CTC/conformer_large.yaml for your setup
data_folder: /path/to/datasets/LibriSpeech  # Update to your path
output_folder: ./results/my_experiment      # Experiment output directory

# Device-specific optimizations
precision: bf16        # A100/H100: bf16, V100: fp16, CPU: fp32
batch_size: 32         # GPU: 16-32, Apple Silicon: 8, CPU: 4
num_workers: 16        # GPU: 16, Apple Silicon: 4, CPU: 2
grad_accumulation_factor: 4  # Adjust based on memory constraints

# Training duration
number_of_epochs: 500  # Will early stop based on validation performance
```

### Advanced Configuration

```yaml
# Multi-GPU training configuration
# Modify batch_size and grad_accumulation_factor for multiple GPUs
# Effective batch = batch_size * n_gpus * grad_accumulation_factor
# Target: 256-512 effective batch size for stable convergence

# For 2x A100 (40GB each):
batch_size: 32
grad_accumulation_factor: 4  # Effective batch: 32 * 2 * 4 = 256

# For 4x RTX 4090 (24GB each):
batch_size: 16
grad_accumulation_factor: 4  # Effective batch: 16 * 4 * 4 = 256
```

## Model Training

### CTC Training Workflow

```bash
# 1. Single GPU training
python train_CTC.py hparams/CTC/conformer_large.yaml \
    --data_folder /path/to/datasets/LibriSpeech \
    --output_folder ./results/ctc_experiment

# 2. Multi-GPU training (recommended for production)
python -m speechbrain.utils.distributed.ddp_run \
    --nproc_per_node=2 \
    train_CTC.py hparams/CTC/conformer_large.yaml \
    --data_folder /path/to/datasets/LibriSpeech

# 3. Resume from checkpoint
python train_CTC.py hparams/CTC/conformer_large.yaml \
    --output_folder ./results/ctc_experiment \
    --resume ./results/ctc_experiment/save/CKPT+*.ckpt
```

### S2S Training Workflow

```bash
# 1. S2S training with language model
python train_S2S.py hparams/S2S/conformer_large.yaml \
    --data_folder /path/to/datasets/LibriSpeech \
    --output_folder ./results/s2s_experiment

# 2. S2S training without language model (faster)
python train_S2S.py hparams/S2S/conformer_large.yaml \
    --no_lm True \
    --output_folder ./results/s2s_no_lm

# 3. Multi-GPU S2S training
python -m speechbrain.utils.distributed.ddp_run \
    --nproc_per_node=2 \
    train_S2S.py hparams/S2S/conformer_large.yaml
```

### Training Monitoring

```python
# Monitor training progress
import pandas as pd
import matplotlib.pyplot as plt

# Read training log
log_path = "results/ctc_experiment/train_log.txt"
with open(log_path, 'r') as f:
    lines = f.readlines()

# Parse training statistics (simplified)
epochs, train_loss, valid_loss, wer = [], [], [], []
for line in lines:
    if "epoch:" in line:
        # Parse epoch statistics
        # Implementation depends on log format
        pass

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, valid_loss, label='Valid Loss')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs, wer, label='Validation WER')
plt.legend()
plt.title('Word Error Rate')
plt.show()
```

### Training Checkpoints

```bash
# Checkpoint management
ls results/ctc_experiment/save/
# CKPT+2024-01-15+10-30-45+00/  # Checkpoint directory
# - brain.ckpt                  # Model state
# - noam_scheduler.ckpt         # Learning rate scheduler
# - normalizer.ckpt             # Feature normalization
# - counter.ckpt                # Epoch counter

# Manual checkpoint averaging (if needed)
python -c "
import speechbrain as sb
import torch

# Find best checkpoints
ckpts = sb.utils.checkpoints.find_checkpoints(
    'results/ctc_experiment/save',
    min_key='WER'  # or max_key='ACC' for S2S
)

# Average checkpoints
averaged = sb.utils.checkpoints.average_checkpoints(
    ckpts[:10],  # Best 10 checkpoints
    recoverable_name='model'
)

# Save averaged model
torch.save(averaged, 'results/ctc_experiment/averaged_model.pt')
"
```

## Model Evaluation

### Comprehensive Evaluation

```bash
# 1. Evaluate on all test sets (automatic after training)
# Results saved to: results/experiment/wer_test-clean.txt
#                   results/experiment/wer_test-other.txt

# 2. Manual evaluation with specific checkpoint
python -c "
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from train_CTC import ASR, dataio_prepare  # or train_S2S

# Load configuration
with open('hparams/CTC/conformer_large.yaml') as f:
    hparams = load_hyperpyyaml(f, overrides={})

# Setup data pipeline
train_data, valid_data, test_datasets, train_bsampler, valid_bsampler = dataio_prepare(hparams, tokenizer)

# Initialize model
asr_brain = ASR(
    modules=hparams['modules'],
    opt_class=hparams['model_opt_class'],
    hparams=hparams,
    run_opts={'device': 'cuda'}  # or 'mps', 'cpu'
)

# Load specific checkpoint
asr_brain.checkpointer.recover_if_possible(
    device='cuda'  # Specify device
)

# Evaluate on test set
asr_brain.evaluate(
    test_datasets['test-clean'],
    min_key='WER',
    test_loader_kwargs=hparams['test_dataloader_opts']
)
"
```

### Performance Analysis

```python
# Analyze evaluation results
def parse_wer_file(filepath):
    \"\"\"Parse WER statistics from SpeechBrain output.\"\"\"
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract key metrics
    import re
    wer_match = re.search(r'WER: (\d+\.\d+)%', content)
    cer_match = re.search(r'CER: (\d+\.\d+)%', content)
    
    return {
        'WER': float(wer_match.group(1)) if wer_match else None,
        'CER': float(cer_match.group(1)) if cer_match else None
    }

# Compare results across test sets
test_results = {}
for test_set in ['test-clean', 'test-other']:
    wer_file = f'results/experiment/wer_{test_set}.txt'
    test_results[test_set] = parse_wer_file(wer_file)

print("Evaluation Results:")
for test_set, metrics in test_results.items():
    print(f"{test_set}: WER={metrics['WER']:.2f}%, CER={metrics['CER']:.2f}%")
```

### Error Analysis

```python
# Detailed error analysis
def analyze_errors(wer_file):
    \"\"\"Analyze common error patterns.\"\"\"
    with open(wer_file, 'r') as f:
        lines = f.readlines()
    
    # Parse individual utterance results
    errors = []
    for line in lines:
        if 'REF:' in line and 'HYP:' in line:
            # Extract reference and hypothesis
            ref = line.split('REF: ')[1].split(' HYP: ')[0].strip()
            hyp = line.split(' HYP: ')[1].strip()
            errors.append((ref, hyp))
    
    # Analyze error types
    substitutions = sum(1 for ref, hyp in errors if ref != hyp and len(ref.split()) == len(hyp.split()))
    insertions = sum(1 for ref, hyp in errors if len(hyp.split()) > len(ref.split()))
    deletions = sum(1 for ref, hyp in errors if len(hyp.split()) < len(ref.split()))
    
    return {
        'total_errors': len(errors),
        'substitutions': substitutions,
        'insertions': insertions,
        'deletions': deletions
    }

# Analyze errors by test set
for test_set in ['test-clean', 'test-other']:
    wer_file = f'results/experiment/wer_{test_set}.txt'
    error_stats = analyze_errors(wer_file)
    print(f"{test_set} error breakdown: {error_stats}")
```

## Model Deployment

### Model Export for Production

```python
# 1. Export to TorchScript (recommended for production)
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

# Load trained model
with open('hparams/CTC/conformer_large.yaml') as f:
    hparams = load_hyperpyyaml(f, overrides={})

# Initialize model with best checkpoint
model = hparams['modules']['Transformer']
model.load_state_dict(torch.load('results/experiment/averaged_model.pt'))
model.eval()

# Export to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save('mamba_asr_model.pt')
print("Model exported to mamba_asr_model.pt")
```

### Inference Pipeline

```python
# Create inference pipeline
class MambaASRInference:
    def __init__(self, model_path, config_path):
        \"\"\"Initialize Mamba-ASR inference pipeline.\"\"\"
        import torch
        import speechbrain as sb
        from hyperpyyaml import load_hyperpyyaml
        
        # Load configuration
        with open(config_path) as f:
            self.hparams = load_hyperpyyaml(f, overrides={})
        
        # Load model
        self.model = torch.jit.load(model_path)
        self.model.eval()
        
        # Setup feature extraction
        self.feature_extractor = self.hparams['compute_features']
        self.normalizer = self.hparams['modules']['normalize']
        
    def transcribe(self, audio_path):
        \"\"\"Transcribe audio file to text.\"\"\"
        import torchaudio
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Extract features
        features = self.feature_extractor(waveform)
        features = self.normalizer(features, torch.ones(1))
        
        # Model inference
        with torch.no_grad():
            logits = self.model(features)
            
        # Decode (CTC greedy decoding)
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Convert to text (simplified)
        # In practice, use proper CTC decoding or beam search
        text = self.ids_to_text(predicted_ids[0])
        
        return text
    
    def ids_to_text(self, token_ids):
        \"\"\"Convert token IDs to text.\"\"\"
        # Implementation depends on tokenizer type
        # For character-level: map IDs to characters
        # For subword: use tokenizer.decode()
        pass

# Usage
asr = MambaASRInference('mamba_asr_model.pt', 'hparams/CTC/conformer_large.yaml')
transcription = asr.transcribe('path/to/audio.wav')
print(f"Transcription: {transcription}")
```

### API Deployment

```python
# FastAPI deployment example
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
import os

app = FastAPI(title="Mamba-ASR API")

# Initialize model globally
asr_model = MambaASRInference('mamba_asr_model.pt', 'config.yaml')

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    \"\"\"Transcribe uploaded audio file.\"\"\"
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Transcribe
        transcription = asr_model.transcribe(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        return JSONResponse({
            "transcription": transcription,
            "status": "success"
        })
        
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "status": "error"
        }, status_code=500)

@app.get("/health")
async def health_check():
    \"\"\"Health check endpoint.\"\"\"
    return {"status": "healthy"}

# Run with: uvicorn api:app --host 0.0.0.0 --port 8000
```

### Containerized Deployment

```dockerfile
# Dockerfile for Mamba-ASR deployment
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY mamba_asr_model.pt /app/
COPY config.yaml /app/
COPY api.py /app/
COPY inference.py /app/

WORKDIR /app

# Expose port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run container
docker build -t mamba-asr:latest .
docker run -p 8000:8000 --gpus all mamba-asr:latest

# Test API
curl -X POST "http://localhost:8000/transcribe" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_audio.wav"
```

## Monitoring and Maintenance

### Production Monitoring

```python
# Monitoring script for production deployment
import logging
import time
import psutil
import torch
from prometheus_client import start_http_server, Counter, Histogram, Gauge

# Metrics
REQUEST_COUNT = Counter('asr_requests_total', 'Total ASR requests')
REQUEST_DURATION = Histogram('asr_request_duration_seconds', 'ASR request duration')
MODEL_MEMORY = Gauge('asr_model_memory_bytes', 'Model memory usage')
GPU_UTILIZATION = Gauge('asr_gpu_utilization_percent', 'GPU utilization')

class ASRMonitor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.start_time = time.time()
        
    def check_model_health(self):
        \"\"\"Check if model is responsive.\"\"\"
        try:
            # Test inference with dummy input
            dummy_input = torch.randn(1, 1000, 80)  # Example input shape
            with torch.no_grad():
                output = self.model(dummy_input)
            return True
        except Exception as e:
            logging.error(f"Model health check failed: {e}")
            return False
    
    def collect_metrics(self):
        \"\"\"Collect system and model metrics.\"\"\"
        # Memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        MODEL_MEMORY.set(memory_mb)
        
        # GPU utilization (if available)
        if torch.cuda.is_available():
            gpu_util = torch.cuda.utilization()
            GPU_UTILIZATION.set(gpu_util)
    
    def start_monitoring(self, port=8001):
        \"\"\"Start Prometheus metrics server.\"\"\"
        start_http_server(port)
        logging.info(f"Metrics server started on port {port}")
        
        while True:
            self.collect_metrics()
            time.sleep(30)  # Collect metrics every 30 seconds

# Usage
monitor = ASRMonitor('mamba_asr_model.pt')
monitor.start_monitoring()
```

### Model Versioning and Updates

```python
# Model versioning system
import json
import hashlib
import datetime

class ModelVersionManager:
    def __init__(self, models_dir="./models"):
        self.models_dir = models_dir
        self.version_file = f"{models_dir}/versions.json"
        
    def register_model(self, model_path, config_path, metrics):
        \"\"\"Register new model version.\"\"\"
        # Calculate model hash
        with open(model_path, 'rb') as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        
        # Create version info
        version_info = {
            "version": f"v{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "model_hash": model_hash,
            "model_path": model_path,
            "config_path": config_path,
            "metrics": metrics,
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "registered"
        }
        
        # Load existing versions
        try:
            with open(self.version_file, 'r') as f:
                versions = json.load(f)
        except FileNotFoundError:
            versions = []
        
        # Add new version
        versions.append(version_info)
        
        # Save updated versions
        with open(self.version_file, 'w') as f:
            json.dump(versions, f, indent=2)
        
        return version_info["version"]
    
    def deploy_version(self, version):
        \"\"\"Deploy specific model version.\"\"\"
        # Load versions
        with open(self.version_file, 'r') as f:
            versions = json.load(f)
        
        # Find requested version
        target_version = None
        for v in versions:
            if v["version"] == version:
                target_version = v
                break
        
        if not target_version:
            raise ValueError(f"Version {version} not found")
        
        # Update status
        for v in versions:
            v["status"] = "deployed" if v["version"] == version else "registered"
        
        # Save updated versions
        with open(self.version_file, 'w') as f:
            json.dump(versions, f, indent=2)
        
        return target_version

# Usage
version_manager = ModelVersionManager()
version = version_manager.register_model(
    "mamba_asr_model.pt",
    "config.yaml",
    {"WER": 2.85, "CER": 1.2}
)
print(f"Registered model version: {version}")
```

### Performance Optimization

```python
# Production optimization strategies
class ProductionOptimizer:
    def __init__(self, model):
        self.model = model
        
    def optimize_for_inference(self):
        \"\"\"Apply inference optimizations.\"\"\"
        # 1. Set to evaluation mode
        self.model.eval()
        
        # 2. Disable gradient computation globally
        torch.set_grad_enabled(False)
        
        # 3. Use inference mode context
        torch.inference_mode()
        
        # 4. Optimize for inference (if using TorchScript)
        if hasattr(self.model, '_c'):  # TorchScript model
            torch.jit.optimize_for_inference(self.model)
        
        # 5. Set optimal number of threads
        torch.set_num_threads(4)  # Adjust based on CPU cores
        
    def benchmark_performance(self, input_shape=(1, 1000, 80), num_runs=100):
        \"\"\"Benchmark model performance.\"\"\"
        import time
        
        # Warm up
        dummy_input = torch.randn(*input_shape)
        for _ in range(10):
            self.model(dummy_input)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            self.model(dummy_input)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        print(f"Average inference time: {avg_time*1000:.2f}ms")
        print(f"Throughput: {1/avg_time:.2f} inferences/second")
        
        return avg_time

# Usage
optimizer = ProductionOptimizer(model)
optimizer.optimize_for_inference()
avg_time = optimizer.benchmark_performance()
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Training Issues

```python
# Issue: Training loss not decreasing
# Diagnostics:
def diagnose_training():
    # Check learning rate
    print(f"Current LR: {optimizer.param_groups[0]['lr']}")
    
    # Check gradient norms
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    print(f"Gradient norm: {total_norm}")
    
    # Check for NaN/Inf in loss
    if torch.isnan(loss) or torch.isinf(loss):
        print("Warning: Loss is NaN or Inf")

# Solutions:
# - Reduce learning rate: lr_adam: 0.0005 (from 0.001)
# - Increase gradient clipping: max_grad_norm: 1.0 (from 5.0)
# - Check data quality: Verify CSV manifests and audio files
# - Reduce batch size: batch_size: 16 (from 32)
```

#### 2. Memory Issues

```python
# Issue: CUDA out of memory
# Solutions:
memory_solutions = {
    "reduce_batch_size": "batch_size: 8  # Reduce from 32",
    "enable_gradient_checkpointing": "gradient_checkpointing: True",
    "use_dynamic_batching": "dynamic_batching: True",
    "reduce_sequence_length": "max_batch_length_train: 400  # Reduce from 850",
    "use_mixed_precision": "precision: fp16  # Use FP16 instead of FP32",
    "clear_cache": "torch.cuda.empty_cache()  # Clear GPU cache"
}

# Monitor memory usage
def monitor_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
```

#### 3. Device Compatibility Issues

```python
# Issue: MPS/CUDA not working
def debug_device_issues():
    import torch
    
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
    
    print("MPS available:", torch.backends.mps.is_available())
    if torch.backends.mps.is_available():
        print("MPS built:", torch.backends.mps.is_built())

# Solutions:
# - CUDA: Update PyTorch, CUDA drivers
# - MPS: Use macOS 12.3+, native ARM64 Python
# - CPU: Fallback option, reduce num_workers
```

#### 4. Configuration Issues

```yaml
# Common configuration fixes
data_folder: /absolute/path/to/LibriSpeech  # Use absolute paths
output_folder: ./results/unique_experiment  # Avoid conflicts
skip_prep: False  # Regenerate data if corrupted

# Device-specific adjustments
precision: fp32     # Use if bf16/fp16 causes issues
num_workers: 0      # Set to 0 if multiprocessing fails
dynamic_batching: False  # Disable if causing errors
```

### Debugging Tools

```python
# Comprehensive debugging script
def comprehensive_debug():
    \"\"\"Run comprehensive system debugging.\"\"\"
    import torch
    import os
    import psutil
    
    print("=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Platform: {os.uname()}")
    print(f"CPU cores: {os.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    
    print("\\n=== Device Information ===")
    debug_device_issues()
    
    print("\\n=== Data Verification ===")
    # Check if data exists
    data_paths = [
        "data_prepared/train.csv",
        "data_prepared/dev-clean.csv",
        "data_prepared/test-clean.csv"
    ]
    for path in data_paths:
        exists = os.path.exists(path)
        print(f"{path}: {'✓' if exists else '✗'}")
    
    print("\\n=== Model Verification ===")
    try:
        # Test model instantiation
        from hyperpyyaml import load_hyperpyyaml
        with open('hparams/CTC/conformer_large.yaml') as f:
            hparams = load_hyperpyyaml(f, overrides={})
        model = hparams['modules']['Transformer']
        print("Model instantiation: ✓")
    except Exception as e:
        print(f"Model instantiation: ✗ ({e})")

# Run debugging
comprehensive_debug()
```

### Performance Optimization Checklist

```bash
# Pre-training optimization checklist
□ Device selection optimized (CUDA > MPS > CPU)
□ Batch size maximized for available memory
□ Mixed precision enabled (bf16/fp16)
□ Dynamic batching configured
□ Number of workers optimized
□ Gradient accumulation set for target effective batch size

# During training optimization
□ Learning rate schedule appropriate
□ Gradient norms stable (not exploding/vanishing)
□ Memory usage stable (no leaks)
□ Validation metrics improving
□ Checkpoints saving successfully

# Post-training optimization
□ Best checkpoints identified
□ Model averaging applied
□ Evaluation completed on all test sets
□ Results documented and analyzed
□ Model exported for deployment
```

This comprehensive training workflow guide provides complete step-by-step instructions for successfully training and deploying Mamba-ASR models. Each section includes practical code examples, troubleshooting advice, and best practices derived from production experience. The guide follows AI-first documentation principles to ensure the next developer (who is an AI) can successfully navigate the entire machine learning workflow from data preparation through production deployment.