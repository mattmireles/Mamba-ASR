# ############################################################################
# CTC TRAINING PIPELINE - Mamba-ASR Connectionist Temporal Classification
# ############################################################################
#
# SYSTEM ROLE:
# ============
# This script implements the complete training pipeline for CTC-based automatic
# speech recognition models within the Mamba-ASR framework. It orchestrates
# data preparation, model training, validation, and evaluation for encoder-only
# architectures using Connectionist Temporal Classification.
#
# ARCHITECTURAL INTEGRATION:
# ==========================
# The CTC training pipeline integrates with the Mamba-ASR system through:
# - librispeech_prepare.py: Loads CSV manifests for training data
# - modules/TransformerASR.py: Instantiates encoder models (Conformer, ConMamba, Transformer)
# - hparams/CTC/*.yaml: Configuration files defining model and training parameters
# - SpeechBrain Framework: Core training infrastructure and CTC loss implementation
#
# CTC TRAINING ARCHITECTURE:
# ==========================
# 1. Encoder-Only Design: No autoregressive decoder - direct character prediction
# 2. CTC Alignment: Handles variable input/output length alignment automatically
# 3. Greedy Decoding: Fast inference with greedy CTC decoding for validation
# 4. Beam Search: Advanced CTC beam search for final evaluation accuracy
# 5. Character Tokenization: Direct character-level prediction without subwords
#
# TRAINING WORKFLOW:
# ==================
# 1. Data Preparation: Loads LibriSpeech CSV manifests via librispeech_prepare.py
# 2. Model Instantiation: Creates encoder architecture from hparams configuration
# 3. Data Pipeline: Sets up audio preprocessing, feature extraction, and tokenization
# 4. Training Loop: Implements CTC loss optimization with Noam scheduler
# 5. Validation: Performs greedy CTC decoding and WER/CER computation
# 6. Evaluation: Final testing with CTC beam search for maximum accuracy
# 7. Checkpoint Management: Saves best models based on validation WER
#
# PERFORMANCE OPTIMIZATIONS:
# ==========================
# - Dynamic Batching: Memory-efficient batching based on sequence length
# - Mixed Precision: BF16/FP16 training for reduced memory and faster computation
# - Distributed Training: Multi-GPU support via SpeechBrain's DDP integration
# - Feature Augmentation: SpecAugment and speed perturbation for robustness
# - Checkpoint Averaging: Model averaging across multiple checkpoints
#
# INTEGRATION WITH CONFIGURATION:
# ===============================
# Configuration Flow: hparams/*.yaml → HyperPyYAML → SpeechBrain → Training
# - Model architecture: encoder_module, num_encoder_layers, d_model
# - Training parameters: batch_size, lr_model, number_of_epochs
# - Data paths: train_csv, valid_csv, test_csv from librispeech_prepare.py
# - Optimization: grad_accumulation_factor, max_grad_norm, mixed precision
#
# COMPARISON WITH S2S TRAINING:
# =============================
# CTC Advantages:
# - Faster training: No autoregressive decoder complexity
# - Streaming capability: Suitable for real-time ASR applications
# - Memory efficiency: Lower memory footprint than sequence-to-sequence
# - Alignment-free: No explicit alignment required between audio and text
#
# CTC Limitations:
# - Lower accuracy ceiling: No language modeling capability
# - Independence assumption: Frames predicted independently
# - Limited context: No explicit sequence modeling in output
#
# Usage Examples:
# ===============
# Standard CTC training:
# python train_CTC.py hparams/CTC/conformer_large.yaml
#
# With custom data folder:
# python train_CTC.py hparams/CTC/conformer_large.yaml --data_folder /path/to/LibriSpeech
#
# Multi-GPU training:
# python -m speechbrain.utils.distributed.ddp_run --nproc_per_node=2 train_CTC.py hparams/CTC/conformer_large.yaml
#
# Based on SpeechBrain LibriSpeech CTC recipe:
# https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriSpeech/ASR/CTC/train.py
#
# Authors: Titouan Parcollet, Shucong Zhang, Adel Moumen (Original SpeechBrain implementation)
# ############################################################################

"""
CTC Training Pipeline for Mamba-ASR Encoder-Only Models.

This module implements comprehensive CTC-based ASR training for encoder-only
architectures including Conformer, ConMamba, and Transformer models. Provides
optimized training with dynamic batching, mixed precision, and advanced evaluation.

Integrates with librispeech_prepare.py for data preparation and hparams/CTC/*.yaml
for model configuration. Supports distributed training and comprehensive evaluation.

Usage:
    python train_CTC.py hparams/CTC/conformer_large.yaml

Authors:
 * Titouan Parcollet 2021, 2022
 * Shucong Zhang 2023 
 * Adel Moumen 2024
"""

import logging
import os
import sys
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.distributed import if_main_process, run_on_main

logger = logging.getLogger(__name__)

# =============================================================================
# CTC TRAINING CONSTANTS - Configuration and Optimization Parameters
# =============================================================================

class CTCTrainingConstants:
    """Named constants for CTC training pipeline configuration.
    
    These constants ensure consistent behavior across training stages and
    provide clear documentation for AI developers modifying the training
    pipeline.
    """
    
    # Decoding strategies for different training stages
    GREEDY_DECODING = "greedy"  # Fast validation decoding
    BEAM_SEARCH_DECODING = "beam_search"  # Accurate test decoding
    
    # Stage-specific processing modes
    TRAIN_STAGE = "TRAIN"  # Training with augmentation and CTC loss
    VALID_STAGE = "VALID"  # Validation with greedy decoding
    TEST_STAGE = "TEST"    # Testing with beam search decoding
    
    # Data pipeline keys following SpeechBrain conventions
    DATA_PIPELINE_KEYS = ["id", "sig", "wrd", "char_list", "tokens"]
    
    # Audio pipeline input/output mappings
    AUDIO_INPUT_KEY = "wav"    # Input: audio file path
    AUDIO_OUTPUT_KEY = "sig"   # Output: loaded audio signal
    
    # Text pipeline progression
    TEXT_INPUT_KEY = "wrd"           # Input: transcription text
    TEXT_OUTPUTS = ["wrd", "char_list", "tokens_list", "tokens"]  # Processing stages
    
    # Checkpoint management for model averaging
    DEFAULT_CHECKPOINTS_TO_AVERAGE = 10  # Number of best checkpoints to average
    
    # Dynamic batching optimization
    DURATION_SORT_KEY = "duration"  # Key for length-based batching
    
    # WER file naming convention
    WER_FILE_PREFIX = "wer_"       # Prefix for WER result files
    WER_FILE_EXTENSION = ".txt"    # Extension for WER files


# =============================================================================
# CTC ASR TRAINING CLASS - Main Training Pipeline Implementation
# =============================================================================

class ASR(sb.core.Brain):
    """CTC-based Automatic Speech Recognition training class.
    
    This class extends SpeechBrain's core.Brain to implement CTC training
    for encoder-only ASR models. Handles the complete training workflow
    including forward computation, loss calculation, validation, and evaluation.
    
    INTEGRATION WITH MAMBA-ASR ARCHITECTURE:
    ========================================
    Called by:
    - Main training loop: Orchestrates epoch-based training via fit() method
    - Evaluation pipeline: Performs testing on multiple test sets
    
    Uses:
    - modules/TransformerASR.py: Encoder models (Conformer, ConMamba, Transformer)
    - librispeech_prepare.py: Data preparation and CSV manifest generation
    - hparams/CTC/*.yaml: Model configuration and hyperparameters
    
    Integrates with:
    - SpeechBrain DataIO: Efficient data loading and batching
    - SpeechBrain optimizers: AdamW with Noam learning rate scheduling
    - SpeechBrain checkpointing: Model averaging and best checkpoint selection
    
    CTC TRAINING PIPELINE:
    ======================
    1. Forward Pass: Audio → Features → Encoder → CTC Logits
    2. CTC Loss: Alignment-free loss between logits and character targets
    3. Validation: Greedy CTC decoding for fast WER computation
    4. Testing: Beam search CTC decoding for maximum accuracy
    5. Metrics: Character Error Rate (CER) and Word Error Rate (WER)
    
    PERFORMANCE OPTIMIZATIONS:
    ==========================
    - Mixed Precision: Supports BF16/FP16 for memory efficiency
    - Dynamic Batching: Length-based batching for optimal GPU utilization
    - Feature Augmentation: SpecAugment during training for robustness
    - Checkpoint Averaging: Averages multiple best checkpoints for stability
    - Distributed Training: Multi-GPU support via SpeechBrain DDP
    
    Key Methods
    -----------
    compute_forward : Forward pass from audio to CTC probabilities
    compute_objectives : CTC loss computation and metric calculation
    on_stage_start : Initialize metrics for each training stage
    on_stage_end : Log statistics and save checkpoints
    on_evaluate_start : Load averaged checkpoints for evaluation
    
    Attributes
    ----------
    tokenizer : SentencePiece tokenizer for character-level encoding
    wer_metric : Word Error Rate computation and tracking
    cer_metric : Character Error Rate computation and tracking
    train_stats : Training stage statistics for logging
    """
    def compute_forward(self, batch, stage):
        """Performs forward pass from audio waveforms to CTC probability distributions.
        
        This method implements the complete forward computation pipeline for CTC
        training, including feature extraction, encoder processing, and CTC logit
        generation. Handles stage-specific processing for training, validation,
        and testing phases.
        
        FORWARD COMPUTATION PIPELINE:
        =============================
        1. Feature Extraction: Raw audio → Mel-scale filterbank features (80-dim)
        2. Normalization: Global feature normalization with adaptive statistics
        3. Augmentation: SpecAugment applied during training for robustness
        4. CNN Frontend: Convolutional downsampling for computational efficiency
        5. Encoder Processing: Transformer/Conformer/ConMamba encoding
        6. CTC Projection: Linear layer to vocabulary size for CTC alignment
        7. Log Softmax: Convert logits to log probabilities for CTC loss
        
        STAGE-SPECIFIC PROCESSING:
        ==========================
        Training (TRAIN):
        - Applies feature augmentation (SpecAugment) for robustness
        - No decoding performed (only CTC loss computation)
        
        Validation (VALID):
        - Greedy CTC decoding for fast WER computation
        - No augmentation applied for consistent evaluation
        
        Testing (TEST):
        - Advanced CTC beam search for maximum accuracy
        - Multiple beam candidates for optimal results
        
        MEMORY OPTIMIZATION:
        ====================
        - Mixed precision support: BF16/FP16 for reduced memory usage
        - Dynamic batching: Variable batch sizes based on sequence length
        - Feature normalization: Adaptive statistics updated during training
        
        Arguments
        ---------
        batch : dict
            Batch dictionary containing:
            - sig: Audio waveforms (batch_size, max_length)
            - tokens: Character-level targets for CTC loss
            - wav_lens: Sequence length ratios for padding handling
            
        stage : sb.Stage
            Training stage (TRAIN, VALID, TEST) determining processing mode
            
        Returns
        -------
        tuple
            (p_ctc, wav_lens, p_tokens) where:
            - p_ctc: CTC log probabilities (batch_size, time, vocab_size)
            - wav_lens: Sequence length ratios for loss masking
            - p_tokens: Decoded tokens (None for training, predictions for eval)
        
        Notes
        -----
        The forward pass integrates with modules/TransformerASR.py for encoder
        instantiation and applies stage-specific optimizations for training
        efficiency and evaluation accuracy.
        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        # compute features
        feats = self.hparams.compute_features(wavs) # (B, T, 80)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        # Add feature augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "fea_augment"):
            feats, fea_lens = self.hparams.fea_augment(feats, wav_lens)

        # forward modules
        src = self.modules.CNN(feats)

        enc_out, pred = self.modules.Transformer(
            src, tgt=None, wav_len=wav_lens
        )

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        p_tokens = None
        if stage == sb.Stage.VALID:
            p_tokens = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
        elif stage == sb.Stage.TEST:
            p_tokens = test_searcher(p_ctc, wav_lens)

        return p_ctc, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes CTC loss and evaluation metrics for training optimization.
        
        This method calculates the Connectionist Temporal Classification loss
        and updates evaluation metrics (WER/CER) based on the training stage.
        Essential for gradient computation and model performance monitoring.
        
        CTC LOSS COMPUTATION:
        ====================
        CTC loss handles alignment between:
        - Input: Variable-length encoder outputs (time × vocab_size)
        - Target: Variable-length character sequences
        - Alignment: Automatically learned through CTC forward-backward algorithm
        
        Key CTC Properties:
        - Alignment-free: No explicit time-character alignment required
        - Monotonic: Maintains temporal order in transcription
        - Blank tokens: Special tokens for handling repetitions and silence
        
        EVALUATION METRICS:
        ==================
        Character Error Rate (CER):
        - Character-level accuracy measurement
        - More granular than WER, sensitive to spelling errors
        - Calculated via Levenshtein distance on character sequences
        
        Word Error Rate (WER):
        - Word-level accuracy measurement  
        - Standard ASR evaluation metric
        - Calculated via Levenshtein distance on word sequences
        
        STAGE-SPECIFIC PROCESSING:
        ==========================
        Training (TRAIN):
        - Only CTC loss computation for gradient optimization
        - No metric calculation for computational efficiency
        - Label augmentation applied if feature augmentation enabled
        
        Validation/Testing (VALID/TEST):
        - CTC loss + WER/CER metric calculation
        - Token decoding varies by stage (greedy vs beam search)
        - Metrics accumulated across batches for epoch statistics
        
        Arguments
        ---------
        predictions : tuple
            Forward pass outputs: (p_ctc, wav_lens, predicted_tokens)
            - p_ctc: CTC log probabilities for loss computation
            - wav_lens: Sequence lengths for proper loss masking
            - predicted_tokens: Decoded predictions (None during training)
            
        batch : dict
            Batch containing ground truth data:
            - tokens: Character-level targets for CTC loss
            - wrd: Word-level transcriptions for WER computation
            - id: Utterance IDs for metric tracking
            
        stage : sb.Stage
            Training stage determining metric computation strategy
            
        Returns
        -------
        torch.Tensor
            CTC loss value for backpropagation and optimization
            
        Side Effects
        ------------
        - Updates wer_metric and cer_metric during validation/testing
        - Handles label augmentation replication during training
        - Accumulates metrics across batches for epoch-level statistics
        
        Notes
        -----
        The CTC loss integrates with SpeechBrain's CTC implementation and
        supports both training optimization and comprehensive evaluation
        across multiple test sets.
        """

        p_ctc, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens, tokens_lens = batch.tokens

        old_tokens = tokens.detach().clone()
        old_tokens_lens = tokens_lens.detach().clone()

        # Label Augmentation
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "fea_augment"):
            tokens = self.hparams.fea_augment.replicate_labels(tokens)
            tokens_lens = self.hparams.fea_augment.replicate_labels(tokens_lens)

        loss = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens).sum()

        if stage == sb.Stage.VALID:
            # Decode token terms to words
            predicted_words = self.tokenizer(
                predicted_tokens, task="decode_from_list"
            )
        elif stage == sb.Stage.TEST:
            predicted_words = [
                hyp[0].text.split(" ") for hyp in predicted_tokens
            ]

        if stage != sb.Stage.TRAIN:
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def on_evaluate_start(self, max_key=None, min_key=None):
        """Loads averaged checkpoints before evaluation for improved performance.
        
        This method implements checkpoint averaging, a critical technique for
        improving model performance by combining multiple best checkpoints.
        Called automatically before validation and testing phases.
        
        CHECKPOINT AVERAGING BENEFITS:
        ==============================
        - Reduced Variance: Averages out optimization noise from individual checkpoints
        - Improved Accuracy: Typically provides 0.5-1.0% WER improvement
        - Stability: More robust performance across different evaluation sets
        - Ensemble Effect: Captures complementary model behaviors from different epochs
        
        AVERAGING STRATEGY:
        ==================
        - Checkpoint Selection: Uses best N checkpoints based on validation WER
        - Parameter Averaging: Averages model weights (not gradients or optimizer state)
        - Strict Loading: Ensures complete parameter matching for averaged model
        - Evaluation Mode: Sets model to eval() for consistent inference behavior
        
        INTEGRATION WITH TRAINING:
        ==========================
        Called by:
        - SpeechBrain evaluation pipeline: Automatic checkpoint loading
        - Manual evaluation: Direct method invocation for custom evaluation
        
        Uses:
        - Checkpointer: Finds and ranks checkpoints by specified metric
        - Model averaging: SpeechBrain utility for parameter averaging
        
        Arguments
        ---------
        max_key : str, optional
            Metric key for selecting checkpoints (higher is better)
            Example: 'accuracy' for accuracy-based selection
            
        min_key : str, optional
            Metric key for selecting checkpoints (lower is better)
            Example: 'WER' for WER-based selection (default for ASR)
            
        Side Effects
        ------------
        - Modifies model parameters with averaged weights
        - Sets model to evaluation mode
        - Logs checkpoint averaging completion
        
        Notes
        -----
        The number of checkpoints to average is controlled by hparams.avg_checkpoints
        (typically 10). Checkpoint selection prioritizes validation performance
        over training metrics for better generalization.
        """
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts,
            recoverable_name="model",
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()
        print("Loaded the average")

    def on_stage_start(self, stage, epoch):
        """Initializes metrics and stage-specific configurations at epoch start.
        
        This method sets up the evaluation infrastructure for each training stage,
        ensuring proper metric initialization and stage-specific optimizations.
        Called automatically by SpeechBrain at the beginning of each epoch.
        
        STAGE-SPECIFIC INITIALIZATION:
        ==============================
        Training Stage (TRAIN):
        - No metric initialization (training focuses on loss optimization)
        - Feature augmentation enabled for robustness
        - Model in training mode for gradient computation
        
        Validation/Testing Stages (VALID/TEST):
        - Fresh metric instances for accurate epoch-level computation
        - Character Error Rate (CER) metric initialization
        - Word Error Rate (WER) metric initialization
        - Model in evaluation mode for consistent inference
        
        METRIC INITIALIZATION STRATEGY:
        ===============================
        - Fresh Instances: New metric objects prevent contamination across epochs
        - Accumulated Statistics: Metrics accumulate across batches within epoch
        - Thread Safety: Metric objects handle distributed training correctly
        
        INTEGRATION WITH EVALUATION:
        ============================
        Initialized metrics used by:
        - compute_objectives(): Updates metrics during batch processing
        - on_stage_end(): Summarizes metrics for epoch-level statistics
        - Logging: Provides epoch-level WER/CER for monitoring
        
        Arguments
        ---------
        stage : sb.Stage
            Current training stage (TRAIN, VALID, TEST)
            Determines metric initialization requirements
            
        epoch : int
            Current epoch number for logging and checkpoint management
            
        Side Effects
        ------------
        - Creates new CER and WER metric instances for evaluation stages
        - Ensures metrics start fresh for each evaluation phase
        - Prevents metric contamination between epochs
        
        Notes
        -----
        Metric initialization follows SpeechBrain conventions and integrates
        with the distributed training framework for multi-GPU scenarios.
        """
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.wer_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Processes epoch completion with logging, checkpointing, and metric reporting.
        
        This method handles epoch-level statistics, checkpoint management, and
        comprehensive logging for training monitoring and model selection.
        Called automatically by SpeechBrain at the end of each training stage.
        
        EPOCH PROCESSING WORKFLOW:
        ==========================
        1. Statistics Collection: Aggregates stage-specific metrics and loss
        2. Metric Summarization: Computes final WER/CER from accumulated batches
        3. Checkpoint Management: Saves best models based on validation performance
        4. Logging: Records training progress for monitoring and analysis
        5. Model Selection: Maintains best checkpoints for final evaluation
        
        STAGE-SPECIFIC PROCESSING:
        ==========================
        Training Stage (TRAIN):
        - Records training loss for convergence monitoring
        - No checkpoint saving (validation drives model selection)
        - Stores statistics for combined train/validation logging
        
        Validation Stage (VALID):
        - Computes WER/CER from accumulated validation metrics
        - Saves checkpoints based on validation WER improvement
        - Logs comprehensive training progress (loss, WER, learning rate)
        - Maintains best N checkpoints for model averaging
        
        Testing Stage (TEST):
        - Final evaluation with comprehensive metric reporting
        - Saves detailed WER statistics to file for analysis
        - No checkpoint saving (uses pre-selected best models)
        
        CHECKPOINT MANAGEMENT STRATEGY:
        ===============================
        - Best Model Selection: Saves checkpoints with lowest validation WER
        - Model Averaging: Keeps configurable number of best checkpoints
        - Metadata Storage: Includes WER and epoch information for ranking
        - Automatic Cleanup: Removes older checkpoints to save disk space
        
        LOGGING AND MONITORING:
        =======================
        Logged Information:
        - Training/Validation Loss: Convergence monitoring
        - WER/CER Metrics: Model performance tracking
        - Learning Rate: Optimization schedule monitoring
        - Optimizer Steps: Training progress quantification
        
        Arguments
        ---------
        stage : sb.Stage
            Completed training stage (TRAIN, VALID, TEST)
            
        stage_loss : float
            Average loss for the completed stage
            
        epoch : int
            Current epoch number for logging and checkpoint naming
            
        Side Effects
        ------------
        - Saves model checkpoints during validation
        - Writes WER statistics to files during testing
        - Updates training logs for monitoring
        - Maintains checkpoint ranking based on validation performance
        
        Notes
        -----
        Checkpoint selection prioritizes validation WER over training loss
        for better generalization. The number of maintained checkpoints
        is configurable via hparams.avg_checkpoints.
        """
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID:

            lr = self.hparams.noam_annealing.current_lr
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"], "epoch": epoch},
                min_keys=["WER"],
                num_to_keep=self.hparams.avg_checkpoints,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.wer_file, "w") as w:
                    self.wer_metric.write_stats(w)

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Updates learning rate schedule after each training batch.
        
        This method implements the Noam learning rate scheduling strategy,
        critical for transformer-based model convergence. Called automatically
        by SpeechBrain after each training batch.
        
        NOAM SCHEDULER INTEGRATION:
        ===========================
        - Warmup Phase: Gradually increases learning rate from 0 to peak
        - Decay Phase: Decreases learning rate following 1/sqrt(step) schedule
        - Transformer Optimization: Essential for stable transformer training
        - Gradient Accumulation: Coordinates with batch accumulation steps
        
        LEARNING RATE SCHEDULE:
        ======================
        lr(step) = lr_initial * min(step^(-0.5), step * warmup_steps^(-1.5))
        
        Where:
        - lr_initial: Base learning rate from hparams
        - warmup_steps: Number of warmup steps (typically 7500-30000)
        - step: Current optimizer step (considers gradient accumulation)
        
        Arguments
        ---------
        batch : dict
            Current training batch (unused in this method)
            
        outputs : tuple
            Model outputs from forward pass (unused in this method)
            
        loss : torch.Tensor
            Computed loss value (unused in this method)
            
        should_step : bool
            Whether optimizer step should be taken (considers gradient accumulation)
            Only updates learning rate when gradients are actually applied
            
        Side Effects
        ------------
        - Updates optimizer learning rate according to Noam schedule
        - Coordinates with gradient accumulation for proper step counting
        
        Notes
        -----
        The should_step parameter ensures learning rate updates align with
        actual optimizer steps, accounting for gradient accumulation delays.
        """
        if should_step:
            self.hparams.noam_annealing(self.optimizer)


def dataio_prepare(hparams, tokenizer):
    """Prepares comprehensive data pipeline for CTC training with optimized processing.
    
    This function creates the complete data processing infrastructure required
    for CTC training, including dataset loading, preprocessing pipelines, and
    dynamic batching strategies. Essential for efficient training and evaluation.
    
    DATA PIPELINE ARCHITECTURE:
    ===========================
    1. Dataset Loading: CSV manifests → SpeechBrain DynamicItemDataset
    2. Audio Pipeline: File paths → Loaded audio signals with augmentation
    3. Text Pipeline: Transcriptions → Character tokens for CTC training
    4. Dynamic Batching: Length-based batching for memory optimization
    5. Multi-processing: Parallel data loading for I/O efficiency
    
    INTEGRATION WITH TRAINING SYSTEM:
    =================================
    Called by:
    - Main training script: Sets up data infrastructure before training
    - ASR.fit(): Uses prepared datasets and batch samplers for training
    
    Uses:
    - librispeech_prepare.py: Generated CSV manifests for data loading
    - SpeechBrain DataIO: Efficient dataset and batching infrastructure
    - SentencePiece tokenizer: Character-level tokenization for CTC
    
    Provides data for:
    - ASR.compute_forward(): Audio signals and metadata
    - ASR.compute_objectives(): Ground truth tokens and transcriptions
    
    DATASET ORGANIZATION:
    ====================
    Training Data:
    - Multiple LibriSpeech splits combined for comprehensive training
    - Speed perturbation applied for robustness
    - Sorting strategies for computational efficiency
    
    Validation Data:
    - Single development set for consistent monitoring
    - Duration-sorted for predictable batch processing
    - No augmentation for reliable evaluation
    
    Test Data:
    - Multiple test sets (clean, other) for comprehensive evaluation
    - Duration-sorted for consistent processing
    - Separate evaluation for different acoustic conditions
    
    PERFORMANCE OPTIMIZATIONS:
    ==========================
    Dynamic Batching:
    - Length-based grouping minimizes padding overhead
    - Configurable batch length limits for memory management
    - Separate samplers for training and validation
    
    Multi-processing:
    - Parallel audio loading via DataLoader workers
    - Speed perturbation in worker threads for efficiency
    - Optimized I/O pipeline for large-scale training
    
    Memory Efficiency:
    - Lazy loading prevents memory overflow
    - Efficient padding strategies for variable lengths
    - Configurable batch sizes based on available memory
    
    AUDIO PROCESSING PIPELINE:
    ==========================
    Training Audio:
    - Speed perturbation (95%, 100%, 105%) for robustness
    - Multi-threaded processing via DataLoader workers
    - Dynamic augmentation for improved generalization
    
    Validation/Test Audio:
    - Standard loading without augmentation
    - Consistent processing for reliable evaluation
    - Deterministic pipeline for reproducible results
    
    TEXT PROCESSING PIPELINE:
    =========================
    Text → Characters → Token IDs → Tensors:
    1. Raw transcription text from CSV manifests
    2. Character-level segmentation for CTC compatibility
    3. SentencePiece encoding to token indices
    4. PyTorch tensor conversion for training
    
    Arguments
    ---------
    hparams : dict
        Hyperparameter dictionary containing:
        - data_folder: Root path to LibriSpeech dataset
        - CSV paths: train_csv, valid_csv, test_csv from preparation
        - Batch configuration: sizes, workers, dynamic batching settings
        - Sorting strategy: random, ascending, descending
        
    tokenizer : SentencePiece
        Character-level tokenizer for CTC training.
        Provides encoding/decoding between text and token indices
        
    Returns
    -------
    tuple
        (train_data, valid_data, test_datasets, train_bsampler, valid_bsampler)
        - train_data: Training dataset with augmentation pipeline
        - valid_data: Validation dataset for monitoring
        - test_datasets: Dictionary of test sets for evaluation
        - train_bsampler: Dynamic batch sampler for training
        - valid_bsampler: Dynamic batch sampler for validation
        
    Notes
    -----
    The function implements SpeechBrain's data pipeline conventions with
    optimizations specific to CTC training requirements. Dynamic batching
    is optional but recommended for memory efficiency with variable-length
    sequences.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]
    valtest_datasets = [valid_data] + [i for k, i in test_datasets.items()]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes(CTCTrainingConstants.AUDIO_INPUT_KEY)
    @sb.utils.data_pipeline.provides(CTCTrainingConstants.AUDIO_OUTPUT_KEY)
    def audio_pipeline(wav):
        """Standard audio loading pipeline for validation and testing.
        
        Loads audio files without augmentation for consistent evaluation.
        Used for validation and test datasets where reproducible results
        are required.
        
        Arguments
        ---------
        wav : str
            Path to audio file (FLAC format for LibriSpeech)
            
        Returns
        -------
        torch.Tensor
            Loaded audio signal as PyTorch tensor
        """
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(valtest_datasets, audio_pipeline)

    @sb.utils.data_pipeline.takes(CTCTrainingConstants.AUDIO_INPUT_KEY)
    @sb.utils.data_pipeline.provides(CTCTrainingConstants.AUDIO_OUTPUT_KEY)
    def audio_pipeline_train(wav):
        """Training audio pipeline with speed perturbation augmentation.
        
        Applies speed perturbation during training for improved robustness.
        Multi-threaded execution via DataLoader workers provides efficient
        augmentation without blocking the main training loop.
        
        SPEED PERTURBATION AUGMENTATION:
        ================================
        - Speeds: 95%, 100%, 105% (configurable via hparams)
        - Benefits: Improved robustness to speaking rate variations
        - Implementation: Random selection per training sample
        - Performance: Multi-threaded via DataLoader workers
        
        Arguments
        ---------
        wav : str
            Path to audio file for training
            
        Returns
        -------
        torch.Tensor
            Loaded and potentially speed-perturbed audio signal
        """
        # Speed Perturb is done here so it is multi-threaded with the
        # workers of the dataloader (faster).
        if hparams["speed_perturb"]:
            sig = sb.dataio.dataio.read_audio(wav)
            sig = hparams["speed_perturb"](sig.unsqueeze(0)).squeeze(0)
        else:
            sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline_train)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes(CTCTrainingConstants.TEXT_INPUT_KEY)
    @sb.utils.data_pipeline.provides(*CTCTrainingConstants.TEXT_OUTPUTS)
    def text_pipeline(wrd):
        """Text processing pipeline for CTC training preparation.
        
        Converts raw transcription text through multiple processing stages
        required for CTC training, including character segmentation and
        tokenization via SentencePiece.
        
        TEXT PROCESSING STAGES:
        ======================
        1. Raw Text: Original transcription from LibriSpeech
        2. Character List: Character-level segmentation for analysis
        3. Token List: SentencePiece encoding to token indices
        4. Token Tensor: PyTorch tensor for training/evaluation
        
        CTC COMPATIBILITY:
        ==================
        - Character-level: CTC requires character or subword tokens
        - Variable length: CTC handles variable input/output lengths
        - Blank tokens: SentencePiece tokenizer includes CTC blank token
        
        Arguments
        ---------
        wrd : str
            Raw transcription text from LibriSpeech CSV
            
        Yields
        ------
        str : Original transcription text
        list : Character-level segmentation
        list : SentencePiece token indices
        torch.Tensor : Token tensor for training
        """
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, CTCTrainingConstants.DATA_PIPELINE_KEYS,
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams_train = hparams["dynamic_batch_sampler_train"]
        dynamic_hparams_val = hparams["dynamic_batch_sampler_val"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x[CTCTrainingConstants.DURATION_SORT_KEY],
            **dynamic_hparams_train,
        )

        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            length_func=lambda x: x[CTCTrainingConstants.DURATION_SORT_KEY],
            **dynamic_hparams_val,
        )

    return (
        train_data,
        valid_data,
        test_datasets,
        train_batch_sampler,
        valid_batch_sampler,
    )


# =============================================================================
# MAIN EXECUTION PIPELINE - CTC Training Orchestration
# =============================================================================

if __name__ == "__main__":
    """Main execution pipeline for CTC-based ASR training.
    
    This section orchestrates the complete training workflow from command-line
    argument parsing through final model evaluation. Integrates all components
    of the Mamba-ASR CTC training system.
    
    EXECUTION WORKFLOW:
    ==================
    1. Configuration Loading: Parse YAML hyperparameters and CLI overrides
    2. Distributed Setup: Initialize multi-GPU training if specified
    3. Data Preparation: Generate CSV manifests via librispeech_prepare.py
    4. Experiment Setup: Create output directories and save configuration
    5. Tokenizer Creation: Initialize SentencePiece for character encoding
    6. Data Pipeline: Set up training, validation, and test datasets
    7. Model Training: Execute training loop with validation monitoring
    8. Model Evaluation: Test on multiple evaluation sets with best models
    
    INTEGRATION WITH MAMBA-ASR:
    ===========================
    - hparams/CTC/*.yaml: Configuration file specifying model and training parameters
    - librispeech_prepare.py: Data preparation generating required CSV manifests
    - modules/TransformerASR.py: Model architectures instantiated via configuration
    - ASR class: Training logic implementing CTC optimization and evaluation
    """
    
    # ==========================================================================
    # CONFIGURATION AND DISTRIBUTED TRAINING SETUP
    # ==========================================================================
    
    # CLI argument parsing with SpeechBrain framework
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize distributed training if --distributed_launch specified
    # Creates DDP communication group with appropriate backend (NCCL/Gloo)
    sb.utils.distributed.ddp_init_group(run_opts)

    # ==========================================================================
    # DATA PREPARATION AND EXPERIMENT SETUP
    # ==========================================================================
    
    # Import data preparation function for LibriSpeech processing
    from librispeech_prepare import prepare_librispeech  # noqa

    # Create experiment directory structure for outputs and logging
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Multi-GPU distributed data preparation (run only on main process)
    # Prevents multiple processes from simultaneously preparing the same data
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],         # LibriSpeech root directory
            "tr_splits": hparams["train_splits"],         # Training splits to prepare
            "dev_splits": hparams["dev_splits"],           # Validation splits to prepare  
            "te_splits": hparams["test_splits"],           # Test splits to prepare
            "save_folder": hparams["output_folder"],       # Output directory for CSV files
            "merge_lst": hparams["train_splits"],          # Merge training splits into single CSV
            "merge_name": "train.csv",                     # Name for merged training CSV
            "skip_prep": hparams["skip_prep"],             # Skip if data already prepared
        },
    )

    # ==========================================================================
    # TOKENIZER INITIALIZATION FOR CHARACTER-LEVEL CTC TRAINING
    # ==========================================================================
    
    # Initialize SentencePiece tokenizer for character-level encoding
    # CTC training requires character or subword tokenization for alignment
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],                    # Directory for tokenizer model
        vocab_size=hparams["output_neurons"],               # Vocabulary size (31 for char-level)
        annotation_train=hparams["train_csv"],              # Training text for tokenizer fitting
        annotation_read=CTCTrainingConstants.TEXT_INPUT_KEY, # Column name for text in CSV
        model_type=hparams["token_type"],                   # Tokenization strategy (char/bpe/unigram)
        character_coverage=hparams["character_coverage"],   # Character coverage for vocabulary
        bos_id=hparams["bos_index"],                        # Beginning-of-sequence token ID
        eos_id=hparams["eos_index"],                        # End-of-sequence token ID
    )

    # ==========================================================================
    # DATA PIPELINE SETUP WITH OPTIMIZED BATCHING
    # ==========================================================================
    
    # Create comprehensive data pipeline with datasets, preprocessing, and batching
    # Includes audio/text processing, dynamic batching, and multi-processing setup
    (
        train_data,        # Training dataset with augmentation
        valid_data,        # Validation dataset for monitoring
        test_datasets,     # Dictionary of test sets for evaluation
        train_bsampler,    # Dynamic batch sampler for training
        valid_bsampler,    # Dynamic batch sampler for validation
    ) = dataio_prepare(hparams, tokenizer)

    # ==========================================================================
    # EXPERIMENT TRACKING SETUP
    # ==========================================================================
    
    # Initialize Weights & Biases logging if enabled in configuration
    if hparams['use_wandb']:
        hparams['train_logger'] = hparams['wandb_logger']()

    # ==========================================================================
    # MODEL INITIALIZATION AND CTC DECODER SETUP
    # ==========================================================================
    
    # Initialize ASR training class with SpeechBrain Brain framework
    asr_brain = ASR(
        modules=hparams["modules"],           # Model components (CNN, Transformer, etc.)
        opt_class=hparams["model_opt_class"], # Optimizer class (AdamW)
        hparams=hparams,                      # Complete hyperparameter dictionary
        run_opts=run_opts,                    # Runtime options (device, precision, etc.)
        checkpointer=hparams["checkpointer"], # Checkpoint management for model saving
    )

    # Attach tokenizer to training class for decoding
    asr_brain.tokenizer = tokenizer
    
    # Create vocabulary list for CTC beam search decoder
    vocab_list = [
        tokenizer.sp.id_to_piece(i) for i in range(tokenizer.sp.vocab_size())
    ]

    # Import CTC beam search decoder for final evaluation
    from speechbrain.decoders.ctc import CTCBeamSearcher

    # Initialize CTC beam search decoder for test-time evaluation
    test_searcher = CTCBeamSearcher(
        **hparams["test_beam_search"],  # Beam search parameters (beam_size, etc.)
        vocab_list=vocab_list,          # Token vocabulary for decoding
    )

    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }

    if valid_bsampler is not None:
        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

    # ==========================================================================
    # TRAINING AND EVALUATION EXECUTION
    # ==========================================================================
    
    if not hparams['skip_train']:
        # Execute main training loop with validation monitoring
        # Implements CTC loss optimization with Noam scheduling and checkpointing
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,     # Epoch counter for training duration
            train_data,                          # Training dataset with augmentation
            valid_data,                          # Validation dataset for monitoring
            train_loader_kwargs=train_dataloader_opts, # Training data loading configuration
            valid_loader_kwargs=valid_dataloader_opts, # Validation data loading configuration
        )

    # ==========================================================================
    # COMPREHENSIVE MODEL EVALUATION
    # ==========================================================================
    
    # Evaluate trained model on all test sets with detailed metric reporting
    for k in test_datasets.keys():  # Keys: test-clean, test-other, etc.
        # Configure WER output file for current test set
        asr_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], 
            f"{CTCTrainingConstants.WER_FILE_PREFIX}{k}{CTCTrainingConstants.WER_FILE_EXTENSION}"
        )
        
        # Perform evaluation with checkpoint averaging and beam search
        asr_brain.evaluate(
            test_datasets[k],                           # Current test dataset
            min_key="WER",                             # Select best checkpoint by WER
            test_loader_kwargs=hparams["test_dataloader_opts"], # Test data loading options
        )
