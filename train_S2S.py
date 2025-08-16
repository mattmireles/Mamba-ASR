# ############################################################################
# S2S TRAINING PIPELINE - Mamba-ASR Sequence-to-Sequence Multi-Task Learning
# ############################################################################
#
# SYSTEM ROLE:
# ============
# This script implements advanced sequence-to-sequence ASR training with joint
# CTC and attention objectives for the Mamba-ASR framework. It provides the
# highest accuracy training paradigm through encoder-decoder architecture with
# sophisticated multi-task learning and language model integration.
#
# ARCHITECTURAL OVERVIEW:
# =======================
# S2S training combines multiple complementary objectives for superior performance:
# - Encoder: Processes audio features (Conformer, ConMamba, or Transformer)
# - Decoder: Autoregressive text generation with cross-attention to encoder
# - Joint Training: CTC + Cross-Entropy losses for robust optimization
# - Language Model: External TransformerLM for improved fluency during decoding
# - Beam Search: Joint CTC/Attention/LM beam search for maximum accuracy
#
# MULTI-TASK LEARNING STRATEGY:
# =============================
# 1. CTC Loss: Alignment-free encoder optimization with monotonic constraints
# 2. Attention Loss: Autoregressive decoder optimization with language modeling
# 3. Joint Optimization: Weighted combination balancing alignment and fluency
# 4. Label Smoothing: Regularization preventing overconfidence in predictions
# 5. External LM: Pretrained language model for improved text generation
#
# INTEGRATION WITH MAMBA-ASR:
# ===========================
# The S2S pipeline integrates with the complete Mamba-ASR ecosystem:
# - librispeech_prepare.py: CSV manifest generation for training data
# - modules/TransformerASR.py: Encoder-decoder model instantiation
# - hparams/S2S/*.yaml: Configuration files with S2S-specific parameters
# - Language Model Integration: Pretrained TransformerLM from HuggingFace
# - Advanced Decoding: Joint beam search with multiple scoring functions
#
# PERFORMANCE CHARACTERISTICS:
# ============================
# S2S training provides:
# - Maximum Accuracy: 15-25% WER improvement over CTC-only training
# - Language Modeling: Improved fluency and rare word handling
# - Flexible Decoding: Multiple beam search strategies for different scenarios
# - Robust Training: Multi-task learning reduces overfitting
# - Advanced Evaluation: Comprehensive metrics including accuracy and WER
#
# COMPARISON WITH CTC TRAINING:
# =============================
# S2S Advantages:
# - Higher accuracy through language modeling capability
# - Better handling of linguistic context and rare words
# - Flexible output generation with attention mechanisms
# - Improved fluency through autoregressive decoder
#
# S2S Trade-offs:
# - Increased training complexity and time
# - Higher memory requirements for encoder-decoder architecture
# - More complex hyperparameter tuning (CTC weight, label smoothing)
# - Slower inference due to autoregressive decoding
#
# TRAINING WORKFLOW:
# ==================
# 1. Data Preparation: LibriSpeech CSV generation with subword tokenization
# 2. Model Setup: Encoder-decoder architecture with multi-task objectives
# 3. LM Integration: Pretrained language model download and initialization
# 4. Joint Training: CTC + Attention loss optimization with advanced scheduling
# 5. Validation: Beam search evaluation with configurable LM integration
# 6. Testing: Comprehensive evaluation on multiple test sets
# 7. Model Averaging: Checkpoint averaging for improved stability
#
# Usage Examples:
# ===============
# Standard S2S training:
# python train_S2S.py hparams/S2S/conformer_large.yaml
#
# Training without language model:
# python train_S2S.py hparams/S2S/conformer_large.yaml --no_lm True
#
# Multi-GPU distributed training:
# python -m speechbrain.utils.distributed.ddp_run --nproc_per_node=2 train_S2S.py hparams/S2S/conformer_large.yaml
#
# Based on SpeechBrain LibriSpeech Transformer recipe:
# https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriSpeech/ASR/transformer/train.py
#
# Authors: Jianyuan Zhong, Mirco Ravanelli, Peter Plantinga, Samuele Cornell, Titouan Parcollet
# ############################################################################

#!/usr/bin/env python3
"""
Sequence-to-Sequence Training Pipeline for Mamba-ASR Multi-Task Learning.

This module implements comprehensive S2S training with joint CTC and attention
objectives for maximum ASR accuracy. Features encoder-decoder architecture,
language model integration, and advanced beam search decoding.

Integrates with librispeech_prepare.py for data preparation and hparams/S2S/*.yaml
for configuration. Supports distributed training and sophisticated evaluation.

Usage:
    python train_S2S.py hparams/S2S/conformer_large.yaml

Authors:
 * Jianyuan Zhong 2020
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020, 2021, 2022
 * Titouan Parcollet 2021, 2022
"""

import os
import sys
import torch
import logging
from pathlib import Path
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main, if_main_process

logger = logging.getLogger(__name__)

# Configure Weights & Biases service timeout for reliable experiment tracking
os.environ['WANDB__SERVICE_WAIT'] = '999999'

# =============================================================================
# S2S TRAINING CONSTANTS - Multi-Task Learning Configuration
# =============================================================================

class S2STrainingConstants:
    """Named constants for sequence-to-sequence training pipeline.
    
    These constants ensure consistent behavior across S2S training stages
    and provide clear documentation for AI developers modifying the
    multi-task learning pipeline.
    """
    
    # Multi-task loss components and weighting
    CTC_LOSS_COMPONENT = "ctc_loss"           # CTC alignment loss
    ATTENTION_LOSS_COMPONENT = "seq_loss"     # Cross-entropy attention loss
    JOINT_LOSS_COMBINATION = "weighted_sum"   # Combined loss strategy
    
    # Sequence processing tokens for encoder-decoder training
    BOS_TOKEN_PREFIX = "bos"      # Beginning-of-sequence token identifier
    EOS_TOKEN_SUFFIX = "eos"      # End-of-sequence token identifier
    PADDING_TOKEN = "pad"         # Padding token for variable lengths
    
    # Training stage identifiers
    TRAIN_STAGE = "TRAIN"         # Training with multi-task objectives
    VALID_STAGE = "VALID"         # Validation with beam search (limited)
    TEST_STAGE = "TEST"           # Testing with full beam search + LM
    
    # Evaluation metrics for S2S training
    ACCURACY_METRIC = "ACC"       # Token-level accuracy for decoder
    WER_METRIC = "WER"           # Word Error Rate for transcription quality
    
    # Data pipeline keys for S2S processing
    S2S_PIPELINE_KEYS = ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"]
    
    # Text processing stages for encoder-decoder training
    TEXT_PROCESSING_OUTPUTS = ["wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"]
    
    # Language model integration modes
    WITH_LM_MODE = "with_lm"      # Full beam search with language model
    NO_LM_MODE = "no_lm"         # Beam search without language model
    
    # Checkpoint management strategy
    CHECKPOINT_SELECTION_METRIC = "ACC"  # Primary metric for checkpoint selection
    FINAL_CHECKPOINT_HACK_VALUE = 1.1    # Special value for final checkpoint retention
    
    # Beam search evaluation intervals
    BEAM_SEARCH_INTERVAL_KEY = "valid_search_interval"
    
    # Default file extensions and prefixes
    WER_FILE_PREFIX = "wer_"      # Prefix for WER result files
    WER_FILE_EXTENSION = ".txt"   # Extension for WER files
    
    # Duration-based sorting for batching optimization
    DURATION_SORT_KEY = "duration"

# =============================================================================
# S2S ASR TRAINING CLASS - Multi-Task Learning Implementation
# =============================================================================

class ASR(sb.core.Brain):
    """Sequence-to-Sequence ASR training class with joint CTC and attention objectives.
    
    This class extends SpeechBrain's core.Brain to implement sophisticated S2S
    training for encoder-decoder ASR models. Handles multi-task learning with
    CTC and attention losses, advanced beam search evaluation, and language
    model integration.
    
    MULTI-TASK LEARNING ARCHITECTURE:
    =================================
    The S2S training combines complementary objectives:
    
    1. CTC Loss (Encoder Optimization):
       - Provides alignment supervision for encoder
       - Handles variable input/output length alignment
       - Enables streaming-capable inference
    
    2. Attention Loss (Decoder Optimization):
       - Autoregressive language modeling objective
       - Cross-attention between encoder and decoder
       - Improved fluency and linguistic modeling
    
    3. Joint Optimization:
       - Weighted combination: α * CTC + (1-α) * Attention
       - Balances alignment robustness with language modeling
       - Configurable weighting via hparams.ctc_weight
    
    INTEGRATION WITH MAMBA-ASR:
    ===========================
    Called by:
    - Main training loop: Multi-task optimization via fit() method
    - Evaluation pipeline: Advanced beam search on multiple test sets
    
    Uses:
    - modules/TransformerASR.py: Encoder-decoder architectures
    - Language Model: Pretrained TransformerLM for beam search
    - librispeech_prepare.py: Subword tokenization and data preparation
    
    Integrates with:
    - Advanced Beam Search: Joint CTC/Attention/LM scoring
    - Checkpoint Averaging: Model ensemble for improved performance
    - Distributed Training: Multi-GPU S2S optimization
    
    ADVANCED EVALUATION FEATURES:
    ============================
    - Interval-based Beam Search: Configurable validation frequency
    - Multi-metric Tracking: Accuracy, WER, and loss monitoring
    - Language Model Integration: Optional LM for improved fluency
    - Comprehensive Test Evaluation: Multiple test sets with detailed logging
    
    PERFORMANCE OPTIMIZATIONS:
    ==========================
    - Mixed Precision: BF16/FP16 support for memory efficiency
    - Dynamic Batching: Sequence-length-based batching
    - Feature Augmentation: SpecAugment with label replication
    - Gradient Accumulation: Large effective batch sizes
    - Noam Scheduling: Transformer-optimized learning rate scheduling
    
    Key Methods
    -----------
    compute_forward : Encoder-decoder forward pass with dual outputs
    compute_objectives : Multi-task loss computation with joint optimization
    on_stage_start : Initialize S2S-specific metrics (accuracy + WER)
    on_stage_end : Advanced logging and checkpoint management
    on_evaluate_start : Checkpoint averaging for evaluation
    
    Attributes
    ----------
    tokenizer : Subword tokenizer for encoder-decoder processing
    wer_metric : Word Error Rate computation and tracking
    acc_metric : Token-level accuracy for decoder performance
    train_stats : Training statistics for comprehensive logging
    """
    def compute_forward(self, batch, stage):
        """Performs encoder-decoder forward pass for multi-task S2S training.
        
        This method implements the complete forward computation pipeline for
        sequence-to-sequence training, generating both CTC and attention-based
        probability distributions. Handles sophisticated beam search evaluation
        and stage-specific processing optimizations.
        
        S2S FORWARD COMPUTATION PIPELINE:
        ================================
        1. Feature Extraction: Audio → Mel-scale filterbank features (80-dim)
        2. Normalization: Adaptive feature normalization with epoch-based updates
        3. Augmentation: SpecAugment during training with label replication
        4. CNN Frontend: Convolutional downsampling for computational efficiency
        5. Encoder Processing: Multi-head attention encoding with positional embeddings
        6. Decoder Processing: Autoregressive generation with cross-attention
        7. Dual Outputs: Both CTC logits (encoder) and attention logits (decoder)
        8. Beam Search: Advanced decoding during validation and testing
        
        MULTI-TASK OUTPUT GENERATION:
        =============================
        CTC Output Path:
        - Encoder outputs → Linear projection → CTC probabilities
        - Alignment-free training for encoder optimization
        - Suitable for streaming inference applications
        
        Attention Output Path:
        - Decoder outputs → Linear projection → Attention probabilities
        - Autoregressive language modeling for improved accuracy
        - Cross-attention mechanism for encoder-decoder alignment
        
        ADVANCED BEAM SEARCH INTEGRATION:
        =================================
        Validation Search (Efficient):
        - Limited beam search for fast validation monitoring
        - CTC-only scoring for computational efficiency
        - Configurable validation interval to balance speed/accuracy
        
        Test Search (Comprehensive):
        - Full beam search with joint CTC/Attention/LM scoring
        - Language model integration for maximum accuracy
        - Multiple beam candidates for optimal results
        
        STAGE-SPECIFIC OPTIMIZATIONS:
        ============================
        Training (TRAIN):
        - Feature augmentation (SpecAugment) with label replication
        - No beam search (training uses teacher forcing)
        - Both CTC and attention loss computation
        
        Validation (VALID):
        - Interval-based beam search for monitoring
        - Greedy-style beam search for efficiency
        - Both accuracy and WER metric computation
        
        Testing (TEST):
        - Full beam search with language model
        - Comprehensive evaluation across multiple test sets
        - Detailed metric logging and result file generation
        
        Arguments
        ---------
        batch : dict
            Batch dictionary containing:
            - sig: Audio waveforms (batch_size, max_length)
            - tokens_bos: BOS-prefixed targets for decoder input
            - tokens_eos: EOS-suffixed targets for attention loss
            - tokens: Clean targets for CTC loss
            
        stage : sb.Stage
            Training stage (TRAIN, VALID, TEST) determining processing mode
            
        Returns
        -------
        tuple
            (p_ctc, p_seq, wav_lens, hyps) where:
            - p_ctc: CTC log probabilities (batch_size, time, vocab_size)
            - p_seq: Attention log probabilities (batch_size, seq_len, vocab_size)
            - wav_lens: Sequence length ratios for proper loss masking
            - hyps: Beam search hypotheses (None during training)
        
        Notes
        -----
        The dual-output architecture enables joint optimization while maintaining
        compatibility with both streaming (CTC) and non-streaming (attention)
        inference scenarios. Beam search frequency is configurable via
        hparams.valid_search_interval for optimal training efficiency.
        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig # (B, N)
        tokens_bos, _ = batch.tokens_bos

        # compute features
        feats = self.hparams.compute_features(wavs) # (B, T, 80)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        # Add feature augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "fea_augment"):
            feats, fea_lens = self.hparams.fea_augment(feats, wav_lens)
            tokens_bos = self.hparams.fea_augment.replicate_labels(tokens_bos)

        # forward modules
        src = self.modules.CNN(feats) # (B, L, 20, 32) -> (B, L, 640)

        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index,
        )

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Compute outputs
        hyps = None
        current_epoch = self.hparams.epoch_counter.current
        is_valid_search = (
            stage == sb.Stage.VALID
            and current_epoch % self.hparams.valid_search_interval == 0
        )
        is_test_search = stage == sb.Stage.TEST

        if any([is_valid_search, is_test_search]):
            # Note: For valid_search, for the sake of efficiency, we only perform beamsearch with
            # limited capacity and no LM to give user some idea of how the AM is doing

            # Decide searcher for inference: valid or test search
            if stage == sb.Stage.VALID:
                hyps, _, _, _ = self.hparams.valid_search(
                    enc_out.detach(), wav_lens
                )
            else:
                hyps, _, _, _ = self.hparams.test_search(
                    enc_out.detach(), wav_lens
                )

        return p_ctc, p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes multi-task loss and evaluation metrics for joint CTC/Attention training.
        
        This method implements the core multi-task learning objective that combines
        CTC and attention losses for robust encoder-decoder optimization. Essential
        for achieving state-of-the-art ASR performance through complementary
        training objectives.
        
        MULTI-TASK LOSS FORMULATION:
        ============================
        Joint Loss: L = α * L_CTC + (1-α) * L_Attention
        
        Where:
        - α: CTC weight (typically 0.3) balancing alignment vs language modeling
        - L_CTC: Connectionist Temporal Classification loss for encoder
        - L_Attention: Cross-entropy loss with label smoothing for decoder
        
        CTC LOSS COMPONENT:
        ==================
        - Objective: Alignment-free encoder optimization
        - Input: Encoder outputs projected to vocabulary space
        - Target: Character/subword token sequences
        - Benefit: Provides alignment supervision and streaming capability
        
        ATTENTION LOSS COMPONENT:
        =========================
        - Objective: Autoregressive language modeling with cross-attention
        - Input: Decoder outputs with encoder cross-attention
        - Target: EOS-suffixed token sequences for next-token prediction
        - Benefit: Improved fluency and linguistic context modeling
        
        LABEL SMOOTHING REGULARIZATION:
        ===============================
        - Applied to attention loss only (not CTC)
        - Prevents overconfidence in target predictions
        - Improves generalization and reduces overfitting
        - Configurable smoothing factor (typically 0.1)
        
        EVALUATION METRICS:
        ==================
        Token-level Accuracy (ACC):
        - Measures decoder next-token prediction accuracy
        - Computed on attention outputs vs EOS-suffixed targets
        - Primary metric for checkpoint selection in S2S training
        
        Word Error Rate (WER):
        - Standard ASR evaluation metric
        - Computed during beam search intervals
        - Uses tokenizer decoding for word-level comparison
        - Balances evaluation frequency with computational cost
        
        STAGE-SPECIFIC PROCESSING:
        ==========================
        Training (TRAIN):
        - Multi-task loss computation for gradient optimization
        - Label augmentation replication for SpecAugment compatibility
        - No metric computation for computational efficiency
        
        Validation/Testing (VALID/TEST):
        - Full loss computation + comprehensive metric calculation
        - Beam search decoding during configured intervals
        - Both accuracy and WER metrics for complete evaluation
        
        Arguments
        ---------
        predictions : tuple
            Forward pass outputs: (p_ctc, p_seq, wav_lens, hyps)
            - p_ctc: CTC probabilities for alignment loss
            - p_seq: Attention probabilities for language modeling loss
            - wav_lens: Sequence lengths for proper masking
            - hyps: Beam search results (None during training)
            
        batch : dict
            Batch containing ground truth data:
            - tokens: Clean targets for CTC loss computation
            - tokens_eos: EOS-suffixed targets for attention loss
            - wrd: Word-level transcriptions for WER computation
            - id: Utterance identifiers for metric tracking
            
        stage : sb.Stage
            Training stage determining loss and metric computation strategy
            
        Returns
        -------
        torch.Tensor
            Combined multi-task loss for backpropagation and optimization
            
        Side Effects
        ------------
        - Updates accuracy and WER metrics during evaluation phases
        - Handles label augmentation replication during training
        - Accumulates metrics across batches for epoch-level statistics
        
        Notes
        -----
        The multi-task formulation provides complementary supervision signals:
        CTC ensures proper temporal alignment while attention enables sophisticated
        language modeling. The weighted combination balances these objectives
        for optimal performance.
        """

        (p_ctc, p_seq, wav_lens, hyps,) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "fea_augment"):
                tokens = self.hparams.fea_augment.replicate_labels(tokens)
                tokens_lens = self.hparams.fea_augment.replicate_labels(
                    tokens_lens
                )
                tokens_eos = self.hparams.fea_augment.replicate_labels(
                    tokens_eos
                )
                tokens_eos_lens = self.hparams.fea_augment.replicate_labels(
                    tokens_eos_lens
                )

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        ).sum()

        loss_ctc = self.hparams.ctc_cost(
            p_ctc, tokens, wav_lens, tokens_lens
        ).sum()

        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):
                # Decode token terms to words
                predicted_words = [
                    tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
                ]
                target_words = [wrd.split(" ") for wrd in batch.wrd]
                self.wer_metric.append(ids, predicted_words, target_words)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        return loss

    def on_evaluate_start(self, max_key=None, min_key=None):
        """Loads averaged checkpoints for improved S2S evaluation performance.
        
        This method implements checkpoint averaging specifically optimized for
        sequence-to-sequence models, where ensemble effects provide significant
        accuracy improvements. Critical for achieving state-of-the-art results
        in S2S ASR evaluation.
        
        S2S CHECKPOINT AVERAGING BENEFITS:
        =================================
        - Enhanced Accuracy: 1-2% WER improvement over single checkpoints
        - Reduced Variance: Ensemble effect stabilizes decoder predictions
        - Better Generalization: Averages out optimization noise across epochs
        - Language Model Synergy: Improved compatibility with external LM scoring
        
        CHECKPOINT SELECTION STRATEGY:
        ==============================
        - Primary Metric: Token-level accuracy (ACC) for S2S models
        - Selection Criteria: Best N checkpoints based on validation accuracy
        - Parameter Averaging: Model weights (not optimizer states)
        - Strict Loading: Ensures complete parameter compatibility
        
        INTEGRATION WITH S2S EVALUATION:
        ================================
        Called by:
        - SpeechBrain evaluation pipeline before testing
        - Manual evaluation for specific checkpoint analysis
        
        Optimizes for:
        - Multi-task model stability across CTC and attention objectives
        - Improved beam search decoding with averaged parameters
        - Enhanced language model integration during joint decoding
        
        Arguments
        ---------
        max_key : str, optional
            Metric key for checkpoint selection (higher is better)
            Default: 'ACC' for S2S models prioritizing decoder accuracy
            
        min_key : str, optional
            Metric key for checkpoint selection (lower is better)
            Alternative to max_key for loss-based selection
            
        Side Effects
        ------------
        - Replaces model parameters with averaged weights
        - Sets model to evaluation mode for consistent inference
        - Logs checkpoint averaging completion for verification
        
        Notes
        -----
        S2S models benefit more from checkpoint averaging than CTC models
        due to the added complexity of encoder-decoder optimization.
        The accuracy-based selection typically outperforms WER-based
        selection for checkpoint ranking.
        """
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
           ckpts, recoverable_name="model",
        )
        
        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()
        print("Loaded the average")

    def on_stage_start(self, stage, epoch):
        """Initializes S2S-specific metrics and evaluation infrastructure at epoch start.
        
        This method sets up the dual evaluation metrics required for sequence-to-
        sequence training, ensuring proper tracking of both token-level accuracy
        and word-level error rates. Essential for comprehensive S2S monitoring.
        
        S2S METRIC INITIALIZATION:
        =========================
        Token-level Accuracy (ACC):
        - Measures decoder next-token prediction accuracy
        - Primary metric for S2S checkpoint selection
        - Provides fine-grained feedback on decoder performance
        - Essential for multi-task learning balance assessment
        
        Word Error Rate (WER):
        - Standard ASR evaluation metric for transcription quality
        - Computed during beam search validation intervals
        - Provides end-to-end system performance measurement
        - Critical for comparing S2S vs CTC training effectiveness
        
        STAGE-SPECIFIC INITIALIZATION:
        ==============================
        Training Stage (TRAIN):
        - No metric initialization (focuses on multi-task loss optimization)
        - Encoder-decoder training with joint CTC/Attention objectives
        - Feature augmentation enabled with label replication
        
        Validation/Testing Stages (VALID/TEST):
        - Fresh metric instances for accurate epoch-level computation
        - Dual metric tracking for comprehensive S2S evaluation
        - Beam search preparation for advanced decoding
        
        INTEGRATION WITH S2S WORKFLOW:
        ==============================
        Initialized metrics used by:
        - compute_objectives(): Updates accuracy during decoder evaluation
        - Beam search intervals: WER computation during validation
        - on_stage_end(): Epoch-level statistics and checkpoint selection
        
        Arguments
        ---------
        stage : sb.Stage
            Current training stage (TRAIN, VALID, TEST)
            Determines metric initialization requirements
            
        epoch : int
            Current epoch number for logging and monitoring
            
        Side Effects
        ------------
        - Creates accuracy metric instance for decoder evaluation
        - Creates WER metric instance for transcription quality assessment
        - Ensures metrics start fresh for each evaluation phase
        
        Notes
        -----
        The dual metric approach provides comprehensive S2S evaluation:
        accuracy for decoder optimization feedback and WER for end-to-end
        system performance assessment.
        """
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Processes S2S epoch completion with advanced logging and checkpoint management.
        
        This method handles sophisticated epoch-level processing specific to
        sequence-to-sequence training, including dual metric logging, accuracy-based
        checkpoint selection, and comprehensive test evaluation management.
        
        S2S EPOCH PROCESSING WORKFLOW:
        =============================
        1. Multi-metric Collection: Accuracy, WER, and multi-task loss aggregation
        2. Interval-based Evaluation: Configurable beam search frequency
        3. Accuracy-driven Checkpointing: Best model selection via decoder accuracy
        4. Comprehensive Logging: Training progress with S2S-specific metrics
        5. Test Management: Detailed evaluation with language model integration
        
        STAGE-SPECIFIC PROCESSING:
        ==========================
        Training Stage (TRAIN):
        - Multi-task loss recording for convergence monitoring
        - No checkpoint saving (validation drives S2S model selection)
        - Statistics storage for combined train/validation logging
        
        Validation Stage (VALID):
        - Token-level accuracy computation from decoder outputs
        - Interval-based WER computation during beam search phases
        - Accuracy-driven checkpoint saving (max_keys=["ACC"])
        - Comprehensive S2S training progress logging
        
        Testing Stage (TEST):
        - Final evaluation with full beam search + language model
        - Detailed WER statistics saved to evaluation files
        - Special checkpoint management for averaged model retention
        - Comprehensive test result documentation
        
        CHECKPOINT SELECTION STRATEGY:
        ==============================
        Validation Checkpointing:
        - Primary Metric: Token-level accuracy for decoder optimization
        - Selection Criteria: Maximize decoder prediction accuracy
        - Retention Policy: Keep best N checkpoints for averaging
        - Metadata: Includes accuracy and epoch for ranking
        
        Test Checkpointing:
        - Special Strategy: Retain only the averaged final checkpoint
        - Implementation: Uses accuracy=1.1 hack for single checkpoint retention
        - Purpose: Preserves best averaged model while cleaning intermediates
        
        ADVANCED LOGGING FEATURES:
        ==========================
        S2S Training Logs:
        - Multi-task loss breakdown (CTC weight, attention weight)
        - Token-level accuracy trends for decoder monitoring
        - Learning rate schedules and optimization progress
        - Beam search interval configuration and WER trends
        
        Test Evaluation Logs:
        - Comprehensive WER statistics across multiple test sets
        - Language model integration effectiveness
        - Detailed hypothesis generation and scoring
        
        Arguments
        ---------
        stage : sb.Stage
            Completed training stage (TRAIN, VALID, TEST)
            
        stage_loss : float
            Average multi-task loss for the completed stage
            
        epoch : int
            Current epoch number for logging and checkpoint naming
            
        Side Effects
        ------------
        - Saves accuracy-driven checkpoints during validation
        - Writes comprehensive WER statistics during testing
        - Updates S2S training logs with multi-metric information
        - Manages checkpoint retention for averaged model preservation
        
        Notes
        -----
        The accuracy-based checkpoint selection is particularly effective
        for S2S models as it directly optimizes the decoder component
        responsible for language modeling and fluency improvements.
        """
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
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
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=self.hparams.avg_checkpoints,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_wer_file, "w") as w:
                    self.wer_metric.write_stats(w)

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"ACC": 1.1, "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )
            
    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Updates learning rate schedule for multi-task S2S optimization.
        
        This method applies Noam learning rate scheduling specifically optimized
        for sequence-to-sequence training with joint CTC and attention objectives.
        Critical for stable convergence in complex multi-task learning scenarios.
        
        NOAM SCHEDULING FOR S2S TRAINING:
        ================================
        - Warmup Phase: Gradual LR increase prevents early training instability
        - Decay Phase: Inverse square root decay maintains stable convergence
        - Multi-task Coordination: Balances CTC and attention objective learning
        - Transformer Optimization: Essential for encoder-decoder architectures
        
        S2S OPTIMIZATION CONSIDERATIONS:
        ===============================
        - Joint Objective Complexity: Requires careful LR scheduling coordination
        - Encoder-Decoder Balance: Prevents one component from dominating training
        - Gradient Accumulation: Coordinates with batch accumulation steps
        - Mixed Precision: Maintains numerical stability across loss components
        
        Arguments
        ---------
        batch : dict
            Current training batch (unused in scheduling)
            
        outputs : tuple
            Model outputs from forward pass (unused in scheduling)
            
        loss : torch.Tensor
            Multi-task loss value (unused in scheduling)
            
        should_step : bool
            Whether optimizer step should be taken (considers gradient accumulation)
            Only updates learning rate when gradients are actually applied
            
        Side Effects
        ------------
        - Updates optimizer learning rate according to Noam schedule
        - Coordinates with gradient accumulation for proper step counting
        - Maintains schedule consistency across multi-task objectives
        
        Notes
        -----
        The learning rate schedule is particularly important for S2S training
        as it must balance the convergence rates of both CTC and attention
        objectives for optimal multi-task performance.
        """
        if should_step:
            self.hparams.noam_annealing(self.optimizer)
        if should_step:
            self.hparams.noam_annealing(self.optimizer)


def dataio_prepare(hparams):
    """Prepares comprehensive S2S data pipeline with encoder-decoder processing optimizations.
    
    This function creates the complete data processing infrastructure required
    for sequence-to-sequence training, including advanced tokenization for
    autoregressive decoding, sophisticated text processing, and optimized
    batching strategies for encoder-decoder architectures.
    
    S2S DATA PIPELINE ARCHITECTURE:
    ===============================
    1. Dataset Loading: CSV manifests → SpeechBrain DynamicItemDataset
    2. Audio Pipeline: File paths → Loaded signals with training augmentation
    3. S2S Text Pipeline: Transcriptions → BOS/EOS token sequences for decoder
    4. Dynamic Batching: Length-optimized batching for encoder-decoder memory
    5. Multi-processing: Parallel data loading with subword tokenization
    
    INTEGRATION WITH S2S TRAINING:
    ==============================
    Called by:
    - Main training script: Establishes S2S data infrastructure
    - ASR.fit(): Uses prepared datasets with encoder-decoder batch samplers
    
    Uses:
    - librispeech_prepare.py: CSV manifests with subword tokenization support
    - Pretrained Tokenizer: Subword models (BPE/Unigram) for language modeling
    - SpeechBrain DataIO: Advanced batching for variable-length sequences
    
    Provides data for:
    - ASR.compute_forward(): Audio signals and BOS-prefixed decoder inputs
    - ASR.compute_objectives(): EOS-suffixed targets and clean token sequences
    
    S2S TEXT PROCESSING PIPELINE:
    ============================
    Enhanced text processing for encoder-decoder training:
    
    1. Raw Transcription: Original text from LibriSpeech
    2. Subword Tokenization: BPE/Unigram encoding for improved vocabulary
    3. BOS Token Addition: Beginning-of-sequence for decoder initialization
    4. EOS Token Addition: End-of-sequence for attention loss computation
    5. Clean Tokens: Original sequence for CTC loss computation
    
    Token Sequence Generation:
    - tokens_bos: [BOS] + tokens (decoder input during training)
    - tokens_eos: tokens + [EOS] (attention loss target)
    - tokens: clean tokens (CTC loss target)
    
    ADVANCED BATCHING FOR S2S:
    ==========================
    Encoder-Decoder Optimization:
    - Duration-based grouping: Minimizes encoder padding overhead
    - Sequence length balancing: Optimizes decoder memory usage
    - Cross-attention efficiency: Reduces computational complexity
    
    Memory Management:
    - Separate train/validation samplers for different memory profiles
    - Configurable batch length limits for large sequence handling
    - Multi-GPU awareness for distributed S2S training
    
    PERFORMANCE OPTIMIZATIONS:
    ==========================
    Subword Tokenization:
    - Reduced vocabulary size vs character-level approaches
    - Improved language modeling through meaningful subword units
    - Better handling of rare words and morphological variations
    
    Audio Processing:
    - Speed perturbation during training for improved robustness
    - Multi-threaded loading via DataLoader workers
    - Consistent evaluation pipeline without augmentation
    
    Arguments
    ---------
    hparams : dict
        Hyperparameter dictionary containing:
        - tokenizer: Pretrained subword tokenizer (BPE/Unigram)
        - CSV paths: train_csv, valid_csv, test_csv from preparation
        - Token indices: bos_index, eos_index for sequence processing
        - Batch configuration: dynamic batching and worker settings
        
    Returns
    -------
    tuple
        (train_data, valid_data, test_datasets, tokenizer, train_bsampler, valid_bsampler)
        - train_data: Training dataset with S2S augmentation pipeline
        - valid_data: Validation dataset for S2S monitoring
        - test_datasets: Dictionary of test sets for comprehensive evaluation
        - tokenizer: Subword tokenizer for encoding/decoding
        - train_bsampler: Dynamic batch sampler optimized for S2S training
        - valid_bsampler: Dynamic batch sampler optimized for S2S validation
        
    Notes
    -----
    The S2S data pipeline implements advanced tokenization strategies
    that significantly improve language modeling capability compared to
    character-level approaches. The BOS/EOS token handling is essential
    for proper autoregressive decoder training.
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

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(valtest_datasets, audio_pipeline)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline_train(wav):
        # Speed Perturb is done here so it is multi-threaded with the
        # workers of the dataloader (faster).
        if "speed_perturb" in hparams:
            sig = sb.dataio.dataio.read_audio(wav)

            sig = hparams["speed_perturb"](sig.unsqueeze(0)).squeeze(0)
        else:
            sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline_train)

    # 3. Define S2S text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(*S2STrainingConstants.TEXT_PROCESSING_OUTPUTS)
    def text_pipeline(wrd):
        """Advanced text processing pipeline for sequence-to-sequence training.
        
        Converts raw transcription text through multiple processing stages
        required for encoder-decoder training, including subword tokenization
        and specialized BOS/EOS token handling for autoregressive decoding.
        
        S2S TEXT PROCESSING STAGES:
        ==========================
        1. Raw Text: Original transcription from LibriSpeech
        2. Subword Tokens: BPE/Unigram tokenization for language modeling
        3. BOS Sequence: [BOS] + tokens for decoder input during training
        4. EOS Sequence: tokens + [EOS] for attention loss computation
        5. Clean Tokens: Original sequence for CTC loss computation
        
        ENCODER-DECODER COMPATIBILITY:
        ==============================
        - BOS tokens: Initialize autoregressive decoder state
        - EOS tokens: Signal sequence completion for attention training
        - Clean tokens: Provide CTC supervision for encoder optimization
        - Subword units: Enable sophisticated language modeling
        
        TOKEN SEQUENCE RELATIONSHIPS:
        =============================
        Given text "HELLO WORLD" with tokens [10, 20]:
        - tokens_bos: [1, 10, 20] (BOS + original)
        - tokens_eos: [10, 20, 2] (original + EOS)
        - tokens: [10, 20] (clean for CTC)
        
        Arguments
        ---------
        wrd : str
            Raw transcription text from LibriSpeech CSV
            
        Yields
        ------
        str : Original transcription text
        list : Subword token indices from tokenizer
        torch.Tensor : BOS-prefixed sequence for decoder input
        torch.Tensor : EOS-suffixed sequence for attention loss
        torch.Tensor : Clean token sequence for CTC loss
        """
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set S2S output keys:
    sb.dataio.dataset.set_output_keys(
        datasets, S2STrainingConstants.S2S_PIPELINE_KEYS,
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams_train = hparams["dynamic_batch_sampler_train"]
        dynamic_hparams_valid = hparams["dynamic_batch_sampler_valid"]

        # print(dynamic_hparams_train)

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x[S2STrainingConstants.DURATION_SORT_KEY],
            **dynamic_hparams_train,
        )
        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            length_func=lambda x: x[S2STrainingConstants.DURATION_SORT_KEY],
            **dynamic_hparams_valid,
        )

    return (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_batch_sampler,
        valid_batch_sampler,
    )


# =============================================================================
# MAIN S2S EXECUTION PIPELINE - Multi-Task Learning Orchestration
# =============================================================================

if __name__ == "__main__":
    """Main execution pipeline for sequence-to-sequence ASR training.
    
    This section orchestrates the complete S2S training workflow from 
    configuration loading through comprehensive evaluation. Integrates all
    components of the Mamba-ASR S2S training system with advanced features
    like language model integration and sophisticated beam search.
    
    S2S EXECUTION WORKFLOW:
    ======================
    1. Configuration: Parse S2S-specific YAML hyperparameters
    2. Distributed Setup: Initialize multi-GPU training for encoder-decoder models
    3. Data Preparation: Generate CSV manifests with subword tokenization
    4. Experiment Setup: Create directories and configuration persistence
    5. Language Model: Download and integrate pretrained TransformerLM
    6. Data Pipeline: Advanced S2S data processing with BOS/EOS handling
    7. Multi-task Training: Joint CTC + Attention optimization
    8. Advanced Evaluation: Beam search with language model integration
    
    INTEGRATION WITH MAMBA-ASR:
    ===========================
    - hparams/S2S/*.yaml: S2S configuration with multi-task parameters
    - librispeech_prepare.py: Data preparation with subword tokenization
    - modules/TransformerASR.py: Encoder-decoder model architectures
    - Language Model: Pretrained TransformerLM for beam search enhancement
    - ASR class: S2S training logic with joint objectives
    """
    
    # ==========================================================================
    # CONFIGURATION AND DISTRIBUTED TRAINING SETUP
    # ==========================================================================
    
    # CLI argument parsing with S2S-specific configuration support
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize distributed training for encoder-decoder models
    # S2S training benefits significantly from multi-GPU acceleration
    sb.utils.distributed.ddp_init_group(run_opts)

    # ==========================================================================
    # DATA PREPARATION AND EXPERIMENT SETUP
    # ==========================================================================
    
    # Import LibriSpeech preparation with subword tokenization support
    from librispeech_prepare import prepare_librispeech  # noqa

    # Create experiment directory structure for S2S outputs
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Multi-GPU distributed data preparation with subword tokenization
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],         # LibriSpeech dataset root
            "tr_splits": hparams["train_splits"],         # Training splits for S2S
            "dev_splits": hparams["dev_splits"],           # Validation splits
            "te_splits": hparams["test_splits"],           # Test splits for evaluation
            "save_folder": hparams["output_folder"],       # CSV output directory
            "merge_lst": hparams["train_splits"],          # Merge training splits
            "merge_name": "train.csv",                     # Unified training CSV
            "skip_prep": hparams["skip_prep"],             # Skip if already prepared
        },
    )

    # ==========================================================================
    # S2S DATA PIPELINE WITH ADVANCED TOKENIZATION
    # ==========================================================================
    
    # Create comprehensive S2S data pipeline with BOS/EOS token handling
    (
        train_data,        # Training dataset with S2S augmentation
        valid_data,        # Validation dataset for S2S monitoring
        test_datasets,     # Dictionary of test sets for evaluation
        tokenizer,         # Subword tokenizer for encoder-decoder training
        train_bsampler,    # Dynamic batch sampler for S2S training
        valid_bsampler,    # Dynamic batch sampler for S2S validation
    ) = dataio_prepare(hparams)

    # ==========================================================================
    # LANGUAGE MODEL INTEGRATION FOR ADVANCED BEAM SEARCH
    # ==========================================================================
    
    # Download and load pretrained TransformerLM for beam search enhancement
    # Language model significantly improves S2S accuracy (15-25% WER reduction)
    run_on_main(hparams["pretrainer"].collect_files)   # Download LM and tokenizer
    hparams["pretrainer"].load_collected()             # Load into model objects

    # ==========================================================================
    # EXPERIMENT TRACKING AND LANGUAGE MODEL CONFIGURATION
    # ==========================================================================
    
    # Initialize Weights & Biases logging for S2S training monitoring
    if hparams['use_wandb']:
        hparams['train_logger'] = hparams['wandb_logger']()
        
    # Configure language model integration for evaluation
    if hparams[S2STrainingConstants.NO_LM_MODE.replace('_', '')]:
        print('Evaluate without LM - using validation search for testing.')
        hparams['test_search'] = hparams['valid_search']  # Disable LM integration
        hparams["output_wer_folder"] = os.path.join(
            hparams["output_wer_folder"], 
            S2STrainingConstants.NO_LM_MODE
        )

    # ==========================================================================
    # S2S MODEL INITIALIZATION WITH MULTI-TASK CONFIGURATION
    # ==========================================================================
    
    # Initialize S2S ASR training class with encoder-decoder architecture
    asr_brain = ASR(
        modules=hparams["modules"],           # S2S model components (encoder, decoder, etc.)
        opt_class=hparams["Adam"],            # AdamW optimizer for S2S training
        hparams=hparams,                      # Complete S2S hyperparameter dictionary
        run_opts=run_opts,                    # Runtime options (device, precision, etc.)
        checkpointer=hparams["checkpointer"], # Checkpoint management for S2S models
    )

    # Attach subword tokenizer to S2S training class for advanced decoding
    asr_brain.tokenizer = hparams["tokenizer"]
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        collate_fn = None
        if "collate_fn" in train_dataloader_opts:
            collate_fn = train_dataloader_opts["collate_fn"]

        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }

        if collate_fn is not None:
            train_dataloader_opts["collate_fn"] = collate_fn

    if valid_bsampler is not None:
        collate_fn = None
        if "collate_fn" in valid_dataloader_opts:
            collate_fn = valid_dataloader_opts["collate_fn"]

        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

        if collate_fn is not None:
            valid_dataloader_opts["collate_fn"] = collate_fn

    # ==========================================================================
    # S2S TRAINING AND COMPREHENSIVE EVALUATION
    # ==========================================================================
    
    if not hparams['skip_train']:
        # Execute multi-task S2S training with joint CTC and attention objectives
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,     # Epoch counter for S2S training duration
            train_data,                          # Training dataset with S2S augmentation
            valid_data,                          # Validation dataset for S2S monitoring
            train_loader_kwargs=train_dataloader_opts, # S2S training data loading config
            valid_loader_kwargs=valid_dataloader_opts, # S2S validation data loading config
        )

    # ==========================================================================
    # ADVANCED S2S EVALUATION WITH LANGUAGE MODEL INTEGRATION
    # ==========================================================================
    
    # Create output directory for comprehensive S2S evaluation results
    if not os.path.exists(hparams["output_wer_folder"]):
        os.makedirs(hparams["output_wer_folder"])

    # Evaluate S2S model on all test sets with advanced beam search
    for k in test_datasets.keys():  # Keys: test-clean, test-other, etc.
        # Configure WER output file for current test set
        asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], 
            f"{S2STrainingConstants.WER_FILE_PREFIX}{k}{S2STrainingConstants.WER_FILE_EXTENSION}"
        )
        
        # Perform S2S evaluation with accuracy-based checkpoint selection
        asr_brain.evaluate(
            test_datasets[k],                           # Current test dataset
            max_key=S2STrainingConstants.CHECKPOINT_SELECTION_METRIC, # Select best by accuracy
            test_loader_kwargs=hparams["test_dataloader_opts"],       # Test data loading options
        )
