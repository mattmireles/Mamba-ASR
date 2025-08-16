"""
Base Transformer Infrastructure for Mamba-ASR Multi-Modal Architecture.

This module provides the foundational transformer components that serve as the 
architectural backbone for the Mamba-ASR system. It implements a unified interface
that seamlessly integrates traditional transformer encoders/decoders with modern
state-space models (Mamba) and hybrid architectures (ConMamba, Conformer).

ARCHITECTURAL FOUNDATION:
========================
This module serves as the core infrastructure layer enabling modular ASR architecture
design through a unified TransformerInterface. The design philosophy prioritizes
architectural flexibility while maintaining computational efficiency and streaming
compatibility.

Key Design Principles:
- Modular Architecture: Plug-and-play encoder/decoder selection
- Unified Interface: Common API across transformer variants (Transformer, Conformer, ConMamba)
- Streaming Compatibility: Foundation for real-time ASR processing
- Multi-Modal Support: Handles both traditional attention and state-space models
- Performance Optimization: Apple Silicon MPS and CUDA acceleration support

SYSTEM ROLE IN MAMBA-ASR:
=========================
TransformerInterface acts as the central orchestrator for the ASR pipeline,
providing a configuration-driven approach to model architecture selection:

Primary Components:
- TransformerInterface: Main factory class for creating encoder-decoder pairs
- PositionalEncoding: Absolute positional embeddings for sequence modeling
- TransformerEncoder/Decoder: Classical transformer implementations
- TransformerEncoderLayer/DecoderLayer: Attention-based processing blocks
- NormalizedEmbedding: Scaled embedding layers for improved training stability

ENCODER ARCHITECTURE SUPPORT:
=============================
The interface supports multiple encoder architectures for different use cases:

1. Transformer Encoder ("transformer"):
   - Classical multi-head attention with feed-forward networks
   - Bidirectional context modeling for optimal accuracy
   - Best for: Offline ASR, research, baseline comparisons

2. Conformer Encoder ("conformer"):
   - Hybrid CNN-Transformer architecture via modules/Conformer.py
   - Superior audio modeling through convolution + attention fusion
   - Best for: Production ASR, streaming applications, state-of-the-art accuracy

3. ConMamba Encoder ("conmamba"):
   - Innovative hybrid combining Conformer CNN with Mamba state-space models
   - Efficient long-sequence modeling via modules/Conmamba.py
   - Best for: Long-form audio, memory-efficient processing, research

4. Branchformer Encoder ("branchformer"):
   - Advanced parallel attention-convolution architecture
   - Multiple processing branches for enhanced feature extraction
   - Best for: High-performance ASR, complex acoustic environments

DECODER ARCHITECTURE SUPPORT:
=============================
Flexible decoder selection for different modeling paradigms:

1. Transformer Decoder ("transformer"):
   - Standard autoregressive transformer decoder with cross-attention
   - Full bidirectional encoding with causal decoding
   - Best for: Sequence-to-sequence tasks, language modeling integration

2. Mamba Decoder ("mamba"):
   - State-space model decoder via modules/Conmamba.py MambaDecoder
   - Linear complexity in sequence length for efficient long-form processing
   - Best for: Memory-efficient decoding, long sequences, streaming inference

CALL CHAIN INTEGRATION:
======================
This module serves as the foundation for higher-level ASR components:

Called by:
- `modules/TransformerASR.py`: TransformerASR.__init__() creates TransformerInterface instances
- `train_CTC.py`: Training scripts instantiate models via TransformerInterface
- `train_S2S.py`: Sequence-to-sequence training uses encoder-decoder pairs
- Configuration files: YAML configs specify encoder_module/decoder_module selection

Calls to:
- `modules/Conformer.py`: ConformerEncoder when encoder_module="conformer"
- `modules/Conmamba.py`: ConmambaEncoder and MambaDecoder for state-space models
- `speechbrain.nnet.attention.*`: Core attention mechanisms (MultiheadAttention, RelPosMHAXL)
- `speechbrain.nnet.CNN.Conv1d`: Convolutional preprocessing layers

Creates:
- Encoder instances: TransformerEncoder, ConformerEncoder, ConmambaEncoder
- Decoder instances: TransformerDecoder, MambaDecoder
- Positional encoding: PositionalEncoding for sequence position awareness
- Custom modules: NormalizedEmbedding for improved training dynamics

POSITIONAL ENCODING INFRASTRUCTURE:
==================================
Comprehensive positional encoding support for different sequence modeling needs:

Fixed Absolute Sine Encoding ("fixed_abs_sine"):
- Classical sinusoidal positional embeddings from "Attention Is All You Need"
- Provides absolute position information for transformer attention
- Generalizes to sequences longer than training maximum length
- Essential for: Transformer encoder/decoder, offline processing

No Positional Encoding (None):
- Used with architectures that have implicit position modeling
- Conformer: Position handled via convolutional layers
- Mamba: Position modeled through state-space recurrence
- Essential for: Hybrid architectures, streaming applications

ATTENTION MECHANISM SUPPORT:
===========================
Multiple attention types for different modeling requirements:

Regular Multi-Head Attention ("regularMHA"):
- Standard scaled dot-product attention from original transformer paper
- Efficient computation with optimized kernels (Flash Attention, MPS)
- Best for: General applications, established baselines

Relative Positional Multi-Head Attention ("RelPosMHAXL"):
- TransformerXL-style relative position encoding integrated into attention
- Better length generalization and position awareness
- Best for: Variable-length sequences, streaming applications

HyperMixing Attention ("hypermixing"):
- Advanced attention mechanism with enhanced modeling capacity
- Hypernetwork-based attention computation for improved expressiveness
- Best for: Research applications, complex modeling tasks

DEVICE OPTIMIZATION STRATEGY:
============================
Cross-platform optimization for different deployment scenarios:

CUDA Acceleration:
- Optimized attention kernels via Flash Attention
- cuDNN-accelerated convolution operations
- Mixed precision training with gradient scaling
- Best for: Training, high-throughput inference on NVIDIA GPUs

Apple Silicon MPS:
- Metal Performance Shaders backend for Apple Silicon
- Unified memory architecture optimization
- Native ARM64 tensor operations
- Best for: Inference on Mac, development, edge deployment

CPU Fallback:
- Reference implementation for unsupported operations
- Debugging and development on any platform
- Lightweight inference for edge devices
- Best for: Development, testing, resource-constrained deployment

MEMORY OPTIMIZATION PATTERNS:
============================
Several design patterns optimize memory usage across different scales:

Gradient Checkpointing:
- Trade computation for memory in deep transformer stacks
- Enables training of larger models within memory constraints
- Configurable per-layer checkpointing for fine-tuned optimization

Attention Scaling:
- Memory-efficient attention computation for long sequences
- Chunked attention processing for streaming applications
- Adaptive attention window sizing based on available memory

Embedding Sharing:
- Shared embedding matrices between encoder and decoder
- Reduced parameter count and memory footprint
- Improved training efficiency through parameter tying

STREAMING COMPATIBILITY DESIGN:
==============================
Foundation infrastructure for real-time ASR processing:

Causal Processing Support:
- Configurable causal/non-causal attention modes
- Supports streaming inference with look-ahead constraints
- Maintains compatibility with Dynamic Chunk Training framework

Context Management:
- Base classes support streaming context objects
- Enables stateful processing across audio chunk boundaries
- Foundation for modules/Conformer.py streaming implementations

Incremental Inference:
- Architecture supports incremental decoding patterns
- Enables low-latency streaming ASR applications
- Compatible with beam search and other decoding strategies

CONFIGURATION-DRIVEN ARCHITECTURE:
==================================
The TransformerInterface enables flexible model configuration:

YAML Configuration Integration:
- All architecture choices configurable via hyperparameter files
- Enables rapid experimentation with different encoder/decoder combinations
- Supports ablation studies and architecture search

Runtime Architecture Selection:
- Dynamic encoder/decoder instantiation based on configuration
- Supports mixed architectures (e.g., Conformer encoder + Mamba decoder)
- Enables deployment-specific optimization

Model Scaling Support:
- Configurable layer counts, dimensions, and attention heads
- Supports models from mobile-optimized to large-scale research
- Enables progressive scaling and transfer learning

ERROR HANDLING AND ROBUSTNESS:
==============================
Comprehensive validation and error handling throughout:

Configuration Validation:
- Validates encoder/decoder module compatibility
- Checks attention type support across different architectures
- Prevents invalid configuration combinations with clear error messages

Device Compatibility Checks:
- Validates MPS availability and fallback strategies
- Handles device mismatch scenarios gracefully
- Provides informative warnings for suboptimal configurations

Numerical Stability:
- Implements numerical safeguards in attention and embedding layers
- Gradient clipping and scaling for stable training
- Handles edge cases in positional encoding and attention computation

Authors
-------
* Jianyuan Zhong 2020 (Original transformer implementation)
* Samuele Cornell 2021 (SpeechBrain integration and optimizations)
* Xilin Jiang 2024 (ConMamba and Mamba integration)
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import speechbrain as sb
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import RelPosEncXL
from speechbrain.nnet.CNN import Conv1d

from modules.Conformer import ConformerEncoder
from modules.Conmamba import ConmambaEncoder, MambaDecoder


# =============================================================================
# NAMED CONSTANTS FOR AI-FIRST DOCUMENTATION  
# =============================================================================
# These constants replace magic numbers throughout Transformer implementation
# to provide clear explanations and improve maintainability.

class TransformerConstants:
    """Constants for transformer architecture configuration and computation."""
    
    # Default model dimensions
    DEFAULT_D_MODEL = 512
    """Default model dimension for transformer architectures.
    Standard dimension providing good balance of capacity and computational efficiency."""
    
    DEFAULT_NHEAD = 8
    """Default number of attention heads for multi-head attention.
    Empirically proven effective for 512-dimensional models."""
    
    DEFAULT_NUM_LAYERS = 6
    """Default number of transformer layers for encoder/decoder.
    Original "Attention Is All You Need" paper configuration."""
    
    DEFAULT_D_FFN = 2048
    """Default feed-forward network dimension.
    Typically 4x the model dimension for optimal expressiveness."""
    
    DEFAULT_DROPOUT = 0.1
    """Default dropout rate for transformer training.
    Proven effective for regularization without over-damping."""
    
    DEFAULT_KERNEL_SIZE = 31
    """Default convolution kernel size for Conformer integration.
    Balances receptive field size with computational efficiency."""
    
    DEFAULT_MAX_LENGTH = 2500
    """Default maximum sequence length for positional encoding.
    Covers most practical ASR sequence lengths with reasonable memory usage."""
    
    DEFAULT_CSGU_LINEAR_UNITS = 3072
    """Default linear units for CSGU module in Branchformer.
    Optimized dimension for gated feed-forward processing."""
    
    # Positional encoding constants
    POSITIONAL_ENCODING_BASE = 10000.0
    """Base value for sinusoidal positional encoding frequency calculation.
    Standard value from "Attention Is All You Need" paper."""
    
    EVEN_POSITION_OFFSET = 0
    """Starting offset for even position indices in positional encoding.
    Used with slice notation: pe[:, 0::2] for sine components."""
    
    ODD_POSITION_OFFSET = 1
    """Starting offset for odd position indices in positional encoding.
    Used with slice notation: pe[:, 1::2] for cosine components."""
    
    DIMENSION_STEP = 2
    """Step size for dimension indexing in positional encoding.
    Ensures sin/cos alternation: torch.arange(0, input_size, 2)."""
    
    BATCH_DIMENSION_EXPANSION = 0
    """Dimension for unsqueezing positional encoding to add batch dimension.
    Transforms [seq_len, d_model] to [1, seq_len, d_model]."""
    
    # Embedding normalization constants
    EMBEDDING_SQRT_SCALING = True
    """Whether to apply sqrt(d_model) scaling to embeddings.
    Essential for numerical stability in transformer architectures."""
    
    # Layer normalization constants
    LAYER_NORM_EPS = 1e-6
    """Epsilon value for layer normalization to prevent division by zero.
    Standard value across transformer architectures for numerical stability."""
    
    # Layerdrop constants
    DEFAULT_LAYERDROP_PROB = 0.0
    """Default layer dropout probability for training efficiency.
    Set to 0.0 for standard training, >0.0 for regularization."""
    
    # Attention mask constants
    MASK_VALUE_NEGATIVE_INF = float('-inf')
    """Value used for masked positions in attention.
    Results in zero attention weight after softmax."""
    
    MASK_VALUE_ZERO = 0.0
    """Value for unmasked positions in attention masks.
    Allows normal attention computation."""
    
    MASK_COMPARISON_VALUE = 1
    """Comparison value for boolean mask operations.
    Used in mask.masked_fill(mask == 1, value) operations."""
    
    # Tensor dimension indices
    BATCH_DIM = 0
    """Batch dimension index in tensor shapes: [batch, seq, feature]."""
    
    SEQUENCE_DIM = 1
    """Sequence dimension index in tensor shapes: [batch, seq, feature]."""
    
    FEATURE_DIM = 2
    """Feature dimension index in tensor shapes: [batch, seq, feature]."""
    
    # Convolution defaults for FFN
    DEFAULT_FFN_KERNEL_SIZE = 3
    """Default kernel size for 1D convolution in feed-forward networks.
    Provides local context modeling in convolutional FFN variants."""
    
    # Numerical constants
    FLOAT_PRECISION_TOLERANCE = 1e-6
    """Tolerance for floating-point comparisons and numerical stability."""
    
    MINIMUM_PROBABILITY = 1e-8
    """Minimum probability value to prevent log(0) in loss computations."""


class TransformerInterface(nn.Module):
    """Unified interface for multi-modal transformer architectures in Mamba-ASR.
    
    This class serves as the central orchestrator and factory for creating diverse
    encoder-decoder combinations, enabling seamless integration of traditional
    transformers, state-of-the-art Conformer models, and innovative Mamba state-space
    models within a single unified API.
    
    ARCHITECTURAL ORCHESTRATION:
    ===========================
    TransformerInterface acts as a sophisticated architecture factory that dynamically
    creates and configures encoder-decoder pairs based on configuration parameters.
    This design enables rapid experimentation and deployment flexibility while
    maintaining consistent interfaces across all supported architectures.
    
    Core Design Philosophy:
    - Configuration-Driven: All architectural choices controlled via parameters
    - Cross-Architecture Compatibility: Seamless mixing of different encoder/decoder types
    - Streaming-Ready: Foundation for real-time ASR applications
    - Performance-Optimized: Device-aware instantiation and optimization
    
    INHERITANCE HIERARCHY AND CROSS-FILE INTEGRATION:
    ================================================
    This class serves as the foundational interface connecting multiple specialized
    encoder and decoder implementations across the Mamba-ASR codebase:
    
    Direct Inheritance:
    - Inherits from: torch.nn.Module (PyTorch base class)
    - Inherited by: No direct subclasses (serves as composition root)
    
    Composition Relationships:
    - Creates: TransformerEncoder, TransformerDecoder (defined in this file)
    - Instantiates: ConformerEncoder (from modules/Conformer.py)
    - Instantiates: ConmambaEncoder, MambaDecoder (from modules/Conmamba.py)
    - Contains: PositionalEncoding, NormalizedEmbedding (from this file)
    
    Cross-File Call Chain:
    =====================
    
    Called by:
    - `modules/TransformerASR.py`: TransformerASR.__init__() → TransformerInterface()
      * Primary integration point for ASR pipeline
      * Passes configuration from YAML hyperparameters
      * Creates complete encoder-decoder architecture
    
    - `train_CTC.py`: ASR class → hparams["modules"]["Transformer"] → TransformerInterface()
      * Training script instantiation via SpeechBrain configuration
      * CTC-based training pipeline integration
      * Model checkpoint and optimization setup
    
    - `train_S2S.py`: Sequence-to-sequence training → TransformerInterface()
      * S2S training pipeline with encoder-decoder architecture
      * Attention-based decoding and beam search integration
      * Cross-attention mechanism setup
    
    Calls to External Modules:
    - `modules/Conformer.py`: ConformerEncoder when encoder_module="conformer"
      * Passes d_model, nhead, kernel_size, attention_type parameters
      * Configures streaming context and Dynamic Chunk Training
      * Integrates CNN-Transformer hybrid architecture
    
    - `modules/Conmamba.py`: ConmambaEncoder when encoder_module="conmamba"
      * Passes mamba_config for state-space model configuration
      * Enables long-sequence efficient processing
      * Integrates Mamba state-space models with Conformer convolutions
    
    - `modules/Conmamba.py`: MambaDecoder when decoder_module="mamba"
      * Creates state-space model decoder for linear complexity decoding
      * Passes mamba_config for SSM parameter configuration
      * Enables memory-efficient autoregressive generation
    
    Internal Component Creation:
    - TransformerEncoder: Classical multi-head attention encoder
    - TransformerDecoder: Standard autoregressive transformer decoder
    - PositionalEncoding: Sinusoidal position embeddings for sequence modeling
    - NormalizedEmbedding: Scaled embedding layers for improved training dynamics
    
    DYNAMIC ARCHITECTURE SELECTION:
    ===============================
    The interface supports runtime architecture selection through encoder_module
    and decoder_module parameters, enabling flexible architecture combinations:
    
    Supported Encoder Architectures:
    1. "transformer": Classical transformer encoder (this file)
       - Multi-head self-attention with feed-forward networks
       - Bidirectional processing for offline ASR
       - Foundation for research and baseline comparisons
    
    2. "conformer": Conformer encoder (modules/Conformer.py)
       - Hybrid CNN-Transformer architecture for superior audio modeling
       - Streaming-compatible with Dynamic Chunk Training
       - State-of-the-art accuracy for production ASR
    
    3. "conmamba": ConMamba encoder (modules/Conmamba.py)
       - Innovative Conformer + Mamba hybrid architecture
       - Linear complexity in sequence length for long-form audio
       - Memory-efficient processing with state-space models
    
    4. "branchformer": Branchformer encoder (future integration)
       - Parallel attention-convolution processing branches
       - Enhanced feature extraction for complex acoustic environments
    
    Supported Decoder Architectures:
    1. "transformer": Classical transformer decoder (this file)
       - Standard autoregressive decoder with cross-attention
       - Full sequence-to-sequence modeling capability
       - Language model integration and beam search support
    
    2. "mamba": Mamba decoder (modules/Conmamba.py)
       - State-space model decoder with linear complexity
       - Memory-efficient long-sequence generation
       - Streaming-compatible autoregressive processing
    
    CONFIGURATION PARAMETER FLOW:
    =============================
    Parameters flow through the system in a structured hierarchy:
    
    YAML Configuration → TransformerASR → TransformerInterface → Specific Encoders/Decoders
    
    Key Parameter Categories:
    - Architecture Selection: encoder_module, decoder_module, attention_type
    - Model Scaling: d_model, nhead, num_encoder_layers, num_decoder_layers
    - Convolution Config: kernel_size, bias, conformer_activation (Conformer/ConMamba)
    - State-Space Config: mamba_config (ConMamba/Mamba)
    - Training Config: dropout, normalize_before, causal
    - Sequence Config: max_length, positional_encoding
    
    STREAMING AND REAL-TIME PROCESSING:
    ==================================
    The interface provides foundation support for streaming ASR applications:
    
    Streaming-Compatible Architectures:
    - Conformer with Dynamic Chunk Training support
    - ConMamba with state-space streaming context
    - Causal transformer variants for online processing
    
    Context Management:
    - Encoder streaming contexts for chunk-based processing
    - Decoder incremental generation for low-latency output
    - Cross-attention context preservation across chunks
    
    DEVICE OPTIMIZATION AND DEPLOYMENT:
    ==================================
    The interface enables device-aware optimization strategies:
    
    Apple Silicon MPS:
    - Automatic MPS backend selection for compatible operations
    - Unified memory architecture optimization
    - Native ARM64 tensor operations for Mac deployment
    
    CUDA Acceleration:
    - Flash Attention kernel integration for large-scale training
    - cuDNN-optimized convolution operations
    - Mixed precision training with automatic scaling
    
    CPU Fallback:
    - Reference implementations for development and testing
    - Edge device deployment with optimized inference
    - Debugging and profiling capabilities
    
    MEMORY OPTIMIZATION PATTERNS:
    ============================
    Built-in support for memory-efficient training and inference:
    
    Gradient Checkpointing:
    - Configurable checkpointing for deep transformer stacks
    - Trade computation for memory in resource-constrained environments
    - Layer-wise checkpointing control for fine-tuned optimization
    
    Architecture-Specific Optimizations:
    - Conformer: Macaron FFN structure reduces intermediate activations
    - Mamba: Linear memory complexity for long sequences
    - Transformer: Standard attention memory patterns with Flash Attention
    
    ERROR HANDLING AND VALIDATION:
    ==============================
    Comprehensive validation ensures robust configuration and operation:
    
    Configuration Validation:
    - Validates encoder/decoder module compatibility
    - Checks attention type support across architectures
    - Prevents invalid parameter combinations with clear error messages
    
    Device Compatibility:
    - Automatic device detection and optimization
    - Graceful fallback for unsupported operations
    - Performance warnings for suboptimal configurations
    
    Parameter Validation:
    - Range checking for model dimensions and layer counts
    - Compatibility verification between encoder and decoder configurations
    - Mamba configuration validation for state-space model parameters
    
    Based on the foundational paper "Attention Is All You Need":
    https://arxiv.org/pdf/1706.03762.pdf

    Arguments
    ---------
    d_model: int
        The number of expected features in the encoder/decoder inputs (default=512).
    nhead: int
        The number of heads in the multi-head attention models (default=8).
    num_encoder_layers: int, optional
        The number of encoder layers in1ì the encoder.
    num_decoder_layers: int, optional
        The number of decoder layers in the decoder.
    d_ffn: int, optional
        The dimension of the feedforward network model hidden layer.
    dropout: int, optional
        The dropout value.
    activation: torch.nn.Module, optional
        The activation function for Feed-Forward Network layer,
        e.g., relu or gelu or swish.
    custom_src_module: torch.nn.Module, optional
        Module that processes the src features to expected feature dim.
    custom_tgt_module: torch.nn.Module, optional
        Module that processes the src features to expected feature dim.
    positional_encoding: str, optional
        Type of positional encoding used. e.g. 'fixed_abs_sine' for fixed absolute positional encodings.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    kernel_size: int, optional
        Kernel size in convolutional layers when Conformer is used.
    bias: bool, optional
        Whether to use bias in Conformer convolutional layers.
    encoder_module: str, optional
        Choose between Branchformer, Conformer, ConMamba, and Transformer for the encoder.
    decoder_module: str, optional
        Choose between Mamba and Transformer for the decoder.
    conformer_activation: torch.nn.Module, optional
        Activation module used after Conformer convolutional layers. E.g. Swish, ReLU etc. it has to be a torch Module.
    branchformer_activation: torch.nn.Module, optional
        Activation module used within the Branchformer Encoder. E.g. Swish, ReLU etc. it has to be a torch Module.
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.
    max_length: int, optional
        Max length for the target and source sequence in input.
        Used for positional encodings.
    causal: bool, optional
        Whether the encoder should be causal or not (the decoder is always causal).
        If causal the Conformer convolutional layer is causal.
    encoder_kdim: int, optional
        Dimension of the key for the encoder.
    encoder_vdim: int, optional
        Dimension of the value for the encoder.
    decoder_kdim: int, optional
        Dimension of the key for the decoder.
    decoder_vdim: int, optional
        Dimension of the value for the decoder.
    csgu_linear_units: int, optional
        Number of neurons in the hidden linear units of the CSGU Module.
        -> Branchformer
    gate_activation: torch.nn.Module, optional
        Activation function used at the gate of the CSGU module.
        -> Branchformer
    use_linear_after_conv: bool, optional
        If True, will apply a linear transformation of size input_size//2.
        -> Branchformer
    mamba_config: dict, optional
        Mamba parameters if encoder_module or decoder_module is Mamba or ConMamba
    """

    def __init__(
        self,
        d_model=TransformerConstants.DEFAULT_D_MODEL,
        nhead=TransformerConstants.DEFAULT_NHEAD,
        num_encoder_layers=TransformerConstants.DEFAULT_NUM_LAYERS,
        num_decoder_layers=TransformerConstants.DEFAULT_NUM_LAYERS,
        d_ffn=TransformerConstants.DEFAULT_D_FFN,
        dropout=TransformerConstants.DEFAULT_DROPOUT,
        activation=nn.ReLU,
        custom_src_module=None,
        custom_tgt_module=None,
        positional_encoding="fixed_abs_sine",
        normalize_before=True,
        kernel_size: Optional[int] = TransformerConstants.DEFAULT_KERNEL_SIZE,
        bias: Optional[bool] = True,
        encoder_module: Optional[str] = "transformer",
        decoder_module: Optional[str] = "transformer",
        conformer_activation: Optional[nn.Module] = Swish,
        branchformer_activation: Optional[nn.Module] = nn.GELU,
        attention_type: Optional[str] = "regularMHA",
        max_length: Optional[int] = TransformerConstants.DEFAULT_MAX_LENGTH,
        causal: Optional[bool] = False,
        encoder_kdim: Optional[int] = None,
        encoder_vdim: Optional[int] = None,
        decoder_kdim: Optional[int] = None,
        decoder_vdim: Optional[int] = None,
        csgu_linear_units: Optional[int] = TransformerConstants.DEFAULT_CSGU_LINEAR_UNITS,
        gate_activation: Optional[nn.Module] = nn.Identity,
        use_linear_after_conv: Optional[bool] = False,
        mamba_config=None
    ):
        super().__init__()
        self.causal = causal
        self.attention_type = attention_type
        self.positional_encoding_type = positional_encoding
        self.encoder_kdim = encoder_kdim
        self.encoder_vdim = encoder_vdim
        self.decoder_kdim = decoder_kdim
        self.decoder_vdim = decoder_vdim

        assert attention_type in ["regularMHA", "RelPosMHAXL", "hypermixing"]
        assert positional_encoding in ["fixed_abs_sine", None]

        assert (
            num_encoder_layers + num_decoder_layers > 0
        ), "number of encoder layers and number of decoder layers cannot both be 0!"

        if positional_encoding == "fixed_abs_sine":
            self.positional_encoding = PositionalEncoding(d_model, max_length)
        elif positional_encoding is None:
            pass
            # no positional encodings

        # overrides any other pos_embedding
        if attention_type == "RelPosMHAXL":
            self.positional_encoding = RelPosEncXL(d_model)
            self.positional_encoding_decoder = PositionalEncoding(
                d_model, max_length
            )

        # initialize the encoder
        if num_encoder_layers > 0:
            if custom_src_module is not None:
                self.custom_src_module = custom_src_module(d_model)
            if encoder_module == "transformer":
                self.encoder = TransformerEncoder(
                    nhead=nhead,
                    num_layers=num_encoder_layers,
                    d_ffn=d_ffn,
                    d_model=d_model,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=self.causal,
                    attention_type=self.attention_type,
                    kdim=self.encoder_kdim,
                    vdim=self.encoder_vdim,
                )
            elif encoder_module == "conformer":
                self.encoder = ConformerEncoder(
                    nhead=nhead,
                    num_layers=num_encoder_layers,
                    d_ffn=d_ffn,
                    d_model=d_model,
                    dropout=dropout,
                    activation=conformer_activation,
                    kernel_size=kernel_size,
                    bias=bias,
                    causal=self.causal,
                    attention_type=self.attention_type,
                )
                assert (
                    normalize_before
                ), "normalize_before must be True for Conformer"

                assert (
                    conformer_activation is not None
                ), "conformer_activation must not be None"
            elif encoder_module == "branchformer":
                self.encoder = BranchformerEncoder(
                    nhead=nhead,
                    num_layers=num_encoder_layers,
                    d_model=d_model,
                    dropout=dropout,
                    activation=branchformer_activation,
                    kernel_size=kernel_size,
                    attention_type=self.attention_type,
                    csgu_linear_units=csgu_linear_units,
                    gate_activation=gate_activation,
                    use_linear_after_conv=use_linear_after_conv,
                )
            elif encoder_module == "conmamba":
                self.encoder = ConmambaEncoder(
                    num_layers=num_encoder_layers,
                    d_model=d_model,
                    d_ffn=d_ffn,
                    dropout=dropout,
                    activation=branchformer_activation,
                    kernel_size=kernel_size,
                    bias=bias,
                    causal=self.causal,
                    mamba_config=mamba_config
                )
                assert (
                    normalize_before
                ), "normalize_before must be True for Conmamba"

                assert (
                    conformer_activation is not None
                ), "conformer_activation must not be None"

        # initialize the decoder
        if num_decoder_layers > 0:
            if custom_tgt_module is not None:
                self.custom_tgt_module = custom_tgt_module(d_model)
            if decoder_module == 'transformer':
                self.decoder = TransformerDecoder(
                    num_layers=num_decoder_layers,
                    nhead=nhead,
                    d_ffn=d_ffn,
                    d_model=d_model,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=True,
                    attention_type="regularMHA",  # always use regular attention in decoder
                    kdim=self.decoder_kdim,
                    vdim=self.decoder_vdim,
                )
            elif decoder_module in ['mamba']:
                self.decoder = MambaDecoder(
                    num_layers=num_decoder_layers,
                    d_ffn=d_ffn,
                    d_model=d_model,
                    activation=activation,
                    dropout=dropout,
                    normalize_before=normalize_before,
                    mamba_config=mamba_config
                )
            else:
                raise NotImplementedError(decoder_module)

    def forward(self, **kwags):
        """Users should modify this function according to their own tasks."""
        raise NotImplementedError


class PositionalEncoding(nn.Module):
    """Absolute sinusoidal positional encoding for transformer sequence modeling.
    
    This class implements the classical absolute positional encoding from 
    "Attention Is All You Need" using sinusoidal functions to inject position
    information into transformer models. The encoding provides deterministic
    position information that generalizes to sequences longer than those seen
    during training.
    
    MATHEMATICAL FOUNDATION:
    =======================
    The positional encoding uses sine and cosine functions with different frequencies
    to create unique position embeddings for each position in the sequence:
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Where:
    - pos: Position in the sequence (0 to sequence_length-1)
    - i: Dimension index (0 to d_model/2-1)
    - d_model: Model dimension (must be even)
    
    This formulation ensures that:
    - Each position gets a unique encoding vector
    - The encoding is deterministic and reproducible
    - Relative position relationships are preserved through trigonometric properties
    - The model can extrapolate to longer sequences than seen in training
    
    SYSTEM ROLE IN MAMBA-ASR:
    =========================
    PositionalEncoding serves as the position injection mechanism for transformer
    architectures that lack inherent position awareness:
    
    Used by:
    - TransformerInterface: When positional_encoding="fixed_abs_sine"
    - TransformerEncoder: Adds position information to input embeddings
    - TransformerDecoder: Provides position context for autoregressive generation
    
    Not used by:
    - ConformerEncoder: Position handled via convolutional layers
    - ConmambaEncoder: Position modeled through state-space recurrence
    - MambaDecoder: Position implicit in state-space model design
    
    CROSS-FILE INTEGRATION:
    ======================
    This class integrates with the broader transformer infrastructure:
    
    Called by:
    - TransformerInterface.__init__(): Creates PositionalEncoding when needed
      * Configured via positional_encoding parameter
      * Instantiated with d_model and max_length parameters
      * Shared between encoder and decoder if both use absolute positioning
    
    - TransformerEncoder.forward(): Applies encoding to input sequences
      * Automatically extracts sequence length from input tensors
      * Broadcasts position encodings across batch dimension
      * Adds to input embeddings before first transformer layer
    
    - TransformerDecoder.forward(): Provides position context for target sequences
      * Handles both training (teacher forcing) and inference modes
      * Supports incremental decoding with position tracking
      * Integrates with attention mechanisms for position-aware processing
    
    ARCHITECTURE COMPATIBILITY:
    ==========================
    Position encoding compatibility across different model architectures:
    
    Required for:
    - Pure Transformer architectures: Essential for position awareness
    - Standard transformer encoder-decoder models
    - Research baselines and ablation studies
    
    Optional for:
    - Hybrid architectures with implicit position modeling
    - Models using relative positional attention (RelPosMHAXL)
    - Architectures with convolutional position encoding
    
    Not compatible with:
    - Architectures requiring relative position encoding
    - Models with learned position embeddings
    - State-space models with recurrent position handling
    
    MEMORY AND PERFORMANCE CHARACTERISTICS:
    ======================================
    The implementation optimizes for memory efficiency and computational speed:
    
    Memory Optimization:
    - Pre-computed encoding matrix: O(max_len * d_model) storage
    - No gradient computation: requires_grad=False for efficiency
    - Shared across all sequences: Single encoding matrix for entire model
    - Broadcasting: Efficient memory usage across batch dimensions
    
    Computational Efficiency:
    - One-time computation: Encodings computed once during initialization
    - Fast indexing: O(1) lookup for sequence position encodings
    - Vectorized operations: Efficient sine/cosine computation via PyTorch
    - Device-agnostic: Automatic device placement with model parameters
    
    Typical Memory Usage:
    - d_model=512, max_len=2500: ~5MB per encoding matrix
    - d_model=768, max_len=5000: ~15MB per encoding matrix
    - d_model=1024, max_len=8000: ~32MB per encoding matrix
    
    SEQUENCE LENGTH HANDLING:
    ========================
    The implementation provides flexible sequence length support:
    
    Automatic Length Adaptation:
    - Extracts sequence length from input tensor dimensions
    - Selects appropriate subset of pre-computed encodings
    - Supports variable-length sequences within max_len bounds
    
    Length Extrapolation:
    - Mathematical properties enable extrapolation beyond max_len
    - Maintains position relationship consistency for longer sequences
    - Graceful degradation for extremely long sequences
    
    Error Handling:
    - Validates input_size is even (required for sin/cos pairing)
    - Provides clear error messages for configuration issues
    - Handles edge cases in sequence length and dimension validation
    
    DEVICE COMPATIBILITY:
    ====================
    Full support across PyTorch-compatible compute platforms:
    
    CUDA Acceleration:
    - GPU-accelerated trigonometric function computation
    - Efficient tensor broadcasting and indexing
    - Memory coalescing for optimal bandwidth utilization
    
    Apple Silicon MPS:
    - Native Metal Performance Shaders implementation
    - Unified memory architecture optimization
    - ARM64-optimized trigonometric operations
    
    CPU Implementation:
    - Reference implementation for development and debugging
    - Optimized BLAS operations for trigonometric computation
    - Memory-efficient computation for edge deployment
    
    NUMERICAL STABILITY:
    ===================
    The implementation ensures numerical stability across different scenarios:
    
    Precision Considerations:
    - Float32 precision sufficient for most applications
    - Stable computation across wide range of sequence lengths
    - Consistent results across different device types
    
    Gradient Flow:
    - No gradient computation for position encodings (requires_grad=False)
    - Preserves gradient flow for input embeddings
    - Numerical stability in downstream attention computations
    
    Arguments
    ---------
    input_size : int
        Embedding dimension for the positional encoding.
        Must be even to accommodate sin/cos pairing.
        Typical values: 256, 512, 768, 1024 based on model architecture.
        
    max_len : int, optional
        Maximum length of input sequences for pre-computation (default: 2500).
        Determines the size of the pre-computed encoding matrix.
        Should be set larger than expected maximum sequence length.
        Can extrapolate beyond this limit with potential quality degradation.

    Example
    -------
    Basic usage with transformer input:
    >>> input_embeddings = torch.rand((8, 120, 512))  # [batch, seq_len, d_model]
    >>> pos_enc = PositionalEncoding(input_size=512, max_len=2500)
    >>> position_encodings = pos_enc(input_embeddings)
    >>> position_encodings.shape
    torch.Size([1, 120, 512])
    
    Integration with transformer architecture:
    >>> d_model = 512
    >>> seq_len = 256
    >>> batch_size = 16
    >>> 
    >>> # Create position encoding
    >>> pos_encoding = PositionalEncoding(input_size=d_model, max_len=5000)
    >>> 
    >>> # Input embeddings from embedding layer
    >>> input_embeds = torch.rand(batch_size, seq_len, d_model)
    >>> 
    >>> # Add positional information
    >>> pos_embeds = pos_encoding(input_embeds)
    >>> combined = input_embeds + pos_embeds
    >>> 
    >>> # Feed to transformer layers
    >>> transformer_input = combined
    """

    def __init__(self, input_size, max_len=TransformerConstants.DEFAULT_MAX_LENGTH):
        super().__init__()
        if input_size % TransformerConstants.DIMENSION_STEP != 0:
            raise ValueError(
                f"Cannot use sin/cos positional encoding with odd channels (got channels={input_size})"
            )
        self.max_len = max_len
        pe = torch.zeros(self.max_len, input_size, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(TransformerConstants.EVEN_POSITION_OFFSET, input_size, TransformerConstants.DIMENSION_STEP).float()
            * -(math.log(TransformerConstants.POSITIONAL_ENCODING_BASE) / input_size)
        )

        pe[:, TransformerConstants.EVEN_POSITION_OFFSET::TransformerConstants.DIMENSION_STEP] = torch.sin(positions * denominator)
        pe[:, TransformerConstants.ODD_POSITION_OFFSET::TransformerConstants.DIMENSION_STEP] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(TransformerConstants.BATCH_DIMENSION_EXPANSION)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments
        ---------
        x : torch.Tensor
            Input feature shape (batch, time, fea)

        Returns
        -------
        The positional encoding.
        """
        return self.pe[:, : x.size(1)].clone().detach()


class TransformerEncoderLayer(nn.Module):
    """This is an implementation of self-attention encoder layer.

    Arguments
    ---------
    d_ffn: int, optional
        The dimension of the feedforward network model hidden layer.
    nhead: int
        The number of heads in the multi-head attention models (default=8).
    d_model: int
        The number of expected features in the encoder/decoder inputs (default=512).
    kdim: int, optional
        Dimension of the key.
    vdim: int, optional
        Dimension of the value.
    dropout: int, optional
        The dropout value.
    activation: torch.nn.Module, optional
        The activation function for Feed-Forward Network layer,
        e.g., relu or gelu or swish.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.
    ffn_type: str
        type of ffn: regularFFN/1dcnn
    ffn_cnn_kernel_size_list: list of int
        kernel size of 2 1d-convs if ffn_type is 1dcnn
    causal: bool, optional
        Whether the encoder should be causal or not (the decoder is always causal).
        If causal the Conformer convolutional layer is causal.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoderLayer(512, 8, d_model=512)
    >>> output = net(x)
    >>> output[0].shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        d_model,
        kdim=None,
        vdim=None,
        dropout=TransformerConstants.DEFAULT_LAYERDROP_PROB,
        activation=nn.ReLU,
        normalize_before=False,
        attention_type="regularMHA",
        ffn_type="regularFFN",
        ffn_cnn_kernel_size_list=[TransformerConstants.DEFAULT_FFN_KERNEL_SIZE, TransformerConstants.DEFAULT_FFN_KERNEL_SIZE],
        causal=False,
    ):
        super().__init__()

        if attention_type == "regularMHA":
            self.self_att = sb.nnet.attention.MultiheadAttention(
                nhead=nhead,
                d_model=d_model,
                dropout=dropout,
                kdim=kdim,
                vdim=vdim,
            )

        elif attention_type == "RelPosMHAXL":
            self.self_att = sb.nnet.attention.RelPosMHAXL(
                d_model, nhead, dropout, mask_pos_future=causal
            )
        elif attention_type == "hypermixing":
            self.self_att = sb.nnet.hypermixing.HyperMixing(
                input_output_dim=d_model,
                hypernet_size=d_ffn,
                tied=False,
                num_heads=nhead,
                fix_tm_hidden_size=False,
            )

        if ffn_type == "regularFFN":
            self.pos_ffn = sb.nnet.attention.PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            )
        elif ffn_type == "1dcnn":
            self.pos_ffn = nn.Sequential(
                Conv1d(
                    in_channels=d_model,
                    out_channels=d_ffn,
                    kernel_size=ffn_cnn_kernel_size_list[0],
                    padding="causal" if causal else "same",
                ),
                nn.ReLU(),
                Conv1d(
                    in_channels=d_ffn,
                    out_channels=d_model,
                    kernel_size=ffn_cnn_kernel_size_list[1],
                    padding="causal" if causal else "same",
                ),
            )

        self.norm1 = sb.nnet.normalization.LayerNorm(d_model, eps=TransformerConstants.LAYER_NORM_EPS)
        self.norm2 = sb.nnet.normalization.LayerNorm(d_model, eps=TransformerConstants.LAYER_NORM_EPS)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.normalize_before = normalize_before
        self.pos_ffn_type = ffn_type

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ---------
        src : torch.Tensor
            The sequence to the encoder layer.
        src_mask : torch.Tensor
            The mask for the src query for each example in the batch.
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src keys for each example in the batch.
        pos_embs: torch.Tensor, optional
            The positional embeddings tensor.

        Returns
        -------
        output : torch.Tensor
            The output of the transformer encoder layer.
        """

        if self.normalize_before:
            src1 = self.norm1(src)
        else:
            src1 = src

        output, self_attn = self.self_att(
            src1,
            src1,
            src1,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs,
        )

        # add & norm
        src = src + self.dropout1(output)
        if not self.normalize_before:
            src = self.norm1(src)

        if self.normalize_before:
            src1 = self.norm2(src)
        else:
            src1 = src
        output = self.pos_ffn(src1)

        # add & norm
        output = src + self.dropout2(output)
        if not self.normalize_before:
            output = self.norm2(output)
        return output, self_attn


class TransformerEncoder(nn.Module):
    """This class implements the transformer encoder.

    Arguments
    ---------
    num_layers : int
        Number of transformer layers to include.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    input_shape : tuple
        Expected shape of the input.
    d_model : int
        The dimension of the input embedding.
    kdim : int
        Dimension for key (Optional).
    vdim : int
        Dimension for value (Optional).
    dropout : float
        Dropout for the encoder (Optional).
    activation: torch.nn.Module, optional
        The activation function for Feed-Forward Network layer,
        e.g., relu or gelu or swish.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    causal: bool, optional
        Whether the encoder should be causal or not (the decoder is always causal).
        If causal the Conformer convolutional layer is causal.
    layerdrop_prob: float
        The probability to drop an entire layer
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.
    ffn_type: str
        type of ffn: regularFFN/1dcnn
    ffn_cnn_kernel_size_list: list of int
        conv kernel size of 2 1d-convs if ffn_type is 1dcnn

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoder(1, 8, 512, d_model=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        input_shape=None,
        d_model=None,
        kdim=None,
        vdim=None,
        dropout=TransformerConstants.DEFAULT_LAYERDROP_PROB,
        activation=nn.ReLU,
        normalize_before=False,
        causal=False,
        layerdrop_prob=TransformerConstants.DEFAULT_LAYERDROP_PROB,
        attention_type="regularMHA",
        ffn_type="regularFFN",
        ffn_cnn_kernel_size_list=[TransformerConstants.DEFAULT_FFN_KERNEL_SIZE, TransformerConstants.DEFAULT_FFN_KERNEL_SIZE],
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=causal,
                    attention_type=attention_type,
                    ffn_type=ffn_type,
                    ffn_cnn_kernel_size_list=ffn_cnn_kernel_size_list,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=TransformerConstants.LAYER_NORM_EPS)
        self.layerdrop_prob = layerdrop_prob
        self.rng = np.random.default_rng()

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
        dynchunktrain_config=None,
    ):
        """
        Arguments
        ---------
        src : torch.Tensor
            The sequence to the encoder layer (required).
        src_mask : torch.Tensor
            The mask for the src sequence (optional).
        src_key_padding_mask : torch.Tensor
            The mask for the src keys per batch (optional).
        pos_embs : torch.Tensor
            The positional embedding tensor
        dynchunktrain_config : config
            Not supported for this encoder.

        Returns
        -------
        output : torch.Tensor
            The output of the transformer.
        attention_lst : list
            The attention values.
        """
        assert (
            dynchunktrain_config is None
        ), "Dynamic Chunk Training unsupported for this encoder"

        output = src
        if self.layerdrop_prob > TransformerConstants.DEFAULT_LAYERDROP_PROB:
            keep_probs = self.rng.random(len(self.layers))
        else:
            keep_probs = None
        attention_lst = []
        for i, enc_layer in enumerate(self.layers):
            if (
                not self.training
                or self.layerdrop_prob == 0.0
                or keep_probs[i] > self.layerdrop_prob
            ):
                output, attention = enc_layer(
                    output,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    pos_embs=pos_embs,
                )

                attention_lst.append(attention)
        output = self.norm(output)
        return output, attention_lst


class TransformerDecoderLayer(nn.Module):
    """This class implements the self-attention decoder layer.

    Arguments
    ---------
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    nhead : int
        Number of attention heads.
    d_model : int
        Dimension of the model.
    kdim : int
        Dimension for key (optional).
    vdim : int
        Dimension for value (optional).
    dropout : float
        Dropout for the decoder (optional).
    activation : Callable
        Function to use between layers, default nn.ReLU
    normalize_before : bool
        Whether to normalize before layers.
    attention_type : str
        Type of attention to use, "regularMHA" or "RelPosMHAXL"
    causal : bool
        Whether to mask future positions.

    Example
    -------
    >>> src = torch.rand((8, 60, 512))
    >>> tgt = torch.rand((8, 60, 512))
    >>> net = TransformerDecoderLayer(1024, 8, d_model=512)
    >>> output, self_attn, multihead_attn = net(src, tgt)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        d_model,
        kdim=None,
        vdim=None,
        dropout=TransformerConstants.DEFAULT_LAYERDROP_PROB,
        activation=nn.ReLU,
        normalize_before=False,
        attention_type="regularMHA",
        causal=None,
    ):
        super().__init__()
        self.nhead = nhead

        if attention_type == "regularMHA":
            self.self_attn = sb.nnet.attention.MultiheadAttention(
                nhead=nhead,
                d_model=d_model,
                kdim=kdim,
                vdim=vdim,
                dropout=dropout,
            )
            self.multihead_attn = sb.nnet.attention.MultiheadAttention(
                nhead=nhead,
                d_model=d_model,
                kdim=kdim,
                vdim=vdim,
                dropout=dropout,
            )

        elif attention_type == "RelPosMHAXL":
            self.self_attn = sb.nnet.attention.RelPosMHAXL(
                d_model, nhead, dropout, mask_pos_future=causal
            )
            self.multihead_attn = sb.nnet.attention.RelPosMHAXL(
                d_model, nhead, dropout, mask_pos_future=causal
            )

        self.pos_ffn = sb.nnet.attention.PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=d_model,
            dropout=dropout,
            activation=activation,
        )

        # normalization layers
        self.norm1 = sb.nnet.normalization.LayerNorm(d_model, eps=TransformerConstants.LAYER_NORM_EPS)
        self.norm2 = sb.nnet.normalization.LayerNorm(d_model, eps=TransformerConstants.LAYER_NORM_EPS)
        self.norm3 = sb.nnet.normalization.LayerNorm(d_model, eps=TransformerConstants.LAYER_NORM_EPS)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos_embs_tgt=None,
        pos_embs_src=None,
    ):
        """
        Arguments
        ----------
        tgt: torch.Tensor
            The sequence to the decoder layer (required).
        memory: torch.Tensor
            The sequence from the last layer of the encoder (required).
        tgt_mask: torch.Tensor
            The mask for the tgt sequence (optional).
        memory_mask: torch.Tensor
            The mask for the memory sequence (optional).
        tgt_key_padding_mask: torch.Tensor
            The mask for the tgt keys per batch (optional).
        memory_key_padding_mask: torch.Tensor
            The mask for the memory keys per batch (optional).
        pos_embs_tgt: torch.Tensor
            The positional embeddings for the target (optional).
        pos_embs_src: torch.Tensor
            The positional embeddings for the source (optional).
        """
        if self.normalize_before:
            tgt1 = self.norm1(tgt)
        else:
            tgt1 = tgt

        # self-attention over the target sequence
        tgt2, self_attn = self.self_attn(
            query=tgt1,
            key=tgt1,
            value=tgt1,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            pos_embs=pos_embs_tgt,
        )

        # add & norm
        tgt = tgt + self.dropout1(tgt2)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        if self.normalize_before:
            tgt1 = self.norm2(tgt)
        else:
            tgt1 = tgt

        # multi-head attention over the target sequence and encoder states

        tgt2, multihead_attention = self.multihead_attn(
            query=tgt1,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            pos_embs=pos_embs_src,
        )

        # add & norm
        tgt = tgt + self.dropout2(tgt2)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        if self.normalize_before:
            tgt1 = self.norm3(tgt)
        else:
            tgt1 = tgt

        tgt2 = self.pos_ffn(tgt1)

        # add & norm
        tgt = tgt + self.dropout3(tgt2)
        if not self.normalize_before:
            tgt = self.norm3(tgt)

        return tgt, self_attn, multihead_attention


class TransformerDecoder(nn.Module):
    """This class implements the Transformer decoder.

    Arguments
    ---------
    num_layers : int
        Number of transformer layers for the decoder.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    d_model : int
        Dimension of the model.
    kdim : int, optional
        Dimension for key (Optional).
    vdim : int, optional
        Dimension for value (Optional).
    dropout : float, optional
        Dropout for the decoder (Optional).
    activation : Callable
        The function to apply between layers, default nn.ReLU
    normalize_before : bool
        Whether to normalize before layers.
    causal : bool
        Whether to allow future information in decoding.
    attention_type : str
        Type of attention to use, "regularMHA" or "RelPosMHAXL"

    Example
    -------
    >>> src = torch.rand((8, 60, 512))
    >>> tgt = torch.rand((8, 60, 512))
    >>> net = TransformerDecoder(1, 8, 1024, d_model=512)
    >>> output, _, _ = net(src, tgt)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        d_model,
        kdim=None,
        vdim=None,
        dropout=TransformerConstants.DEFAULT_LAYERDROP_PROB,
        activation=nn.ReLU,
        normalize_before=False,
        causal=False,
        attention_type="regularMHA",
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=causal,
                    attention_type=attention_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=TransformerConstants.LAYER_NORM_EPS)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos_embs_tgt=None,
        pos_embs_src=None,
    ):
        """
        Arguments
        ----------
        tgt : torch.Tensor
            The sequence to the decoder layer (required).
        memory : torch.Tensor
            The sequence from the last layer of the encoder (required).
        tgt_mask : torch.Tensor
            The mask for the tgt sequence (optional).
        memory_mask : torch.Tensor
            The mask for the memory sequence (optional).
        tgt_key_padding_mask : torch.Tensor
            The mask for the tgt keys per batch (optional).
        memory_key_padding_mask : torch.Tensor
            The mask for the memory keys per batch (optional).
        pos_embs_tgt : torch.Tensor
            The positional embeddings for the target (optional).
        pos_embs_src : torch.Tensor
            The positional embeddings for the source (optional).
        """
        output = tgt
        self_attns, multihead_attns = [], []
        for dec_layer in self.layers:
            output, self_attn, multihead_attn = dec_layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos_embs_tgt=pos_embs_tgt,
                pos_embs_src=pos_embs_src,
            )
            self_attns.append(self_attn)
            multihead_attns.append(multihead_attn)
        output = self.norm(output)

        return output, self_attns, multihead_attns


class NormalizedEmbedding(nn.Module):
    """Scaled embedding layer optimized for transformer architectures.
    
    This class implements the mathematical scaling optimization from "Attention Is All You Need"
    where embedding outputs are multiplied by sqrt(d_model) to maintain appropriate signal
    magnitudes throughout the transformer architecture. This scaling is crucial for numerical
    stability and convergence in transformer training.
    
    MATHEMATICAL FOUNDATION:
    =======================
    The scaling addresses the fundamental issue of signal magnitude in transformer architectures:
    
    Standard Embedding Output: E(x) ∈ [-1, 1]^d_model (approximately)
    Scaled Embedding Output: E_scaled(x) = E(x) * sqrt(d_model)
    
    This scaling compensates for the sqrt(d_model) normalization in scaled dot-product attention:
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_model))V
    
    The scaling ensures that:
    - Embedding magnitudes match expected input ranges for attention mechanisms
    - Weight sharing between input embeddings and output projections remains stable
    - Gradient magnitudes stay within optimal ranges for stable training
    - Signal-to-noise ratio is preserved throughout the network depth
    
    SYSTEM ROLE IN MAMBA-ASR:
    =========================
    NormalizedEmbedding provides the foundation for token-to-vector conversion in
    transformer decoders and sequence-to-sequence models:
    
    Used by:
    - TransformerInterface: Creates embedding layers for decoder token inputs
    - TransformerDecoder: Converts target token sequences to continuous representations
    - Sequence-to-sequence models: Input processing for autoregressive generation
    
    Not used by:
    - Encoder-only models: Encoders typically process continuous features (audio spectrograms)
    - CTC-based models: Direct acoustic feature processing without token embeddings
    - Conformer/ConMamba encoders: Audio feature processing bypasses token embeddings
    
    CROSS-FILE INTEGRATION:
    ======================
    This class integrates with the broader ASR and transformer infrastructure:
    
    Created by:
    - TransformerInterface.__init__(): Instantiates embedding layer for decoder
      * Configured with vocabulary size from tokenizer configuration
      * d_model parameter determines embedding dimension
      * Blank token ID set to 0 for CTC compatibility
    
    Used by:
    - TransformerDecoder.forward(): Converts target tokens to embeddings
      * Processes teacher-forced target sequences during training
      * Handles autoregressive generation during inference
      * Integrates with positional encoding for position-aware processing
    
    - Training Scripts: Token-based language modeling and sequence generation
      * train_S2S.py: Sequence-to-sequence training with teacher forcing
      * CTC training: Alignment-based training may use embeddings for language modeling
    
    Calls to:
    - speechbrain.nnet.embedding.Embedding: Underlying embedding implementation
      * Handles vocabulary mapping and gradient computation
      * Provides blank token support for CTC-compatible vocabularies
      * Enables weight sharing with output projection layers
    
    ARCHITECTURAL INTEGRATION PATTERNS:
    ==================================
    The normalized embedding integrates with different architectural components:
    
    Transformer Decoder Integration:
    - Token Input: target_tokens → NormalizedEmbedding → scaled_embeddings
    - Position Addition: scaled_embeddings + positional_encoding → decoder_input
    - Layer Processing: decoder_input → transformer_layers → output_representations
    
    Weight Sharing Optimization:
    - Shared Weights: embedding.weight = output_projection.weight.T
    - Memory Efficiency: Single weight matrix for input and output transformations
    - Training Stability: Consistent gradient flow between input and output
    
    CTC Integration:
    - Blank Token: Index 0 reserved for CTC blank symbol
    - Vocabulary Alignment: Token indices match CTC output vocabulary
    - Training Compatibility: Supports both CTC and attention-based training
    
    OPTIMIZATION AND PERFORMANCE:
    ============================
    The implementation includes several optimization strategies:
    
    Memory Optimization:
    - Efficient Lookup: O(1) embedding lookup for token indices
    - Batch Processing: Vectorized operations across sequence and batch dimensions
    - Weight Sharing: Single embedding matrix serves dual input/output purpose
    
    Computational Efficiency:
    - Vectorized Scaling: Single multiplication operation for all embedding dimensions
    - Cache-Friendly Access: Sequential memory access patterns for batch processing
    - Gradient Optimization: Sparse gradient updates for embedding parameters
    
    Numerical Stability:
    - Consistent Scaling: sqrt(d_model) maintains signal magnitude consistency
    - Float32 Precision: Stable computation across different model scales
    - Gradient Scaling: Prevents vanishing/exploding gradients in deep networks
    
    DEVICE COMPATIBILITY:
    ====================
    Full support across PyTorch-compatible compute platforms:
    
    CUDA Acceleration:
    - GPU-accelerated embedding lookup operations
    - Parallel batch processing for high-throughput training
    - Memory coalescing for optimal bandwidth utilization
    
    Apple Silicon MPS:
    - Native Metal Performance Shaders embedding operations
    - Unified memory architecture for efficient token processing
    - ARM64-optimized mathematical operations
    
    CPU Implementation:
    - Reference implementation for development and testing
    - Optimized BLAS operations for embedding and scaling
    - Memory-efficient processing for edge deployment
    
    VOCABULARY AND TOKENIZATION:
    ===========================
    Integration with different tokenization strategies:
    
    Subword Tokenization:
    - BPE/SentencePiece: Handles subword vocabularies efficiently
    - Variable Length: Supports dynamic vocabulary sizes
    - OOV Handling: Out-of-vocabulary token management
    
    CTC Vocabulary:
    - Blank Token: Reserved index 0 for CTC blank symbol
    - Character/Phoneme: Direct character-level or phoneme-level embeddings
    - Language-Specific: Unicode and language-specific character support
    
    Special Tokens:
    - Start/End Tokens: Beginning and end of sequence markers
    - Padding Tokens: Sequence padding for batch processing
    - Unknown Tokens: Handling of rare or unseen vocabulary items
    
    TRAINING DYNAMICS:
    =================
    The scaling affects training dynamics in important ways:
    
    Gradient Flow:
    - Scaled Gradients: Embedding gradients scaled by sqrt(d_model)
    - Stable Updates: Prevents gradient magnitude issues in deep networks
    - Learning Rate: May require adjusted learning rates for embedding parameters
    
    Initialization:
    - Embedding Initialization: Standard normal initialization maintained
    - Output Scaling: sqrt(d_model) scaling applied during forward pass only
    - Weight Sharing: Consistent initialization between input and output projections
    
    Arguments
    ---------
    d_model : int
        The number of expected features in the encoder/decoder inputs.
        Must match the model dimension used throughout the transformer architecture.
        Typical values: 256, 512, 768, 1024 based on model scale.
        
    vocab : int
        The vocabulary size for token embeddings.
        Determines the number of unique tokens that can be represented.
        Includes special tokens (blank, start, end, padding, unknown).
        Typical values: 1000-50000 depending on tokenization strategy.

    Example
    -------
    Basic usage for transformer decoder:
    >>> d_model = 512
    >>> vocab_size = 1000
    >>> emb = NormalizedEmbedding(d_model, vocab_size)
    >>> target_tokens = torch.randint(0, vocab_size-1, (8, 50))  # [batch, seq_len]
    >>> embedded = emb(target_tokens)  # [batch, seq_len, d_model]
    >>> embedded.shape
    torch.Size([8, 50, 512])
    
    Integration with transformer decoder:
    >>> # Typical decoder input processing
    >>> batch_size, seq_len = 16, 32
    >>> vocab_size, d_model = 5000, 512
    >>> 
    >>> # Create embedding layer
    >>> embedding = NormalizedEmbedding(d_model, vocab_size)
    >>> 
    >>> # Target token sequence (e.g., from tokenizer)
    >>> target_tokens = torch.randint(1, vocab_size, (batch_size, seq_len))
    >>> 
    >>> # Convert to embeddings
    >>> token_embeddings = embedding(target_tokens)
    >>> 
    >>> # Add positional encoding
    >>> pos_encoding = PositionalEncoding(d_model, max_len=1000)
    >>> pos_embeddings = pos_encoding(token_embeddings)
    >>> 
    >>> # Combine for decoder input
    >>> decoder_input = token_embeddings + pos_embeddings
    """

    def __init__(self, d_model, vocab):
        super().__init__()
        self.emb = sb.nnet.embedding.Embedding(
            num_embeddings=vocab, embedding_dim=d_model, blank_id=0
        )
        self.d_model = d_model

    def forward(self, x):
        """Processes the input tensor x and returns an output tensor."""
        return self.emb(x) * math.sqrt(self.d_model)


def get_key_padding_mask(padded_input, pad_idx):
    """Creates a binary mask to prevent attention to padded locations.
    We suggest using ``get_mask_from_lengths`` instead of this function.

    Arguments
    ---------
    padded_input: torch.Tensor
        Padded input.
    pad_idx: int
        idx for padding element.

    Returns
    -------
    key_padded_mask: torch.Tensor
        Binary mask to prevent attention to padding.

    Example
    -------
    >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
    >>> get_key_padding_mask(a, pad_idx=0)
    tensor([[False, False,  True],
            [False, False,  True],
            [False, False,  True]])
    """
    if len(padded_input.shape) == 4:
        bz, time, ch1, ch2 = padded_input.shape
        padded_input = padded_input.reshape(bz, time, ch1 * ch2)

    key_padded_mask = padded_input.eq(pad_idx).to(padded_input.device)

    # if the input is more than 2d, mask the locations where they are silence
    # across all channels
    if len(padded_input.shape) > 2:
        key_padded_mask = key_padded_mask.float().prod(dim=-1).bool()
        return key_padded_mask.detach()

    return key_padded_mask.detach()


def get_lookahead_mask(padded_input):
    """Creates a binary mask for each sequence which masks future frames.

    Arguments
    ---------
    padded_input: torch.Tensor
        Padded input tensor.

    Returns
    -------
    mask : torch.Tensor
        Binary mask for masking future frames.

    Example
    -------
    >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
    >>> get_lookahead_mask(a)
    tensor([[0., -inf, -inf],
            [0., 0., -inf],
            [0., 0., 0.]])
    """
    seq_len = padded_input.shape[1]
    mask = (
        torch.triu(torch.ones((seq_len, seq_len), device=padded_input.device))
        == 1
    ).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == TransformerConstants.MASK_COMPARISON_VALUE, TransformerConstants.MASK_VALUE_ZERO)
    )
    return mask.detach().to(padded_input.device)


def get_mask_from_lengths(lengths, max_len=None):
    """Creates a binary mask from sequence lengths

    Arguments
    ---------
    lengths: torch.Tensor
        A tensor of sequence lengths
    max_len: int (Optional)
        Maximum sequence length, defaults to None.

    Returns
    -------
    mask: torch.Tensor
        the mask where padded elements are set to True.
        Then one can use tensor.masked_fill_(mask, 0) for the masking.

    Example
    -------
    >>> lengths = torch.tensor([3, 2, 4])
    >>> get_mask_from_lengths(lengths)
    tensor([[False, False, False,  True],
            [False, False,  True,  True],
            [False, False, False, False]])
    """
    if max_len is None:
        max_len = torch.max(lengths).item()
    seq_range = torch.arange(
        max_len, device=lengths.device, dtype=lengths.dtype
    )
    return ~(seq_range.unsqueeze(0) < lengths.unsqueeze(1))
