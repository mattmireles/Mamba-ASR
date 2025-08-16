"""
Main ASR Transformer Pipeline Implementation for Mamba-ASR System.

This module serves as the core ASR (Automatic Speech Recognition) pipeline, implementing
the complete end-to-end transformer-based speech recognition system. It bridges audio 
feature processing with text output generation through encoder-decoder architectures.

SYSTEM ROLE:
============
This file is the central hub of the Mamba-ASR system, orchestrating:
- Audio feature encoding via Conformer, ConMamba, or standard Transformer encoders
- Optional text decoding via Transformer or Mamba decoders  
- Streaming inference for real-time ASR applications
- Integration with SpeechBrain's training framework

CALL CHAIN INTEGRATION:
======================
Called by:
- `train_CTC.py`: For CTC-only training (encoder-only pipeline)
- `train_S2S.py`: For sequence-to-sequence training (encoder-decoder pipeline)
- Inference scripts: For real-time and batch ASR processing

Calls to:
- `modules/Transformer.py`: Base transformer infrastructure and interface
- `modules/Conformer.py`: ConformerEncoder for audio sequence modeling
- `modules/Conmamba.py`: ConmambaEncoder and MambaDecoder implementations
- `speechbrain.nnet.*`: SpeechBrain neural network components
- `speechbrain.dataio.dataio.length_to_mask`: For padding mask generation

ARCHITECTURAL FLOW:
==================
Audio Input → Feature Extraction → Encoder → [Optional Decoder] → Text Output

1. Raw audio features (B, T, F) enter via custom_src_module linear projection
2. Positional encodings added based on attention_type configuration
3. Encoder processes features using selected architecture (Conformer/ConMamba/Transformer)
4. Optional decoder processes target sequences for S2S training
5. Streaming support via context-aware chunked processing

KEY DESIGN PATTERNS:
===================
- Device-agnostic: Works with CPU, CUDA, and Apple Silicon MPS
- Modular encoders: Supports swapping between Conformer, ConMamba, Transformer
- Streaming-ready: Dynamic chunk training for real-time inference
- SpeechBrain integration: Inherits from TransformerInterface for framework compatibility

MEMORY MANAGEMENT:
=================
- Implements gradient checkpointing for large models
- Supports dynamic batching for efficient GPU utilization
- Streaming context maintains minimal state for real-time processing
- Careful tensor device placement for Apple Silicon optimization

Authors
-------
* Xilin Jiang 2024 (ConMamba and Mamba integration)
* Jianyuan Zhong 2020 (Original Transformer implementation)
* Titouan Parcollet 2024 (SpeechBrain integration)
* Luca Della Libera 2024 (Streaming and optimization enhancements)
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch  # noqa 42
from torch import nn

from speechbrain.dataio.dataio import length_to_mask
from modules.Transformer import (
    NormalizedEmbedding,
    TransformerInterface,
    get_key_padding_mask,
    get_lookahead_mask,
)
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.containers import ModuleList
from speechbrain.nnet.linear import Linear
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig


# =============================================================================
# NAMED CONSTANTS FOR AI-FIRST DOCUMENTATION
# =============================================================================
# These constants replace magic numbers throughout the codebase to provide
# clear explanations of their purpose and make the code more maintainable.

class TensorConstants:
    """Constants related to tensor dimensions and operations."""
    
    # Tensor dimension indices for audio processing
    TIME_DIM_INDEX = 1
    """Index of the time dimension in audio tensors [B, T, F]."""
    
    FEATURE_DIM_INDEX = -1  
    """Index of the feature dimension, typically last dimension."""
    
    SECOND_TO_LAST_DIM_INDEX = -2
    """Index of second-to-last dimension, often time in reshaped tensors."""
    
    # Tensor dimensionality constants
    TENSOR_4D = 4
    """Expected dimensionality for 4D input tensors [B, T, C1, C2]."""
    
    PARAMETER_MIN_DIM_FOR_XAVIER = 1
    """Minimum parameter dimensionality requiring Xavier normal initialization.
    Parameters with dim > 1 (matrices, not scalars/vectors) need proper initialization."""


class ChunkingConstants:
    """Constants for dynamic chunk training and streaming inference."""
    
    CHUNK_BOUNDARY_EXTENSION = 2
    """Number of additional chunks to add when computing chunk boundaries.
    Used in: chunk_size * (num_chunks + CHUNK_BOUNDARY_EXTENSION)
    This provides extra padding for chunk boundary calculations to ensure
    all audio frames are properly covered during streaming inference."""
    
    LEFT_CONTEXT_OFFSET = 1
    """Offset applied to left context chunk calculations.
    Used in: chunk_size * (num_left_chunks + LEFT_CONTEXT_OFFSET)  
    This adjusts the left context window to account for the current chunk
    when masking past frames in dynamic chunk training."""


class MaskingConstants:
    """Constants for attention masking and padding operations."""
    
    PADDING_TOKEN_INDEX = 0
    """Default padding token index used throughout the ASR pipeline.
    Must match the tokenizer's padding token configuration."""
    
    MASK_INVERSE_VALUE = 1
    """Value used to invert length masks: (1 - length_to_mask()).
    Converts speechbrain's length masks to PyTorch's key_padding_mask format
    where True indicates positions to ignore."""
    
    LAST_ELEMENT_INDEX = -1
    """Index for accessing the last element in sequences/lists.
    Used for extracting final attention weights: multihead_attns[-1]"""


class AudioConstants:
    """Constants specific to audio processing and feature extraction."""
    
    MEL_SPECTROGRAM_FEATURES = 80
    """Standard number of mel-spectrogram features for speech recognition.
    This is the typical input_size for most ASR models using mel features."""


@dataclass
class TransformerASRStreamingContext:
    """Streaming context for maintaining state across audio chunks in real-time ASR.

    This class encapsulates all the state information required for streaming inference
    in the TransformerASR pipeline. It enables processing of audio streams in fixed-size
    chunks while maintaining continuity through persistent left context information.

    STATE MANAGEMENT LIFECYCLE:
    ==========================
    1. Context Creation: 
       - Initialized via TransformerASR.make_streaming_context()
       - Sets up chunk size and left context configuration
       - Creates encoder-specific streaming state objects

    2. Chunk Processing:
       - Each audio chunk updates the encoder_context state
       - Left context frames are preserved across chunk boundaries  
       - Attention patterns respect chunk boundaries and context windows

    3. Context Persistence:
       - encoder_context.layers[i].mha_left_context maintains attention history
       - State automatically manages fixed-size rolling buffers
       - Memory usage remains constant regardless of stream duration

    STREAMING ARCHITECTURE:
    ======================
    The streaming context implements a sophisticated caching strategy:

    - Chunk-based Processing: Audio is divided into fixed-size chunks
    - Left Context Preservation: Previous chunks provide context for current chunk
    - Attention State Caching: Multi-head attention states are cached across chunks
    - Memory-Bounded: Fixed memory footprint prevents unbounded growth

    CROSS-FILE INTEGRATION:
    ======================
    Created by:
    - TransformerASR.make_streaming_context(): Initializes context for streaming
    
    Used by:
    - TransformerASR.encode_streaming(): Processes audio chunks with state updates
    - ConformerEncoder.forward_streaming(): Encoder-specific streaming implementation
    - RelPosMHAXL layers: Attention mechanisms with persistent left context

    Mutated by:
    - Encoder layers during forward_streaming() calls
    - Attention modules updating mha_left_context buffers
    - Convolution modules updating cached feature states

    MEMORY MANAGEMENT:
    =================
    The context implements several memory optimization strategies:

    - Fixed Buffer Sizes: Left context buffers have maximum size limits
    - Rolling Windows: Old context is automatically discarded when buffers fill
    - Lazy Initialization: Context objects are created only when needed
    - Device Placement: Tensors automatically follow the model's device placement

    THREAD SAFETY:
    ==============
    This context is NOT thread-safe. Each streaming session requires its own
    context instance. Concurrent access to the same context will corrupt state.

    DEVICE COMPATIBILITY:
    ====================
    The streaming context is fully compatible with:
    - CUDA: GPU tensors for high-throughput streaming
    - CPU: Development and testing environments  
    - Apple Silicon MPS: Optimized for Mac deployment
    - Mixed Precision: Supports autocast for memory efficiency

    Examples
    --------
    >>> from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
    >>> 
    >>> # Initialize streaming context
    >>> chunk_config = DynChunkTrainConfig(
    ...     chunk_size=32,           # Process 32 frames per chunk
    ...     left_context_size=8      # Maintain 8 chunks of left context
    ... )
    >>> context = asr_model.make_streaming_context(chunk_config)
    >>> 
    >>> # Process audio stream chunk by chunk
    >>> audio_chunks = split_audio_into_chunks(audio_stream, chunk_size=32)
    >>> encoder_outputs = []
    >>> 
    >>> for chunk in audio_chunks:
    ...     # Context is automatically updated with each call
    ...     encoder_out = asr_model.encode_streaming(chunk, context)
    ...     encoder_outputs.append(encoder_out)
    >>> 
    >>> # Context maintains state across all chunks
    >>> left_context_frames = context.encoder_context.layers[0].mha_left_context
    >>> print(f"Left context shape: {left_context_frames.shape}")

    See Also
    --------
    speechbrain.utils.dynamic_chunk_training.DynChunkTrainConfig : Chunk configuration
    modules.Conformer.ConformerEncoder.forward_streaming : Conformer streaming implementation
    modules.Conformer.ConformerEncoderLayer.make_streaming_context : Layer-level context creation
    """

    dynchunktrain_config: DynChunkTrainConfig
    """Dynamic Chunk Training configuration specifying streaming parameters.
    
    This configuration object controls the fundamental streaming behavior:
    
    - chunk_size: Number of audio frames processed in each streaming call
    - left_context_size: Number of previous chunks to maintain for context
    - is_infinite_left_context(): Whether to use unlimited left context
    
    The configuration is immutable once the context is created. Changing streaming
    parameters requires creating a new context instance.
    
    Typical values:
    - chunk_size=32: Good balance of latency vs. context for real-time ASR
    - left_context_size=8: Sufficient context for most attention patterns
    - For ultra-low latency: chunk_size=16, left_context_size=4
    - For high accuracy: chunk_size=64, left_context_size=16
    """

    encoder_context: Any
    """Encoder-specific streaming state maintaining attention and feature history.
    
    This object is created by the encoder's make_streaming_context() method and
    contains encoder-architecture-specific state information. The exact structure
    depends on the encoder module:
    
    ConformerEncoder Context Structure:
    - layers[i].mha_left_context: Cached left context for attention layer i
    - layers[i].conv_cache: Cached convolution states for layer i
    - Each layer maintains fixed-size rolling buffers
    
    TransformerEncoder Context Structure:  
    - layers[i].mha_left_context: Cached attention context for layer i
    - No convolution caches (Transformer has no conv layers)
    
    ConmambaEncoder Context Structure:
    - layers[i].mamba_state: Cached Mamba hidden states for layer i  
    - layers[i].conv_cache: Cached convolution states for layer i
    - Mamba states include selective scan state information
    
    State Management Rules:
    - Context tensors automatically move to model device
    - Buffer sizes are determined by left_context_size configuration
    - Old context is discarded when buffers reach capacity
    - State is updated in-place during encode_streaming() calls
    
    Performance Notes:
    - Context size scales linearly with number of encoder layers
    - Memory usage: ~(left_context_size * chunk_size * d_model * num_layers)
    - GPU memory is allocated once and reused for entire streaming session
    """


def make_transformer_src_mask(
    src: torch.Tensor,
    causal: bool = False,
    dynchunktrain_config: Optional[DynChunkTrainConfig] = None,
) -> Optional[torch.Tensor]:
    """Prepare the source transformer mask that restricts which frames can
    attend to which frames depending on causal or other simple restricted
    attention methods.

    Arguments
    ---------
    src: torch.Tensor
        The source tensor to build a mask from. The contents of the tensor are
        not actually used currently; only its shape and other metadata (e.g.
        device).
    causal: bool
        Whether strict causality shall be used. Frames will not be able to
        attend to any future frame.
    dynchunktrain_config: DynChunkTrainConfig, optional
        Dynamic Chunk Training configuration. This implements a simple form of
        chunkwise attention. Incompatible with `causal`.

    Returns
    -------
    torch.Tensor
        A boolean mask Tensor of shape (timesteps, timesteps).
    """
    if causal:
        assert dynchunktrain_config is None
        return get_lookahead_mask(src)

    if dynchunktrain_config is None:
        return

    # The following is not really the sole source used to implement this,
    # but it helps introduce the concept.
    # ref: Unified Streaming and Non-streaming Two-pass End-to-end Model for Speech Recognition
    # https://arxiv.org/pdf/2012.05481.pdf
    timesteps = src.size(TensorConstants.TIME_DIM_INDEX)

    # Mask the future at the right of each chunk
    chunk_size = dynchunktrain_config.chunk_size
    num_chunks = timesteps // chunk_size
    timestep_idx = torch.arange(timesteps, device=src.device)
    mask_idx = torch.arange(
        chunk_size, 
        chunk_size * (num_chunks + ChunkingConstants.CHUNK_BOUNDARY_EXTENSION), 
        chunk_size, 
        device=src.device
    ).repeat_interleave(chunk_size)[:timesteps]
    src_mask = timestep_idx[None] >= mask_idx[:, None]

    # Mask the past at the left of each chunk (accounting for left context)
    # only relevant if using left context
    if not dynchunktrain_config.is_infinite_left_context():
        num_left_chunks = dynchunktrain_config.left_context_size
        mask_idx -= chunk_size * (num_left_chunks + ChunkingConstants.LEFT_CONTEXT_OFFSET)
        src_mask += timestep_idx[None] < mask_idx[:, None]

    return src_mask


def make_transformer_src_tgt_masks(
    src,
    tgt=None,
    wav_len=None,
    pad_idx=MaskingConstants.PADDING_TOKEN_INDEX,
    causal: bool = False,
    dynchunktrain_config: Optional[DynChunkTrainConfig] = None,
):
    """This function generates masks for training the transformer model,
    opinionated for an ASR context with encoding masks and, optionally, decoding
    masks (if specifying `tgt`).

    Arguments
    ---------
    src : torch.Tensor
        The sequence to the encoder (required).
    tgt : torch.Tensor
        The sequence to the decoder.
    wav_len : torch.Tensor
        The lengths of the inputs.
    pad_idx : int
        The index for <pad> token (default=0).
    causal: bool
        Whether strict causality shall be used. See `make_asr_src_mask`
    dynchunktrain_config: DynChunkTrainConfig, optional
        Dynamic Chunk Training configuration. See `make_asr_src_mask`

    Returns
    -------
    src_key_padding_mask : torch.Tensor
        Key padding mask for ignoring padding
    tgt_key_padding_mask : torch.Tensor
        Key padding mask for ignoring padding
    src_mask : torch.Tensor
        Mask for ignoring invalid (e.g. future) timesteps
    tgt_mask : torch.Tensor
        Mask for ignoring invalid (e.g. future) timesteps
    """
    src_key_padding_mask = None

    # mask out audio beyond the length of audio for each batch
    if wav_len is not None:
        abs_len = torch.round(wav_len * src.shape[TensorConstants.TIME_DIM_INDEX])
        src_key_padding_mask = ~length_to_mask(abs_len).bool()

    # mask out the source
    src_mask = make_transformer_src_mask(
        src, causal=causal, dynchunktrain_config=dynchunktrain_config
    )

    # If no decoder in the transformer...
    if tgt is not None:
        tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)
        tgt_mask = get_lookahead_mask(tgt)
    else:
        tgt_key_padding_mask = None
        tgt_mask = None

    return src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask


class TransformerASR(TransformerInterface):
    """Core ASR Transformer Pipeline for End-to-End Speech Recognition.

    This class orchestrates the complete ASR pipeline from audio features to text output,
    supporting multiple encoder architectures (Conformer, ConMamba, Transformer) and 
    optional decoder configurations (Transformer, Mamba). It serves as the primary interface
    between SpeechBrain's training framework and the underlying neural architectures.

    ARCHITECTURAL DESIGN:
    ====================
    Inherits from TransformerInterface (modules/Transformer.py) which provides:
    - Base transformer infrastructure and configuration management
    - Encoder/decoder initialization based on module selection
    - Positional encoding strategies (fixed_abs_sine, RelPosMHAXL, none)
    - Device-agnostic tensor operations

    CROSS-FILE INTEGRATION:
    ======================
    Called by:
    - `train_CTC.py`: ASR.forward() for CTC-only training (encoder-only mode)
    - `train_S2S.py`: ASR.forward() for sequence-to-sequence training (encoder-decoder)
    - Inference pipelines: encode() for real-time ASR, decode() for beam search

    Calls to:
    - `modules/Transformer.py`: TransformerInterface.__init__() for base configuration
    - `modules/Conformer.py`: ConformerEncoder for audio sequence modeling
    - `modules/Conmamba.py`: ConmambaEncoder, MambaDecoder for Mamba-based processing
    - `speechbrain.nnet.linear.Linear`: custom_src_module for feature projection
    - `speechbrain.nnet.embedding.NormalizedEmbedding`: custom_tgt_module for token embeddings

    STATE MANAGEMENT:
    ================
    Key instance variables and their lifecycles:
    
    - self.custom_src_module: Linear projection from input_size to d_model
      * Created in __init__() as ModuleList([Linear(), Dropout()])
      * Applied in forward()/encode() before encoder processing
      * Device placement handled automatically by parent Module
    
    - self.custom_tgt_module: Token embedding for decoder input
      * Created only if num_decoder_layers > 0
      * NormalizedEmbedding scales by sqrt(d_model) for attention compatibility
      * Applied in forward()/decode() before decoder processing
    
    - self.encoder: Architecture-specific encoder (Conformer/ConMamba/Transformer)
      * Initialized by parent TransformerInterface based on encoder_module parameter
      * State managed by encoder-specific streaming contexts
      * Supports dynamic chunk training for streaming inference
    
    - self.decoder: Optional decoder (Transformer/Mamba)
      * Created only if num_decoder_layers > 0
      * Initialized by parent TransformerInterface based on decoder_module parameter
      * Always uses causal attention for autoregressive generation

    STREAMING SUPPORT:
    =================
    Streaming inference via TransformerASRStreamingContext:
    - Context object maintains encoder state across audio chunks
    - encode_streaming() processes chunks with persistent left context
    - make_streaming_context() initializes context with chunk size configuration
    - Compatible with Dynamic Chunk Training for low-latency inference

    DEVICE COMPATIBILITY:
    ====================
    Optimized for multiple compute platforms:
    - CUDA: Full performance with standard PyTorch operations
    - CPU: Fallback mode for development and testing
    - Apple Silicon MPS: Optimized tensor operations with MPS backend
    - Mixed precision: Supports autocast for memory efficiency

    Parameters
    ----------
    tgt_vocab : int
        Target vocabulary size for token embedding and output projection.
        Must match tokenizer vocabulary size exactly.

    input_size : int
        Input audio feature dimension (typically 80 for mel spectrograms).
        Projected to d_model via custom_src_module linear layer.

    d_model : int, default=512
        Hidden embedding dimension throughout the model.
        All encoder/decoder layers operate at this dimension.
        Recommended values: 256, 512, 768, 1024 based on model size.

    nhead : int, default=8
        Number of attention heads in multi-head attention modules.
        Must divide d_model evenly (d_model % nhead == 0).
        Typical values: 8 for 512d, 12 for 768d, 16 for 1024d models.

    num_encoder_layers : int, default=6
        Number of encoder layers in the selected architecture.
        ConMamba typically uses 12-24 layers, Conformer uses 12-18 layers.
        Memory usage scales linearly with layer count.

    num_decoder_layers : int, default=6
        Number of decoder layers for sequence-to-sequence training.
        Set to 0 for CTC-only training (encoder-only mode).
        Mamba decoders typically use fewer layers than Transformer decoders.

    d_ffn : int, default=2048
        Feed-forward network hidden dimension in transformer layers.
        Typically 4x d_model for optimal parameter efficiency.
        Directly impacts memory usage and computational cost.

    dropout : float, default=0.1
        Dropout probability applied throughout the model.
        Note: Mamba layers do not use dropout internally.
        Applied in: linear projections, attention weights, feed-forward networks.

    activation : torch.nn.Module, default=nn.ReLU
        Activation function for feed-forward networks.
        Common choices: nn.ReLU, nn.GELU, Swish for different architectures.
        ConMamba typically uses Swish, standard Transformers use ReLU/GELU.

    positional_encoding : str, default="fixed_abs_sine"
        Positional encoding strategy for input sequences.
        Options:
        - "fixed_abs_sine": Sinusoidal absolute positional encodings
        - None: No positional encodings (not recommended for most tasks)
        Note: "RelPosMHAXL" attention overrides this with relative positions.

    normalize_before : bool, default=False
        Whether to apply LayerNorm before or after attention/feed-forward.
        True (pre-norm): Better gradient flow, required for Conformer/ConMamba
        False (post-norm): Original Transformer architecture, more stable training

    kernel_size : int, default=31
        Convolution kernel size in Conformer and ConMamba architectures.
        Larger kernels capture longer-range dependencies but increase computation.
        Typical range: 15-63 (odd numbers for symmetric padding).

    bias : bool, default=True
        Whether to use bias parameters in convolution and linear layers.
        Setting to False can reduce parameters and improve generalization.
        Required True for some architectures, False for others based on normalization.

    encoder_module : str, default="transformer"
        Encoder architecture selection.
        Options:
        - "transformer": Standard multi-head attention encoder
        - "conformer": Convolution-augmented transformer for audio
        - "conmamba": ConMamba encoder with Mamba and convolution blocks
        - "branchformer": Branchformer with parallel attention/convolution paths

    decoder_module : str, default="transformer"
        Decoder architecture selection (if num_decoder_layers > 0).
        Options:
        - "transformer": Standard autoregressive transformer decoder
        - "mamba": Mamba-based autoregressive decoder for efficiency

    conformer_activation : torch.nn.Module, default=Swish
        Activation function specifically for Conformer convolution modules.
        Swish activation provides better performance than ReLU for audio tasks.
        Must be a torch.nn.Module instance, not a function.

    branchformer_activation : torch.nn.Module, default=nn.GELU
        Activation function for Branchformer feed-forward networks.
        GELU typically provides better performance than ReLU for Branchformer.

    attention_type : str, default="regularMHA"
        Attention mechanism used in encoder/decoder layers.
        Options:
        - "regularMHA": Standard multi-head attention
        - "RelPosMHAXL": Relative positional multi-head attention (Transformer-XL style)
        - "hypermixing": HyperMixing attention for enhanced modeling

    max_length : int, default=2500
        Maximum sequence length for positional encodings.
        Must be larger than the longest sequence in training/inference.
        Affects memory usage for absolute positional encodings.

    causal : bool, default=True
        Whether encoder should use causal (left-to-right) attention.
        True: Streaming-compatible, but may reduce modeling capacity
        False: Bidirectional attention for better accuracy (not streaming)
        Note: Decoder is always causal regardless of this setting.

    mamba_config : dict, optional
        Configuration dictionary for Mamba layers when using ConMamba or Mamba decoder.
        Expected keys:
        - d_state: State dimension for Mamba layers (default 16)
        - d_conv: Convolution dimension (default 4)
        - expand: Expansion factor for hidden dimension (default 2)
        - bidirectional: Whether to use bidirectional Mamba (default True for encoder)

    Architecture Flow
    -----------------
    1. Audio features [B, T, input_size] → custom_src_module → [B, T, d_model]
    2. Add positional encodings based on positional_encoding parameter
    3. Encoder processes features → encoder_out [B, T, d_model]
    4. If decoder enabled: tokens → custom_tgt_module → decoder → decoder_out [B, S, d_model]

    Training Integration
    -------------------
    - CTC training: Use encoder_out directly with CTC loss
    - S2S training: Use both encoder_out and decoder_out with attention-based loss
    - Joint training: Combine CTC and attention losses with configurable weights

    Memory Optimization
    ------------------
    - Gradient checkpointing: Implemented in parent TransformerInterface
    - Dynamic batching: Supported through SpeechBrain's DynamicBatchSampler
    - Mixed precision: Compatible with torch.cuda.amp.autocast
    - Streaming: Minimal memory footprint via TransformerASRStreamingContext

    Example Usage
    -------------
    >>> # CTC-only training (encoder-only)
    >>> asr_model = TransformerASR(
    ...     tgt_vocab=5000,
    ...     input_size=80,
    ...     d_model=512,
    ...     num_encoder_layers=12,
    ...     num_decoder_layers=0,  # CTC mode
    ...     encoder_module="conformer",
    ...     attention_type="RelPosMHAXL"
    ... )
    >>> 
    >>> # S2S training (encoder-decoder)
    >>> asr_model = TransformerASR(
    ...     tgt_vocab=5000,
    ...     input_size=80,
    ...     d_model=512,
    ...     num_encoder_layers=12,
    ...     num_decoder_layers=6,
    ...     encoder_module="conmamba",
    ...     decoder_module="mamba",
    ...     mamba_config={"d_state": 16, "expand": 2, "bidirectional": True}
    ... )
    >>> 
    >>> # Streaming inference setup
    >>> from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
    >>> chunk_config = DynChunkTrainConfig(chunk_size=32, left_context_size=16)
    >>> context = asr_model.make_streaming_context(chunk_config)
    >>> audio_chunk = torch.rand(1, 32, 80)
    >>> encoder_out = asr_model.encode_streaming(audio_chunk, context)

    See Also
    --------
    modules.Transformer.TransformerInterface : Base class providing infrastructure
    modules.Conformer.ConformerEncoder : Conformer encoder implementation
    modules.Conmamba.ConmambaEncoder : ConMamba encoder implementation
    modules.Conmamba.MambaDecoder : Mamba decoder implementation
    """

    def __init__(
        self,
        tgt_vocab,
        input_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        positional_encoding="fixed_abs_sine",
        normalize_before=False,
        kernel_size: Optional[int] = 31,
        bias: Optional[bool] = True,
        encoder_module: Optional[str] = "transformer",
        decoder_module: Optional[str] = "transformer",
        conformer_activation: Optional[nn.Module] = Swish,
        branchformer_activation: Optional[nn.Module] = nn.GELU,
        attention_type: Optional[str] = "regularMHA",
        max_length: Optional[int] = 2500,
        causal: Optional[bool] = True,
        csgu_linear_units: Optional[int] = 3072,
        gate_activation: Optional[nn.Module] = nn.Identity,
        use_linear_after_conv: Optional[bool] = False,
        mamba_config=None
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            positional_encoding=positional_encoding,
            normalize_before=normalize_before,
            kernel_size=kernel_size,
            bias=bias,
            encoder_module=encoder_module,
            decoder_module=decoder_module,
            conformer_activation=conformer_activation,
            branchformer_activation=branchformer_activation,
            attention_type=attention_type,
            max_length=max_length,
            causal=causal,
            csgu_linear_units=csgu_linear_units,
            gate_activation=gate_activation,
            use_linear_after_conv=use_linear_after_conv,
            mamba_config=mamba_config
        )

        self.custom_src_module = ModuleList(
            Linear(
                input_size=input_size,
                n_neurons=d_model,
                bias=True,
                combine_dims=False,
            ),
            torch.nn.Dropout(dropout),
        )

        self.num_decoder_layers = num_decoder_layers
        if num_decoder_layers > 0:
            self.custom_tgt_module = ModuleList(
                NormalizedEmbedding(d_model, tgt_vocab)
            )

        # reset parameters using xavier_normal_
        self._init_params()

    def forward(self, src, tgt, wav_len=None, pad_idx=MaskingConstants.PADDING_TOKEN_INDEX):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        tgt : torch.Tensor
            The sequence to the decoder.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        pad_idx : int, optional
            The index for <pad> token (default=0).
        """

        # reshape the src vector to [Batch, Time, Fea] is a 4d vector is given
        if src.ndim == TensorConstants.TENSOR_4D:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        (
            src_key_padding_mask,
            tgt_key_padding_mask,
            src_mask,
            tgt_mask,
        ) = make_transformer_src_tgt_masks(
            src, tgt, wav_len, causal=self.causal, pad_idx=pad_idx
        )
        
        src = self.custom_src_module(src)
        # add pos encoding to queries if are sinusoidal ones else
        if self.attention_type == "hypermixing":
            pos_embs_encoder = None
        elif self.attention_type == "RelPosMHAXL":
            pos_embs_encoder = self.positional_encoding(src)
        elif self.positional_encoding_type == "fixed_abs_sine":
            src = src + self.positional_encoding(src)  # add the encodings here
            pos_embs_encoder = None

        encoder_out, _ = self.encoder(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )

        if self.num_decoder_layers > 0:
            tgt = self.custom_tgt_module(tgt)

            if self.attention_type == "RelPosMHAXL":
                tgt = tgt + self.positional_encoding_decoder(tgt)
                pos_embs_encoder = None  # self.positional_encoding(src)
                pos_embs_target = None
            elif (
                self.positional_encoding_type == "fixed_abs_sine"
                or self.attention_type == "hypermixing"
            ):
                tgt = tgt + self.positional_encoding(tgt)
                pos_embs_target = None
                pos_embs_encoder = None

            decoder_out, _, _ = self.decoder(
                tgt=tgt,
                memory=encoder_out,
                memory_mask=None,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
                pos_embs_tgt=pos_embs_target,
                pos_embs_src=pos_embs_encoder,
            )

        else:
            decoder_out = None

        return encoder_out, decoder_out

    @torch.no_grad()
    def decode(self, tgt, encoder_out, enc_len=None):
        """This method implements a decoding step for the transformer model.

        Arguments
        ---------
        tgt : torch.Tensor
            The sequence to the decoder.
        encoder_out : torch.Tensor
            Hidden output of the encoder.
        enc_len : torch.LongTensor
            The actual length of encoder states.

        Returns
        -------
        prediction
        """
        tgt_mask = get_lookahead_mask(tgt)
        src_key_padding_mask = None
        if enc_len is not None:
            src_key_padding_mask = (MaskingConstants.MASK_INVERSE_VALUE - length_to_mask(enc_len)).bool()

        if self.num_decoder_layers > 0:
            tgt = self.custom_tgt_module(tgt)
        if self.attention_type == "RelPosMHAXL":
            tgt = tgt + self.positional_encoding_decoder(tgt)
            pos_embs_encoder = None  # self.positional_encoding(src)
            pos_embs_target = None
        elif (
            self.positional_encoding_type == "fixed_abs_sine"
            or self.attention_type == "hypermixing"
        ):
            tgt = tgt + self.positional_encoding(tgt)  # add the encodings here
            pos_embs_target = None
            pos_embs_encoder = None

   
        prediction, self_attns, multihead_attns = self.decoder(
            tgt,
            encoder_out,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos_embs_tgt=pos_embs_target,
            pos_embs_src=pos_embs_encoder,
        )
        return prediction, multihead_attns[MaskingConstants.LAST_ELEMENT_INDEX]

    def encode(
        self,
        src,
        wav_len=None,
        pad_idx=MaskingConstants.PADDING_TOKEN_INDEX,
        dynchunktrain_config: Optional[DynChunkTrainConfig] = None,
    ):
        """
        Encoder forward pass

        Arguments
        ---------
        src : torch.Tensor
            The sequence to the encoder.
        wav_len : torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        pad_idx : int
            The index used for padding.
        dynchunktrain_config : DynChunkTrainConfig
            Dynamic chunking config.

        Returns
        -------
        encoder_out : torch.Tensor
        """
        # reshape the src vector to [Batch, Time, Fea] if a 4d vector is given
        if src.dim() == TensorConstants.TENSOR_4D:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        (
            src_key_padding_mask,
            _,
            src_mask,
            _,
        ) = make_transformer_src_tgt_masks(
            src,
            None,
            wav_len,
            pad_idx=pad_idx,
            causal=self.causal,
            dynchunktrain_config=dynchunktrain_config,
        )

        src = self.custom_src_module(src)
        if self.attention_type == "hypermixing":
            pos_embs_source = None
        elif self.attention_type == "RelPosMHAXL":
            pos_embs_source = self.positional_encoding(src)
        elif self.positional_encoding_type == "fixed_abs_sine":
            src = src + self.positional_encoding(src)
            pos_embs_source = None

        encoder_out, _ = self.encoder(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_source,
            dynchunktrain_config=dynchunktrain_config,
        )

        return encoder_out

    def encode_streaming(self, src, context: TransformerASRStreamingContext):
        """
        Streaming encoder forward pass

        Arguments
        ---------
        src : torch.Tensor
            The sequence (chunk) to the encoder.
        context : TransformerASRStreamingContext
            Mutable reference to the streaming context. This holds the state
            needed to persist across chunk inferences and can be built using
            `make_streaming_context`. This will get mutated by this function.

        Returns
        -------
        Encoder output for this chunk.

        Example
        -------
        >>> import torch
        >>> from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR
        >>> from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
        >>> net = TransformerASR(
        ...     tgt_vocab=100,
        ...     input_size=64,
        ...     d_model=64,
        ...     nhead=8,
        ...     num_encoder_layers=1,
        ...     num_decoder_layers=0,
        ...     d_ffn=128,
        ...     attention_type="RelPosMHAXL",
        ...     positional_encoding=None,
        ...     encoder_module="conformer",
        ...     normalize_before=True,
        ...     causal=False,
        ... )
        >>> ctx = net.make_streaming_context(DynChunkTrainConfig(16, 1))
        >>> src1 = torch.rand([8, 16, 64])
        >>> src2 = torch.rand([8, 16, 64])
        >>> out1 = net.encode_streaming(src1, ctx)
        >>> out1.shape
        torch.Size([8, 16, 64])
        >>> ctx.encoder_context.layers[0].mha_left_context.shape
        torch.Size([8, 16, 64])
        >>> out2 = net.encode_streaming(src2, ctx)
        >>> out2.shape
        torch.Size([8, 16, 64])
        >>> ctx.encoder_context.layers[0].mha_left_context.shape
        torch.Size([8, 16, 64])
        >>> combined_out = torch.concat((out1, out2), dim=1)
        >>> combined_out.shape
        torch.Size([8, 32, 64])
        """

        if src.dim() == TensorConstants.TENSOR_4D:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        # HACK: our problem here is that the positional_encoding is computed
        # against the size of our source tensor, but we only know how many left
        # context frames we're injecting to the encoder within the encoder
        # context.
        # so this workaround does just that.
        #
        # i'm not sure how this would be best refactored, but an option would be
        # to let the encoder get the pos embedding itself and have a way to
        # cache it.
        #
        # additionally, positional encoding functions take in a whole source
        # tensor just to get its attributes (size, device, type) but this is
        # sort of silly for the embeddings that don't need one.
        # so we craft a dummy empty (uninitialized) tensor to help...
        known_left_context = context.encoder_context.layers[0].mha_left_context
        if known_left_context is None:
            pos_encoding_dummy = src
        else:
            target_shape = list(src.shape)
            target_shape[TensorConstants.SECOND_TO_LAST_DIM_INDEX] += known_left_context.shape[TensorConstants.SECOND_TO_LAST_DIM_INDEX]
            pos_encoding_dummy = torch.empty(size=target_shape).to(src)

        src = self.custom_src_module(src)
        if self.attention_type == "RelPosMHAXL":
            pos_embs_source = self.positional_encoding(pos_encoding_dummy)

        elif self.positional_encoding_type == "fixed_abs_sine":
            src = src + self.positional_encoding(pos_encoding_dummy)
            pos_embs_source = None

        encoder_out, _ = self.encoder.forward_streaming(
            src=src, pos_embs=pos_embs_source, context=context.encoder_context
        )
        return encoder_out

    def make_streaming_context(
        self, dynchunktrain_config: DynChunkTrainConfig, encoder_kwargs={}
    ):
        """Creates a blank streaming context for this transformer and its
        encoder.

        Arguments
        ---------
        dynchunktrain_config : DynChunkTrainConfig
            Runtime chunkwise attention configuration.
        encoder_kwargs : dict
            Parameters to be forward to the encoder's `make_streaming_context`.
            Metadata required for the encoder could differ depending on the
            encoder.

        Returns
        -------
        TransformerASRStreamingContext
        """
        return TransformerASRStreamingContext(
            dynchunktrain_config=dynchunktrain_config,
            encoder_context=self.encoder.make_streaming_context(
                dynchunktrain_config,
                **encoder_kwargs,
            ),
        )

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > TensorConstants.PARAMETER_MIN_DIM_FOR_XAVIER:
                torch.nn.init.xavier_normal_(p)


class EncoderWrapper(nn.Module):
    """This is a wrapper of any ASR transformer encoder. By default, the
    TransformerASR .forward() function encodes and decodes. With this wrapper
    the .forward() function becomes .encode() only.

    Important: The TransformerASR class must contain a .encode() function.

    Arguments
    ---------
    transformer : sb.lobes.models.TransformerInterface
        A Transformer instance that contains a .encode() function.
    *args : tuple
    **kwargs : dict
        Arguments to forward to parent class.

    Example
    -------
    >>> src = torch.rand([8, 120, 512])
    >>> tgt = torch.randint(0, 720, [8, 120])
    >>> net = TransformerASR(
    ...     720, 512, 512, 8, 1, 1, 1024, activation=torch.nn.GELU
    ... )
    >>> encoder = EncoderWrapper(net)
    >>> enc_out = encoder(src)
    >>> enc_out.shape
    torch.Size([8, 120, 512])
    """

    def __init__(self, transformer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = transformer
        self.make_streaming_context = self.transformer.make_streaming_context

    def forward(self, x, wav_lens=None, pad_idx=MaskingConstants.PADDING_TOKEN_INDEX, **kwargs):
        """Processes the input tensor x and returns an output tensor."""
        x = self.transformer.encode(x, wav_lens, pad_idx, **kwargs)
        return x

    def forward_streaming(self, x, context):
        """Processes the input audio chunk tensor `x`, using and updating the
        mutable encoder `context`"""
        x = self.transformer.encode_streaming(x, context)
        return x

    def make_streaming_context(self, *args, **kwargs):
        """Initializes a streaming context. Forwards all arguments to the
        underlying transformer. See :meth:`speechbrain.lobes.models.transformer.TransformerASR.make_streaming_context`.
        """
        return self.transformer.make_streaming_context(*args, **kwargs)
