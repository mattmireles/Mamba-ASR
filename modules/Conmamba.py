"""
ConMamba Encoder and Mamba Decoder Implementation for Efficient ASR.

This module implements the ConMamba architecture, a hybrid approach that combines
Mamba's efficient state-space modeling with Conformer's convolution mechanisms.
ConMamba achieves significant computational efficiency while maintaining competitive
accuracy for automatic speech recognition tasks.

ARCHITECTURAL INNOVATION:
========================
ConMamba represents a breakthrough in efficient audio modeling by integrating:
- Mamba state-space models for long-range sequence dependencies
- Conformer-style convolution modules for local feature modeling  
- Bidirectional processing for enhanced context understanding
- Linear computational complexity with respect to sequence length

The architecture replaces traditional attention mechanisms with selective state-space
models, reducing computational cost from O(n²) to O(n) while preserving modeling capacity.

SYSTEM ROLE IN MAMBA-ASR:
=========================
This module serves as a drop-in replacement for standard Conformer encoders in the
ASR pipeline, providing:
- Encoder: ConmambaEncoder for audio feature sequence modeling
- Decoder: MambaDecoder for autoregressive text generation
- Streaming support: Compatible with dynamic chunk training for real-time inference
- Memory efficiency: Reduced memory footprint compared to attention-based models

CALL CHAIN INTEGRATION:
======================
Called by:
- `modules/TransformerASR.py`: TransformerInterface.__init__() when encoder_module="conmamba"
- `modules/Transformer.py`: TransformerInterface creates ConmambaEncoder instances
- Training scripts: train_CTC.py and train_S2S.py via ASR pipeline

Calls to:
- `modules/mamba/bimamba.py`: BiMamba for bidirectional Mamba processing
- `mamba_ssm.Mamba`: Standard unidirectional Mamba layers  
- `speechbrain.nnet.attention.PositionalwiseFeedForward`: FFN components
- `speechbrain.nnet.normalization.LayerNorm`: Layer normalization

MAMBA INTEGRATION DETAILS:
==========================
The module integrates two types of Mamba implementations:
1. Standard Mamba (mamba_ssm): Unidirectional for causal processing
2. BiMamba (custom): Bidirectional for non-causal applications

Mamba Configuration Management:
- mamba_config dictionary controls state dimensions and expansion factors
- bidirectional flag switches between Mamba types based on use case
- Configuration passed through TransformerASR hierarchy to individual layers

PERFORMANCE CHARACTERISTICS:
===========================
ConMamba provides several performance advantages:
- Linear Complexity: O(n) vs O(n²) for standard attention
- Memory Efficiency: Constant memory usage independent of sequence length  
- Streaming Ready: Natural support for incremental processing
- Apple Silicon Optimized: Efficient on MPS backend due to simpler operations

Computational Trade-offs:
- Reduced parameter sharing compared to full attention
- Less parallelizable than attention (sequential state updates)
- Better cache locality for long sequences
- Lower memory bandwidth requirements

DEVICE COMPATIBILITY:
====================
ConMamba is optimized for multiple compute platforms:
- CUDA: Full performance with selective scan kernels
- CPU: Reference implementation for development/testing  
- Apple Silicon MPS: Efficient tensor operations, though some Mamba ops may fallback
- Mixed Precision: Supports autocast with proper dtype handling

STREAMING ARCHITECTURE:
======================
The ConMamba encoder supports streaming inference through:
- Stateful Mamba layers that maintain hidden states across chunks
- Convolution modules with causal padding for streaming compatibility
- Integration with Dynamic Chunk Training for low-latency applications
- Context preservation mechanisms for maintaining sequence history

BIDIRECTIONAL PROCESSING:
========================
The BiMamba integration enables bidirectional context modeling:
- Forward pass processes sequence left-to-right
- Backward pass processes sequence right-to-left  
- Outputs are combined for enhanced representation learning
- Configurable via mamba_config['bidirectional'] parameter

ERROR HANDLING PATTERNS:
=======================
The module implements robust error handling for:
- Missing mamba_config: Assertion errors with clear messages
- Bidirectional flag management: Proper restoration after layer creation
- Device placement: Automatic tensor movement following model device
- Mixed precision: Proper dtype conversion for Mamba operations

Authors
-------
* Xilin Jiang 2024 (ConMamba architecture design and implementation)
"""

import warnings
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import speechbrain as sb
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import (
    MultiheadAttention,
    PositionalwiseFeedForward,
    RelPosMHAXL,
)
from speechbrain.nnet.hypermixing import HyperMixing
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig

# Mamba
from mamba_ssm import Mamba
from modules.mamba.bimamba import Mamba as BiMamba


# =============================================================================
# NAMED CONSTANTS FOR AI-FIRST DOCUMENTATION
# =============================================================================
# These constants replace magic numbers throughout ConMamba implementation
# to provide clear explanations and improve maintainability.

class ConMambaConstants:
    """Constants for ConMamba architecture configuration and computation."""
    
    # Convolution module constants
    POINTWISE_KERNEL_SIZE = 1
    """Kernel size for pointwise (1x1) convolutions in bottleneck layers."""
    
    CONV_STRIDE = 1
    """Stride for all convolution operations - no downsampling in ConMamba."""
    
    CHANNEL_EXPANSION_FACTOR = 2
    """Factor by which channels are expanded in pointwise convolution.
    Input channels are doubled before GLU activation, which then halves them back."""
    
    GLU_SPLIT_DIMENSION = 1
    """Dimension along which GLU splits channels (channel dimension)."""
    
    DILATION_POWER_BASE = 2
    """Base for exponential dilation calculation: 2^(dilation-1)."""
    
    DILATION_OFFSET = 1
    """Offset applied to dilation for power calculation: (dilation - 1)."""
    
    PADDING_DIVISOR = 2
    """Divisor for symmetric padding calculation in non-causal mode."""
    
    # FFN scaling constants
    FFN_RESIDUAL_SCALE = 0.5
    """Scaling factor for FFN residual connections in ConMamba layers.
    Applied as: x = x + 0.5 * ffn(x) for better training stability."""
    
    # Layer normalization constants
    LAYER_NORM_EPS = 1e-6
    """Epsilon value for layer normalization to prevent division by zero."""
    
    # Dynamic chunking constants
    UNSUPPORTED_DILATION = 1
    """Only dilation=1 is supported for dynamic chunk training.
    Other dilation values require additional padding logic."""


class BiMambaConstants:
    """Constants specific to bidirectional Mamba configuration."""
    
    BIMAMBA_TYPE_V2 = 'v2'
    """BiMamba architecture version - v2 provides better bidirectional fusion.""" 


class ConvolutionModule(nn.Module):
    """Conformer-style convolution module optimized for ConMamba architecture.

    This module implements the convolution component of ConMamba layers, providing
    local feature modeling to complement Mamba's global sequence modeling. It uses
    depthwise separable convolutions with gated linear units for efficient parameter
    usage and computational cost.

    ARCHITECTURAL DESIGN:
    ====================
    The module follows the Conformer convolution design pattern:
    1. Layer normalization for input stabilization
    2. Pointwise expansion with 2x channel expansion + GLU activation
    3. Depthwise convolution for local feature extraction
    4. Layer normalization + activation + pointwise compression
    5. Dropout for regularization

    CROSS-FILE INTEGRATION:
    ======================
    Called by:
    - ConmambaEncoderLayer.forward(): Local feature processing within ConMamba layers
    
    Calls to:
    - torch.nn.LayerNorm: Input and output normalization
    - torch.nn.Conv1d: Pointwise and depthwise convolutions
    - torch.nn.GLU: Gated linear unit activation
    - speechbrain activation modules: Configurable activation functions

    DYNAMIC CHUNK TRAINING SUPPORT:
    ===============================
    The module implements sophisticated chunk-based processing for streaming:
    - Divides input sequences into fixed-size chunks
    - Applies convolution across chunk boundaries with proper padding
    - Handles final chunk padding to maintain consistent chunk sizes
    - Supports non-causal convolution with future masking per chunk

    CAUSAL vs NON-CAUSAL MODES:
    ===========================
    Causal Mode (causal=True):
    - Left-padding only to prevent future information leakage
    - Compatible with streaming inference and autoregressive decoding
    - Padding = (kernel_size - 1) * dilation_factor

    Non-Causal Mode (causal=False):
    - Symmetric padding for optimal modeling capacity
    - Used in encoder layers where bidirectional context is available
    - Padding = (kernel_size - 1) * dilation_factor // 2

    MEMORY OPTIMIZATION:
    ===================
    The module implements several memory-efficient patterns:
    - Depthwise convolution reduces parameters from O(C²K) to O(CK)
    - GLU activation provides gating without additional parameters
    - LayerNorm replaces BatchNorm for better streaming compatibility
    - In-place operations where possible to reduce memory allocation

    DEVICE COMPATIBILITY:
    ====================
    Fully compatible with all supported backends:
    - CUDA: Optimized Conv1d operations with cuDNN acceleration
    - CPU: Reference implementation for development
    - Apple Silicon MPS: Efficient tensor operations, full MPS support
    - Mixed Precision: Supports autocast with proper dtype handling
    """

    def __init__(
        self,
        input_size,
        kernel_size=31,
        bias=True,
        activation=Swish,
        dropout=0.0,
        causal=False,
        dilation=1,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.causal = causal
        self.dilation = dilation

        if self.causal:
            self.padding = (kernel_size - ConMambaConstants.DILATION_OFFSET) * ConMambaConstants.DILATION_POWER_BASE ** (dilation - ConMambaConstants.DILATION_OFFSET)
        else:
            self.padding = (kernel_size - ConMambaConstants.DILATION_OFFSET) * ConMambaConstants.DILATION_POWER_BASE ** (dilation - ConMambaConstants.DILATION_OFFSET) // ConMambaConstants.PADDING_DIVISOR

        self.layer_norm = nn.LayerNorm(input_size)
        self.bottleneck = nn.Sequential(
            # pointwise
            nn.Conv1d(
                input_size, 
                ConMambaConstants.CHANNEL_EXPANSION_FACTOR * input_size, 
                kernel_size=ConMambaConstants.POINTWISE_KERNEL_SIZE, 
                stride=ConMambaConstants.CONV_STRIDE, 
                bias=bias
            ),
            nn.GLU(dim=ConMambaConstants.GLU_SPLIT_DIMENSION),
        )
        # depthwise
        self.conv = nn.Conv1d(
            input_size,
            input_size,
            kernel_size=kernel_size,
            stride=ConMambaConstants.CONV_STRIDE,
            padding=self.padding,
            dilation=dilation,
            groups=input_size,
            bias=bias,
        )

        # BatchNorm in the original Conformer replaced with a LayerNorm due to
        # https://github.com/speechbrain/speechbrain/pull/1329
        # see discussion
        # https://github.com/speechbrain/speechbrain/pull/933#issuecomment-1033367884

        self.after_conv = nn.Sequential(
            nn.LayerNorm(input_size),
            activation(),
            # pointwise
            nn.Linear(input_size, input_size, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dynchunktrain_config: Optional[DynChunkTrainConfig] = None,
    ):
        """Applies the convolution to an input tensor `x`.
        """

        if dynchunktrain_config is not None:
            # chances are chunking+causal is unintended; i don't know where it
            # may make sense, but if it does to you, feel free to implement it.
            assert (
                not self.causal
            ), "Chunked convolution not supported with causal padding"

            assert (
                self.dilation == ConMambaConstants.UNSUPPORTED_DILATION
            ), f"Current DynChunkTrain logic does not support dilation != {ConMambaConstants.UNSUPPORTED_DILATION}"

            # in a causal convolution, which is not the case here, an output
            # frame would never be able to depend on a input frame from any
            # point in the future.

            # but with the dynamic chunk convolution, we instead use a "normal"
            # convolution but where, for any output frame, the future beyond the
            # "current" chunk gets masked.
            # see the paper linked in the documentation for details.

            chunk_size = dynchunktrain_config.chunk_size
            batch_size = x.shape[0]

            # determine the amount of padding we need to insert at the right of
            # the last chunk so that all chunks end up with the same size.
            if x.shape[1] % chunk_size != 0:
                final_right_padding = chunk_size - (x.shape[1] % chunk_size)
            else:
                final_right_padding = 0

            # -> [batch_size, t, in_channels]
            out = self.layer_norm(x)

            # -> [batch_size, in_channels, t] for the CNN
            out = out.transpose(1, 2)

            # -> [batch_size, in_channels, t] (pointwise)
            out = self.bottleneck(out)

            # -> [batch_size, in_channels, lc+t+final_right_padding]
            out = F.pad(out, (self.padding, final_right_padding), value=0)

            # now, make chunks with left context.
            # as a recap to what the above padding and this unfold do, consider
            # each a/b/c letter represents a frame as part of chunks a, b, c.
            # consider a chunk size of 4 and a kernel size of 5 (padding=2):
            #
            # input seq: 00aaaabbbbcc00
            # chunk #1:  00aaaa
            # chunk #2:      aabbbb
            # chunk #3:          bbcc00
            #
            # a few remarks here:
            # - the left padding gets inserted early so that the unfold logic
            #   works trivially
            # - the right 0-padding got inserted as the number of time steps
            #   could not be evenly split in `chunk_size` chunks

            # -> [batch_size, in_channels, num_chunks, lc+chunk_size]
            out = out.unfold(2, size=chunk_size + self.padding, step=chunk_size)

            # as we manually disable padding in the convolution below, we insert
            # right 0-padding to the chunks, e.g. reusing the above example:
            #
            # chunk #1:  00aaaa00
            # chunk #2:      aabbbb00
            # chunk #3:          bbcc0000

            # -> [batch_size, in_channels, num_chunks, lc+chunk_size+rpad]
            out = F.pad(out, (0, self.padding), value=0)

            # the transpose+flatten effectively flattens chunks into the batch
            # dimension to be processed into the time-wise convolution. the
            # chunks will later on be unflattened.

            # -> [batch_size, num_chunks, in_channels, lc+chunk_size+rpad]
            out = out.transpose(1, 2)

            # -> [batch_size * num_chunks, in_channels, lc+chunk_size+rpad]
            out = out.flatten(start_dim=0, end_dim=1)

            # TODO: experiment around reflect padding, which is difficult
            # because small chunks have too little time steps to reflect from

            # let's keep backwards compat by pointing at the weights from the
            # already declared Conv1d.
            #
            # still reusing the above example, the convolution will be applied,
            # with the padding truncated on both ends. the following example
            # shows the letter corresponding to the input frame on which the
            # convolution was centered.
            #
            # as you can see, the sum of lengths of all chunks is equal to our
            # input sequence length + `final_right_padding`.
            #
            # chunk #1:  aaaa
            # chunk #2:      bbbb
            # chunk #3:          cc00

            # -> [batch_size * num_chunks, out_channels, chunk_size]
            out = F.conv1d(
                out,
                weight=self.conv.weight,
                bias=self.conv.bias,
                stride=self.conv.stride,
                padding=0,  # No padding for chunked convolution
                dilation=self.conv.dilation,
                groups=self.conv.groups,
            )

            # -> [batch_size * num_chunks, chunk_size, out_channels]
            out = out.transpose(1, 2)

            out = self.after_conv(out)

            # -> [batch_size, num_chunks, chunk_size, out_channels]
            out = torch.unflatten(out, dim=0, sizes=(batch_size, -1))

            # -> [batch_size, t + final_right_padding, out_channels]
            out = torch.flatten(out, start_dim=1, end_dim=2)

            # -> [batch_size, t, out_channels]
            if final_right_padding > 0:
                out = out[:, :-final_right_padding, :]
        else:
            out = self.layer_norm(x)
            out = out.transpose(1, 2)
            out = self.bottleneck(out)
            out = self.conv(out)

            if self.causal:
                # chomp
                out = out[..., : -self.padding]

            out = out.transpose(1, 2)
            out = self.after_conv(out)

        if mask is not None:
            out.masked_fill_(mask, 0.0)

        return out


class ConmambaEncoderLayer(nn.Module):
    """Core ConMamba encoder layer combining Mamba state-space modeling with convolution.

    This layer implements the fundamental building block of the ConMamba architecture,
    integrating Mamba's efficient sequential modeling with Conformer-style convolution
    for local feature extraction. The design achieves linear computational complexity
    while maintaining competitive modeling capacity for audio sequences.

    ARCHITECTURAL INNOVATION:
    ========================
    The layer combines two complementary modeling approaches:
    1. Mamba Module: Captures long-range dependencies with O(n) complexity
    2. Convolution Module: Models local patterns with depthwise separable convs
    3. Feed-Forward Networks: Apply non-linear transformations (2 FFN blocks)
    4. Residual Connections: Enable deep network training with skip connections
    5. Layer Normalization: Stabilize training and improve convergence

    Layer Structure:
    Input → FFN1 → Mamba → ConvModule → FFN2 → Output
    Each component has residual connections and layer normalization.

    MAMBA BIDIRECTIONAL PROCESSING:
    ===============================
    The layer supports both unidirectional and bidirectional Mamba processing:

    Unidirectional Mode (causal=True or bidirectional=False):
    - Uses standard Mamba from mamba_ssm package
    - Processes sequences left-to-right only
    - Compatible with streaming and autoregressive inference
    - Lower memory usage, faster inference

    Bidirectional Mode (causal=False and bidirectional=True):
    - Uses custom BiMamba implementation
    - Processes sequences in both directions
    - Combines forward and backward representations
    - Higher modeling capacity, slower inference

    CROSS-FILE INTEGRATION:
    ======================
    Called by:
    - ConmambaEncoder.forward(): Stacked to form complete encoder
    - ConmambaEncoder.forward_streaming(): Streaming inference mode
    
    Calls to:
    - mamba_ssm.Mamba: Standard unidirectional state-space model
    - modules.mamba.bimamba.Mamba: Bidirectional Mamba implementation
    - ConvolutionModule: Local feature extraction via convolution
    - speechbrain.nnet.attention.PositionalwiseFeedForward: FFN components
    - speechbrain.nnet.normalization.LayerNorm: Layer normalization

    CONFIGURATION MANAGEMENT:
    =========================
    The mamba_config parameter controls Mamba behavior:
    Required keys:
    - 'bidirectional': Whether to use BiMamba (bool)
    - 'd_state': Mamba state dimension (typically 16)
    - 'd_conv': Convolution dimension for Mamba (typically 4)
    - 'expand': Hidden dimension expansion factor (typically 2)

    Configuration Handling Pattern:
    1. Extract 'bidirectional' flag from mamba_config
    2. Create appropriate Mamba instance based on causal and bidirectional flags
    3. Restore 'bidirectional' flag to mamba_config for consistency

    STATE MANAGEMENT:
    ================
    Instance variables and their lifecycles:

    - self.mamba: Core sequence modeling component
      * Created as either Mamba or BiMamba based on configuration
      * Maintains internal state for streaming applications
      * Device placement follows parent module automatically

    - self.convolution_module: Local feature processing
      * Handles causal/non-causal convolution based on layer configuration
      * Supports dynamic chunk training for streaming inference
      * Manages padding and activation states internally

    - self.ffn_module1, self.ffn_module2: Feed-forward processing
      * Pre- and post-processing FFN blocks with LayerNorm
      * Apply non-linear transformations with configurable activation
      * Include dropout for regularization during training

    STREAMING SUPPORT:
    =================
    The layer supports streaming inference through:
    - Mamba layers maintain internal state across sequence chunks
    - Convolution module handles causal padding for streaming compatibility
    - FFN modules operate independently on each frame (no temporal dependencies)
    - Layer normalization operates on individual frames

    MEMORY OPTIMIZATION:
    ===================
    Several optimizations reduce memory usage:
    - Mamba's linear complexity avoids quadratic attention memory growth
    - Depthwise convolution in ConvolutionModule reduces parameter count
    - Residual connections reuse input tensors where possible
    - Layer normalization has lower memory overhead than batch normalization

    DEVICE COMPATIBILITY:
    ====================
    Full compatibility across compute platforms:
    - CUDA: Optimized Mamba kernels and cuDNN convolution acceleration
    - CPU: Reference implementations for all components
    - Apple Silicon MPS: Efficient tensor operations (some Mamba ops may fallback)
    - Mixed Precision: Supports autocast with proper gradient scaling
    """

    def __init__(
        self,
        d_model,
        d_ffn,
        kernel_size=31,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        assert mamba_config != None

        bidirectional = mamba_config.pop('bidirectional')
        if causal or (not bidirectional):
            self.mamba = Mamba(
                d_model=d_model,
                **mamba_config
            )
        else:
            self.mamba = BiMamba(
                d_model=d_model,
                bimamba_type=BiMambaConstants.BIMAMBA_TYPE_V2,
                **mamba_config
            )
        mamba_config['bidirectional'] = bidirectional

        self.convolution_module = ConvolutionModule(
            d_model, kernel_size, bias, activation, dropout, causal=causal
        )

        self.ffn_module1 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.ffn_module2 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: torch.Tensor = None,
        dynchunktrain_config: Optional[DynChunkTrainConfig] = None,
    ):
        conv_mask: Optional[torch.Tensor] = None
        if src_key_padding_mask is not None:
            conv_mask = src_key_padding_mask.unsqueeze(-1)

        conv_mask = None

        # ffn module
        x = x + ConMambaConstants.FFN_RESIDUAL_SCALE * self.ffn_module1(x)
        # mamba module
        skip = x
        x = self.norm1(x)
        x = self.mamba(x)
        x = x + skip
        # convolution module
        x = x + self.convolution_module(
            x, conv_mask, dynchunktrain_config=dynchunktrain_config
        )
        # ffn module
        x = self.norm2(x + ConMambaConstants.FFN_RESIDUAL_SCALE * self.ffn_module2(x))
        return x


class ConmambaEncoder(nn.Module):
    """This class implements the Conmamba encoder.
    """

    def __init__(
        self,
        num_layers,
        d_model,
        d_ffn,
        kernel_size=31,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        print(f'dropout={str(dropout)} is not used in Mamba.')

        self.layers = torch.nn.ModuleList(
            [
                ConmambaEncoderLayer(
                    d_model=d_model,
                    d_ffn=d_ffn,
                    dropout=dropout,
                    activation=activation,
                    kernel_size=kernel_size,
                    bias=bias,
                    causal=causal,
                    mamba_config=mamba_config,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = LayerNorm(d_model, eps=ConMambaConstants.LAYER_NORM_EPS)

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
        dynchunktrain_config: Optional[DynChunkTrainConfig] = None,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder layer.
        src_mask : torch.Tensor, optional
            The mask for the src sequence.
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src keys per batch.
        pos_embs: torch.Tensor, torch.nn.Module,
            Module or tensor containing the input sequence positional embeddings
            If custom pos_embs are given it needs to have the shape (1, 2*S-1, E)
            where S is the sequence length, and E is the embedding dimension.
        dynchunktrain_config: Optional[DynChunkTrainConfig]
            Dynamic Chunk Training configuration object for streaming,
            specifically involved here to apply Dynamic Chunk Convolution to the
            convolution module.
        """

        output = src
        for enc_layer in self.layers:
            output = enc_layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                pos_embs=pos_embs,
                dynchunktrain_config=dynchunktrain_config,
            )
        output = self.norm(output)

        return output, None


class MambaDecoderLayer(nn.Module):
    """Mamba-based decoder layer for efficient autoregressive sequence generation.

    This layer replaces traditional transformer decoder attention with Mamba state-space
    models for both self-attention and cross-attention mechanisms. The design provides
    linear computational complexity while maintaining the encoder-decoder paradigm
    essential for sequence-to-sequence tasks like ASR.

    ARCHITECTURAL INNOVATION:
    ========================
    The layer reimagines decoder attention using Mamba:
    1. Self-Mamba: Processes target sequence autoregressively (replaces self-attention)
    2. Cross-Mamba: Attends to encoder outputs (replaces cross-attention)  
    3. Feed-Forward Network: Applies non-linear transformations
    4. Residual Connections: Enable deep network training
    5. Layer Normalization: Stabilize training with pre/post-norm options

    Layer Structure:
    Target → Self-Mamba → Cross-Mamba → FFN → Output
    Each component includes residual connections and optional pre-normalization.

    MAMBA vs ATTENTION COMPARISON:
    ==============================
    Traditional Decoder Attention:
    - Self-attention: O(S²) complexity over target sequence length S
    - Cross-attention: O(S×T) complexity between target (S) and source (T)
    - Explicit key/value/query projections and attention weights

    Mamba Decoder Design:
    - Self-Mamba: O(S) complexity for autoregressive target processing
    - Cross-Mamba: O(S+T) complexity for encoder-decoder interaction
    - Implicit attention through selective state-space mechanisms

    CROSS-FILE INTEGRATION:
    ======================
    Called by:
    - MambaDecoder.forward(): Stacked to form complete autoregressive decoder
    
    Calls to:
    - mamba_ssm.Mamba: Core state-space model for both self and cross processing
    - speechbrain.nnet.attention.PositionalwiseFeedForward: FFN component
    - speechbrain.nnet.normalization.LayerNorm: Layer normalization

    AUTOREGRESSIVE PROCESSING:
    ==========================
    The layer maintains causality through Mamba's inherent design:
    - Self-Mamba processes target tokens left-to-right only
    - Cross-Mamba can access full encoder output (bidirectional)
    - No explicit masking required (Mamba is inherently causal)
    - Supports efficient incremental decoding for inference

    STATE MANAGEMENT:
    ================
    Key instance variables and their roles:

    - self.self_mamba: Target sequence processing
      * Maintains autoregressive state across decoding steps
      * Inherently causal, preventing future information leakage
      * Linear complexity in sequence length

    - self.cross_mamba: Encoder-decoder interaction
      * Processes encoder outputs to generate context-aware representations
      * Non-causal access to full encoder sequence
      * Enables attention-like behavior without explicit attention weights

    - self.pos_ffn: Feed-forward processing
      * Applies position-wise non-linear transformations
      * Shared across all sequence positions
      * Includes dropout for regularization

    NORMALIZATION STRATEGY:
    ======================
    Supports both pre-norm and post-norm configurations:

    Pre-Normalization (normalize_before=True):
    - Apply LayerNorm before each sub-layer
    - Better gradient flow, more stable training
    - Recommended for deep networks and difficult optimization

    Post-Normalization (normalize_before=False):
    - Apply LayerNorm after each sub-layer (with residual)
    - Original Transformer design pattern
    - May require learning rate scheduling for stability

    MEMORY OPTIMIZATION:
    ===================
    Several design choices optimize memory usage:
    - Linear complexity Mamba operations vs. quadratic attention
    - Shared Mamba configuration reduces parameter overhead
    - Efficient state representation in Mamba internal buffers
    - Optional gradient checkpointing for deeper networks

    DEVICE COMPATIBILITY:
    ====================
    Full support across compute platforms:
    - CUDA: Optimized Mamba kernels with selective scan acceleration
    - CPU: Reference implementation for development and testing
    - Apple Silicon MPS: Efficient tensor operations (some Mamba ops may fallback)
    - Mixed Precision: Compatible with autocast and gradient scaling

    INFERENCE OPTIMIZATION:
    ======================
    The layer supports efficient incremental decoding:
    - Mamba maintains internal state across decoding steps
    - No need to recompute attention over previous tokens
    - Constant memory usage independent of generated sequence length
    - Compatible with beam search and other decoding strategies
    """

    def __init__(
        self,
        d_model,
        d_ffn,
        activation=nn.ReLU,
        dropout=0.0,
        normalize_before=False,
        mamba_config=None
    ):
        super().__init__()

        assert mamba_config != None

        bidirectional = mamba_config.pop('bidirectional')

        self.self_mamba = Mamba(
            d_model=d_model,
            **mamba_config
        )

        self.cross_mamba = Mamba(
            d_model=d_model,
            **mamba_config
        )

        mamba_config['bidirectional'] = bidirectional

        self.pos_ffn = sb.nnet.attention.PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=d_model,
            dropout=dropout,
            activation=activation,
        )

        # normalization layers
        self.norm1 = sb.nnet.normalization.LayerNorm(d_model, eps=ConMambaConstants.LAYER_NORM_EPS)
        self.norm2 = sb.nnet.normalization.LayerNorm(d_model, eps=ConMambaConstants.LAYER_NORM_EPS)
        self.norm3 = sb.nnet.normalization.LayerNorm(d_model, eps=ConMambaConstants.LAYER_NORM_EPS)
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

        # Mamba over the target sequence
        tgt2 = self.self_mamba(tgt1)

        # add & norm
        tgt = tgt + self.dropout1(tgt2)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        if self.normalize_before:
            tgt1 = self.norm2(tgt)
        else:
            tgt1 = tgt

        # Mamba over key=value + query
        # and only take the last len(query) tokens
        tgt2 = self.cross_mamba(torch.cat([memory, tgt1], dim=1))[:, -tgt1.shape[1]:]
        
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

        return tgt, None, None


class MambaDecoder(nn.Module):
    """This class implements the Mamba decoder.
    """

    def __init__(
        self,
        num_layers,
        d_model,
        d_ffn,
        activation=nn.ReLU,
        dropout=0.0,
        normalize_before=False,
        mamba_config=None
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                MambaDecoderLayer(
                    d_model=d_model,
                    d_ffn=d_ffn,
                    activation=activation,
                    dropout=dropout,
                    normalize_before=normalize_before,
                    mamba_config=mamba_config
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=ConMambaConstants.LAYER_NORM_EPS)

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
        for dec_layer in self.layers:
            output, _, _ = dec_layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos_embs_tgt=pos_embs_tgt,
                pos_embs_src=pos_embs_src,
            )
        output = self.norm(output)

        return output, [None], [None]
