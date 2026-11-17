# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics neural network modules.

This module provides access to various neural network components used in Ultralytics models, including convolution
blocks, attention mechanisms, transformer components, and detection/segmentation heads.

Examples:
    Visualize a module with Netron
    >>> from ultralytics.nn.modules import Conv
    >>> import torch
    >>> import subprocess
    >>> x = torch.ones(1, 128, 40, 40)
    >>> m = Conv(128, 128)
    >>> f = f"{m._get_name()}.onnx"
    >>> torch.onnx.export(m, x, f)
    >>> subprocess.run(f"onnxslim {f} {f} && open {f}", shell=True, check=True)  # pip install onnxslim
"""

from .block import (
    C1,
    C2,
    C2MSDA,  # Added C2MSDA import
    C2PSA,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    MSDA,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Attention,
    BiLevelRoutingAttention,  # Added BiLevelRoutingAttention import
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3k2_DSConv,  # Added C3k2_DSConv import
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    MaxSigmoidAttnBlock,
    Proto,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    TorchVision,
)
from .conv import (
    CARAFE,  # Added CARAFE to exports
    CBAM,
    WCA,  # Added WCA import
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvBNReLU,  # Added ConvBNReLU to exports
    ConvTranspose,
    DODConv,  # Added DODConv here and in the end.
    DWConv,
    DWConvTranspose2d,
    DySample,  # Added DySample import
    DySnakeConv,  # Added DySnakeConv import
    Focus,
    GAM_Attention,  # Added GAM_Attention import
    GhostConv,
    HetConv,  # Added HetConv to exports
    Index,
    LightConv,
    RepConv,
    ResBlock_CBAM,  # Added ResBlock_CBAM import
    SpatialAttention,
)
from .head import (
    OBB,
    Classify,
    Detect,
    LRPCHead,
    Pose,
    RTDETRDecoder,
    Segment,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    v10Detect,
)
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

__all__ = (
    "AIFI",
    "C1",
    "C2",
    "C2MSDA",  # Added C2MSDA to exports
    "C2PSA",
    "C3",
    "C3TR",
    "CARAFE",  # Added CARAFE to exports
    "CBAM",
    "CIB",
    "DFL",
    "ELAN1",
    "MLP",
    "MSDA",
    "OBB",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "WCA",  # Added WCA to exports
    "A2C2f",
    "AConv",
    "ADown",
    "Attention",
    "BNContrastiveHead",
    "BiLevelRoutingAttention",  # Added BiLevelRoutingAttention to exports
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3k2_DSConv",  # Added C3k2_DSConv to exports
    "C3x",
    "CBFuse",
    "CBLinear",
    "ChannelAttention",
    "Classify",
    "Concat",
    "ContrastiveHead",
    "Conv",
    "Conv2",
    "ConvBNReLU",  # Added ConvBNReLU to exports
    "ConvTranspose",
    "DODConv",  # Added DODConv to exports
    "DWConv",
    "DWConvTranspose2d",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "Detect",
    "DySample",  # Added DySample to exports
    "DySnakeConv",  # Added DySnakeConv to exports
    "Focus",
    "GAM_Attention",  # Added GAM_Attention to exports
    "GhostBottleneck",
    "GhostConv",
    "HGBlock",
    "HGStem",
    "HetConv",  # Added HetConv to exports
    "ImagePoolingAttn",
    "Index",
    "LRPCHead",
    "LayerNorm2d",
    "LightConv",
    "MLPBlock",
    "MSDeformAttn",
    "MaxSigmoidAttnBlock",
    "Pose",
    "Proto",
    "RTDETRDecoder",
    "RepC3",
    "RepConv",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResBlock_CBAM",  # Added ResBlock_CBAM to exports
    "ResNetLayer",
    "SCDown",
    "Segment",
    "SpatialAttention",
    "TorchVision",
    "TransformerBlock",
    "TransformerEncoderLayer",
    "TransformerLayer",
    "WorldDetect",
    "YOLOEDetect",
    "YOLOESegment",
    "v10Detect",
)
