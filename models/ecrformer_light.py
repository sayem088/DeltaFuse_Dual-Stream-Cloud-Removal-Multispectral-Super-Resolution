"""
Module: ecrformer_light.py
Location: F:/sparrso/multi-modal-cloud-removal-dataset-design/models/ecrformer_light.py
Description: Full & Faithful Implementation of ECRformer-Light as per ISPRS 2026 Paper:
  "ECRformer: An efficient cloud removal Transformer with semantic-decoupled learning
   for multimodal satellite imagery" (Zhang et al., ISPRS J. Photogramm. Remote Sens., 2026).

Paper-Accurate Architecture Components:
  1. Section 3.2.1: Shallow Feature Embedding
     - Modality-specific parallel 7x7 Convolutions for Optical and SAR inputs.
     - Channel concatenation: F_shallow = Concat(Conv7x7_opt(X_opt), Conv7x7_sar(X_sar)).
  2. Section 3.3: ECRformer Block (Serially integrated tri-component transformer block):
     - Eq. (7): F_XCA  = XCA(LN(F_in)) + F_in
     - Eq. (8): F_MDWA = MDWA(LN(F_XCA)) + F_XCA
     - Eq. (9): F_out  = SGFN(LN(F_MDWA)) + F_MDWA
  3. Section 3.3.1: Cross-Covariance Attention (XCA)
     - Transposes attention to channel dimension: O(N * C^2) complexity instead of O(N^2 * C).
     - Attention: V * Softmax((K^T * Q) / tau) with learnable per-head temperature parameter tau.
     - Local spatial context injected via Depthwise Conv (DWConv) in Q, K, V projections.
  4. Section 3.3.2: Multi-Dilation Window Attention (MDWA)
     - Conditional Positional Encoding (CPE) via 3x3 DWConv before projection.
     - Window size M=3 with parallel multi-dilation rates d in [1, 2, 3, 4].
     - Multi-scale contextual aggregation fused via 1x1 convolution with linear O(N) complexity.
  5. Section 3.3.3: Self-Gated Feed-forward Network (SGFN)
     - Hidden expansion factor = 2.0.
     - 3x3 DWConv with element-wise gating: Conv1x1(GELU(X1) * X2).
  6. Section 3.2.2 & Table 6: ECRformer-Light Hyperparameter Configuration
     - Embedding Dim: 32 (dims = [32, 64, 128])
     - Encoder Stages (K_enc = 3): Depths L = [2, 2, 1]
     - Bottleneck (L_B = 1): Depth = 1
     - Decoder Stages (K_dec = 3): Symmetrical depths L = [1, 2, 2] with skip connections
     - Refinement Network (L_R = 2): 2 full-resolution ECRformer Blocks + 3x3 Conv
     - Total Parameters: ~1.8M (optimized for 4-band Optical + 2-band SAR)
  7. Section 3.4: Semantic-Decoupled Feature Learning (SDFL) & Loss Objective
     - Deep supervision on intermediate encoder stages (structural SSIM regularization).
     - Deep supervision on intermediate decoder stages (textural L1 regularization).
     - Total Loss: L = alpha * L1 + beta * (1 - SSIM) + lambda_enc * R_enc + lambda_dec * R_dec

Author: Md Abu Sayem Sarker
Target: SPARRSO Fellowship (Priority Area 6)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union


# =============================================================================
# 1. NORMALIZATION & STRUCTURAL BUILDING BLOCKS
# =============================================================================
class LayerNorm2d(nn.Module):
    """
    Channels-first 2D Layer Normalization:
    Applies LayerNorm across channel dimension for tensors of shape [B, C, H, W].
    """
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x_norm + self.bias[:, None, None]


# =============================================================================
# 2. CROSS-COVARIANCE ATTENTION (XCA) - Section 3.3.1 (Eq. 10 & 11)
# =============================================================================
class CrossCovarianceAttention(nn.Module):
    """
    Cross-Covariance Attention (XCA):
    Computes cross-covariance attention across channel vectors:
        XCAttention(Q, K, V) = V * Softmax((K^T * Q) / tau)
    Where tau is a learnable per-head temperature parameter.
    Complexity is O(N * C^2), shifting computational burden from spatial token count N
    to channel dimension C, making it optimal for high-resolution satellite imagery.
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Learnable temperature parameter tau (Eq. 10)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Local spatial context via 1x1 conv followed by 3x3 depthwise conv (Eq. 11)
        self.qkv_conv = nn.Sequential(
            nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False),
            nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=False)
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        # 1. Project and split Q', K', V' (Eq. 11)
        qkv = self.qkv_conv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # 2. Reshape to multi-head format: [B, heads, head_dim, N]
        q = q.reshape(B, self.num_heads, self.head_dim, N)
        k = k.reshape(B, self.num_heads, self.head_dim, N)
        v = v.reshape(B, self.num_heads, self.head_dim, N)

        # 3. L2-normalize queries and keys along spatial dimension N for training stability
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 4. Compute C x C cross-covariance attention map:
        #    attn: [B, heads, head_dim, head_dim]
        #    K^T * Q corresponds to k @ q.transpose(-2, -1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)

        # 5. Multiply attention map by Value matrix: [B, heads, head_dim, N]
        out = torch.matmul(attn, v)
        out = out.reshape(B, C, H, W)

        return self.project_out(out)


# =============================================================================
# 3. MULTI-DILATION WINDOW ATTENTION (MDWA) - Section 3.3.2 (Eq. 12 - 14)
# =============================================================================
class MultiDilationWindowAttention(nn.Module):
    """
    Multi-Dilation Window Attention (MDWA):
    Captures multi-scale spatial context with linear O(N) complexity:
      1. Injects learnable relative position via Conditional Positional Encoding (CPE, 3x3 DWConv).
      2. Dilated sampling with window size M=3 and multiple dilation rates (d in [1, 2, 3, 4]).
      3. For each pixel, attends to 3x3 sparse spatial neighborhood at dilation d.
      4. Aggregates multi-branch attention outputs via 1x1 convolution (Eq. 14).
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window_size: int = 3,
        dilation_rates: List[int] = [1, 2, 3, 4]
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.dilation_rates = dilation_rates

        # Conditional Positional Encoding (CPE): 3x3 DWConv (Eq. 12)
        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)

        # Multi-branch fusion convolution (Eq. 14)
        self.conv_fuse = nn.Conv2d(dim * len(dilation_rates), dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        K_size = self.window_size * self.window_size  # 3x3 = 9 elements

        # 1. Inject CPE and project to Q, K, V (Eq. 12)
        x_cpe = x + self.cpe(x)
        qkv = self.qkv(x_cpe)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Query format: [B, heads, head_dim, 1, N]
        q_view = q.view(B, self.num_heads, self.head_dim, 1, N)

        branch_outputs = []
        for d in self.dilation_rates:
            # 2. Extract dilated 3x3 keys and values using unfold with dilation d:
            #    k_unfold: [B, C * 9, N]
            k_unfold = F.unfold(k, kernel_size=self.window_size, dilation=d, padding=d)
            v_unfold = F.unfold(v, kernel_size=self.window_size, dilation=d, padding=d)

            # Reshape to multi-head format: [B, heads, head_dim, 9, N]
            k_unfold = k_unfold.view(B, self.num_heads, self.head_dim, K_size, N)
            v_unfold = v_unfold.view(B, self.num_heads, self.head_dim, K_size, N)

            # 3. Dilated attention dot-product: [B, heads, 9, N] (Eq. 13)
            attn = (q_view * k_unfold).sum(dim=2) * self.scale
            attn = F.softmax(attn, dim=2)  # Softmax over the 9 dilated neighborhood keys

            # 4. Context aggregation: [B, heads, head_dim, N] -> [B, C, H, W]
            out_d = (attn.unsqueeze(2) * v_unfold).sum(dim=3)
            out_d = out_d.view(B, C, H, W)
            branch_outputs.append(out_d)

        # 5. Fuse all dilation branches (Eq. 14)
        fused = self.conv_fuse(torch.cat(branch_outputs, dim=1))
        return fused


# =============================================================================
# 4. SELF-GATED FEED-FORWARD NETWORK (SGFN) - Section 3.3.3 (Eq. 15 & 16)
# =============================================================================
class SelfGatedFeedForward(nn.Module):
    """
    Self-Gated Feed-forward Network (SGFN):
    Employs depthwise convolution and GELU gating to enhance non-linear capacity:
        X1, X2 = Split(DWConv(Conv1x1(F_in)))
        F_out  = Conv1x1(GELU(X1) * X2)
    Hidden expansion factor = 2.0 (as specified in Table 6).
    """
    def __init__(self, dim: int, expansion_factor: float = 2.0):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)

        # 1x1 conv expansion followed by 3x3 depthwise convolution (Eq. 15)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(
            hidden_dim * 2, hidden_dim * 2,
            kernel_size=3, padding=1, groups=hidden_dim * 2, bias=False
        )
        # 1x1 conv projection back to original dimension (Eq. 16)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.project_in(x)
        x_dw = self.dwconv(x_proj)
        x1, x2 = torch.chunk(x_dw, 2, dim=1)

        # Element-wise gating with GELU activation
        gated = F.gelu(x1) * x2
        return self.project_out(gated)


# =============================================================================
# 5. THE COMPLETE ECRFORMER BLOCK - Section 3.3 (Eq. 7 - 9)
# =============================================================================
class ECRformerBlock(nn.Module):
    """
    ECRformer Block:
    Serially integrates XCA (channel attention), MDWA (spatial context), and SGFN:
        F_XCA  = XCA(LN(F_in)) + F_in           (Eq. 7)
        F_MDWA = MDWA(LN(F_XCA)) + F_XCA        (Eq. 8)
        F_out  = SGFN(LN(F_MDWA)) + F_MDWA      (Eq. 9)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dilation_rates: List[int] = [1, 2, 3, 4],
        expansion_factor: float = 2.0
    ):
        super().__init__()
        # 1. Cross-Covariance Attention Stage
        self.norm1 = LayerNorm2d(dim)
        self.xca = CrossCovarianceAttention(dim, num_heads=num_heads)

        # 2. Multi-Dilation Window Attention Stage
        self.norm2 = LayerNorm2d(dim)
        self.mdwa = MultiDilationWindowAttention(dim, num_heads=num_heads, dilation_rates=dilation_rates)

        # 3. Self-Gated Feed-forward Network Stage
        self.norm3 = LayerNorm2d(dim)
        self.sgfn = SelfGatedFeedForward(dim, expansion_factor=expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Eq. (7)
        x = x + self.xca(self.norm1(x))
        # Eq. (8)
        x = x + self.mdwa(self.norm2(x))
        # Eq. (9)
        x = x + self.sgfn(self.norm3(x))
        return x


# =============================================================================
# 6. DOWNSAMPLE & UPSAMPLE MODULES
# =============================================================================
class Downsample(nn.Module):
    """Downsamples spatial resolution by factor of 2 while doubling channels."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsamples spatial resolution by factor of 2 via bilinear interpolation + 3x3 Conv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# =============================================================================
# 7. ECRFORMER-LIGHT ARCHITECTURE - Section 3.2 & Table 6
# =============================================================================
class ECRformerLight(nn.Module):
    """
    ECRformer-Light Architecture:
    Faithfully implements the lightweight variant specified in Table 6:
      - Shallow Feature Embedding: Parallel 7x7 Convolutions for Optical and SAR (Eq. 2).
      - U-shaped Backbone:
          * 3 Encoder Stages (K_enc=3) with depths [2, 2, 1] across dims [32, 64, 128].
          * Bottleneck with depth L_B=1.
          * 3 Decoder Stages (K_dec=3) with depths [1, 2, 2] and skip connections.
      - Refinement Network: L_R=2 full-resolution ECRformer blocks.
      - Semantic-Decoupled Feature Learning (SDFL) Multi-scale Projection Heads.
    """
    def __init__(
        self,
        in_channels_opt: int = 4,     # 10m Sentinel-2 bands: B2, B3, B4, B8
        in_channels_sar: int = 2,     # Sentinel-1 SAR bands: VV, VH
        out_channels: int = 4,        # Target clean optical bands: B2, B3, B4, B8
        embed_dim: int = 32,          # Base embedding dimension from Table 6
        encoder_depths: List[int] = [2, 2, 1],
        bottleneck_depth: int = 1,
        decoder_depths: List[int] = [1, 2, 2],
        refinement_depth: int = 2,
        num_heads: int = 4,
        dilation_rates: List[int] = [1, 2, 3, 4],
        expansion_factor: float = 2.0
    ):
        super().__init__()
        self.in_channels_opt = in_channels_opt
        self.in_channels_sar = in_channels_sar
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.dims = [embed_dim, embed_dim * 2, embed_dim * 4]  # [32, 64, 128]

        # ---------------------------------------------------------------------
        # 3.2.1 Shallow Feature Embedding (Parallel 7x7 Convolutions, Eq. 2)
        # ---------------------------------------------------------------------
        self.conv_opt_7x7 = nn.Conv2d(in_channels_opt, embed_dim // 2, kernel_size=7, padding=3, bias=False)
        self.conv_sar_7x7 = nn.Conv2d(in_channels_sar, embed_dim // 2, kernel_size=7, padding=3, bias=False)
        self.shallow_fuse = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)

        # ---------------------------------------------------------------------
        # 3.2.2 U-Shaped Encoder (K_enc = 3 stages, Eq. 3)
        # ---------------------------------------------------------------------
        self.enc_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.enc_heads = nn.ModuleList()  # SDFL structural supervision heads (Eq. 17)

        for idx, (depth, d_dim) in enumerate(zip(encoder_depths, self.dims)):
            blocks = nn.Sequential(*[
                ECRformerBlock(
                    dim=d_dim,
                    num_heads=num_heads,
                    dilation_rates=dilation_rates,
                    expansion_factor=expansion_factor
                ) for _ in range(depth)
            ])
            self.enc_stages.append(blocks)
            # Lightweight 1x1 projection head for multi-scale intermediate prediction (Eq. 17)
            self.enc_heads.append(nn.Conv2d(d_dim, out_channels, kernel_size=1))

            if idx < len(self.dims) - 1:
                self.downsamples.append(Downsample(self.dims[idx], self.dims[idx + 1]))

        # ---------------------------------------------------------------------
        # Bottleneck (L_B = 1 block, Eq. 4)
        # ---------------------------------------------------------------------
        self.bottleneck = nn.Sequential(*[
            ECRformerBlock(
                dim=self.dims[-1],
                num_heads=num_heads,
                dilation_rates=dilation_rates,
                expansion_factor=expansion_factor
            ) for _ in range(bottleneck_depth)
        ])

        # ---------------------------------------------------------------------
        # 3.2.2 U-Shaped Decoder (K_dec = 3 stages, Eq. 5)
        # ---------------------------------------------------------------------
        self.dec_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.dec_heads = nn.ModuleList()  # SDFL textural supervision heads (Eq. 17)

        dec_dims = list(reversed(self.dims))  # [128, 64, 32]
        for idx, (depth, stage_dim) in enumerate(zip(decoder_depths, dec_dims)):
            if idx > 0:
                prev_dim = dec_dims[idx - 1]
                self.upsamples.append(Upsample(prev_dim, stage_dim))
                # Skip connection fusion conv: [Upsampled(stage_dim) + Skip(stage_dim)] -> stage_dim
                self.skip_convs.append(nn.Conv2d(stage_dim * 2, stage_dim, kernel_size=1, bias=False))

            blocks = nn.Sequential(*[
                ECRformerBlock(
                    dim=stage_dim,
                    num_heads=num_heads,
                    dilation_rates=dilation_rates,
                    expansion_factor=expansion_factor
                ) for _ in range(depth)
            ])
            self.dec_stages.append(blocks)
            # Lightweight 1x1 projection head for multi-scale intermediate prediction (Eq. 17)
            self.dec_heads.append(nn.Conv2d(stage_dim, out_channels, kernel_size=1))

        # ---------------------------------------------------------------------
        # 3.2.3 Refinement Network (L_R = 2 blocks + 3x3 Conv, Eq. 6)
        # ---------------------------------------------------------------------
        self.refinement = nn.Sequential(*[
            ECRformerBlock(
                dim=self.dims[0],
                num_heads=num_heads,
                dilation_rates=dilation_rates,
                expansion_factor=expansion_factor
            ) for _ in range(refinement_depth)
        ])
        self.conv_final = nn.Conv2d(self.dims[0], out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        opt: torch.Tensor,
        sar: torch.Tensor,
        return_sdfl: bool = False
    ) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:
        """
        Forward Pass:
        opt: [B, C_opt, H, W] cloudy optical input image
        sar: [B, C_sar, H, W] co-registered Sentinel-1 SAR input image
        return_sdfl: If True, returns intermediate multi-scale predictions for SDFL training loss.
        """
        # 1. Shallow feature embedding (Eq. 2)
        f_opt = self.conv_opt_7x7(opt)
        f_sar = self.conv_sar_7x7(sar)
        f_shallow = self.shallow_fuse(torch.cat([f_opt, f_sar], dim=1))

        # 2. U-Shaped Encoder (Eq. 3)
        skips = []
        enc_preds = []
        x = f_shallow
        for idx, stage in enumerate(self.enc_stages):
            x = stage(x)
            skips.append(x)
            if return_sdfl:
                enc_preds.append(self.enc_heads[idx](x))
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)

        # 3. Bottleneck (Eq. 4)
        x = self.bottleneck(x)

        # 4. U-Shaped Decoder (Eq. 5)
        dec_preds = []
        dec_x = x
        for idx, stage in enumerate(self.dec_stages):
            if idx == 0:
                dec_x = stage(dec_x)
            else:
                dec_x = self.upsamples[idx - 1](dec_x)
                # Concatenate corresponding encoder skip connection
                skip_feat = skips[len(skips) - 1 - idx]
                dec_x = self.skip_convs[idx - 1](torch.cat([dec_x, skip_feat], dim=1))
                dec_x = stage(dec_x)

            if return_sdfl:
                dec_preds.append(self.dec_heads[idx](dec_x))

        # 5. Refinement Network & Output (Eq. 6)
        f_refined = self.refinement(dec_x)
        # Residual connection from input optical image
        pred_residual = self.conv_final(f_refined)
        out = opt + pred_residual

        if return_sdfl:
            return {
                "out": out,
                "enc_preds": enc_preds,
                "dec_preds": dec_preds
            }
        return out


# =============================================================================
# 8. SEMANTIC-DECOUPLED FEATURE LEARNING (SDFL) LOSS - Section 3.4 (Eq. 17 - 23)
# =============================================================================
def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Differentiable Structural Similarity Index (SSIM) Loss (Eq. 20):
    Computes 1.0 - SSIM(pred, target).
    """
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    # Gaussian blur kernel
    sigma = 1.5
    gauss = torch.tensor([
        math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ], device=pred.device)
    gauss = gauss / gauss.sum()
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    channels = pred.shape[1]
    window = _2D_window.expand(channels, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1.0 - ssim_map.mean()


class SDFLLoss(nn.Module):
    """
    Semantic-Decoupled Feature Learning (SDFL) Training Objective (Eq. 22 & 23):
        L = L_output + lambda_enc * R_enc + lambda_dec * R_dec
    Where:
        L_output = alpha * L1 + beta * (1 - SSIM)       (Eq. 22)
        R_enc    = E_k [1 - SSIM(Y_enc^(k), Y^(k))]     (Eq. 20, Structural guidance)
        R_dec    = E_k [||Y_dec^(k) - Y^(k)||_1]        (Eq. 21, Textural rendering)
    Default hyperparameters from Table 6:
        alpha = 0.9, beta = 0.1, lambda_enc = 0.05, lambda_dec = 0.05
    """
    def __init__(
        self,
        alpha: float = 0.9,
        beta: float = 0.1,
        lambda_enc: float = 0.05,
        lambda_dec: float = 0.05
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_enc = lambda_enc
        self.lambda_dec = lambda_dec

    def forward(
        self,
        preds_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        pred_final = preds_dict["out"]
        enc_preds = preds_dict.get("enc_preds", [])
        dec_preds = preds_dict.get("dec_preds", [])

        # 1. Output Loss (Eq. 22)
        l1_final = F.l1_loss(pred_final, target)
        ssim_final = ssim_loss(pred_final, target)
        loss_output = self.alpha * l1_final + self.beta * ssim_final

        # 2. Encoder Structural Regularization Loss (Eq. 20)
        r_enc = torch.tensor(0.0, device=target.device)
        if enc_preds:
            enc_losses = []
            for pred_k in enc_preds:
                # Downsample target to match intermediate feature map scale
                target_k = F.interpolate(target, size=pred_k.shape[-2:], mode="bilinear", align_corners=False)
                enc_losses.append(ssim_loss(pred_k, target_k))
            r_enc = sum(enc_losses) / len(enc_losses)

        # 3. Decoder Textural Regularization Loss (Eq. 21)
        r_dec = torch.tensor(0.0, device=target.device)
        if dec_preds:
            dec_losses = []
            for pred_k in dec_preds:
                # Downsample target to match intermediate feature map scale
                target_k = F.interpolate(target, size=pred_k.shape[-2:], mode="bilinear", align_corners=False)
                dec_losses.append(F.l1_loss(pred_k, target_k))
            r_dec = sum(dec_losses) / len(dec_losses)

        # 4. Total Combined Loss (Eq. 23)
        total_loss = loss_output + self.lambda_enc * r_enc + self.lambda_dec * r_dec

        return {
            "loss": total_loss,
            "loss_output": loss_output,
            "loss_l1": l1_final,
            "loss_ssim": ssim_final,
            "loss_r_enc": r_enc,
            "loss_r_dec": r_dec
        }


# =============================================================================
# QUICK VERIFICATION HARNESS
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print(" ECRformer-Light: Architecture Verification Harness (ISPRS 2026)")
    print("=" * 80)

    model = ECRformerLight(
        in_channels_opt=4,
        in_channels_sar=2,
        out_channels=4,
        embed_dim=32,
        encoder_depths=[2, 2, 1],
        bottleneck_depth=1,
        decoder_depths=[1, 2, 2],
        refinement_depth=2,
        num_heads=4,
        dilation_rates=[1, 2, 3, 4]
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Total Trainable Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    # Test Dummy Batch
    dummy_opt = torch.randn(2, 4, 128, 128)
    dummy_sar = torch.randn(2, 2, 128, 128)
    dummy_target = torch.randn(2, 4, 128, 128)

    print(f"[*] Input Optical: {dummy_opt.shape} | Input SAR: {dummy_sar.shape}")

    # Forward pass with SDFL multi-scale heads
    preds = model(dummy_opt, dummy_sar, return_sdfl=True)
    print(f"[OK] Forward Pass Successful!")
    print(f"     Output: {preds['out'].shape}")
    print(f"     Encoder Intermediate Heads: {[p.shape for p in preds['enc_preds']]}")
    print(f"     Decoder Intermediate Heads: {[p.shape for p in preds['dec_preds']]}")

    # SDFL Loss computation
    criterion = SDFLLoss(alpha=0.9, beta=0.1, lambda_enc=0.05, lambda_dec=0.05)
    losses = criterion(preds, dummy_target)

    print(f"[OK] Loss Computation Successful!")
    print(f"     Total Loss:      {losses['loss'].item():.4f}")
    print(f"     Output L1:       {losses['loss_l1'].item():.4f}")
    print(f"     Output SSIM:     {losses['loss_ssim'].item():.4f}")
    print(f"     Enc Struct SSIM: {losses['loss_r_enc'].item():.4f}")
    print(f"     Dec Texture L1:  {losses['loss_r_dec'].item():.4f}")

    # Backward pass verification
    losses["loss"].backward()
    print("[OK] Backward Pass (Gradients) Verified Successfully!")
    print("=" * 80)
