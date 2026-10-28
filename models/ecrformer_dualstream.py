"""
Module: ecrformer_dualstream.py
Location: F:/sparrso/multi-modal-cloud-removal-dataset-design/models/ecrformer_dualstream.py
Description: True Dual-Stream ECRformer-Light Architecture for 10m Cloud Removal.
Based on: ECRformer (Zhang et al., ISPRS Journal of Photogrammetry and Remote Sensing, 2024)
Key Architectural Features:
  1. True Dual-Stream Design:
     - Stream 1: Optical Encoder Branch (4 input channels: B2, B3, B4, B8).
     - Stream 2: SAR Encoder Branch (2 input channels: VV, VH in dB).
  2. Cross-Covariance Attention (XCA):
     - Calculates cross-covariance attention over feature channels rather than an N x N spatial matrix.
     - Reduces complexity scaling with spatial token count: O(N * C^2) vs standard O(N^2 * C).
  3. Self-Gated Feed-forward Network (SGFN) with Depthwise Convolutions.
  4. U-Net style Decoder with cross-modal skip connections reconstructing clean 10m 4-band optical imagery.
Author: Md Abu Sayem Sarker
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class CrossCovarianceAttention(nn.Module):
    """
    Channel-wise Cross-Covariance Attention (XCA)
    Attention is calculated across channel vectors:
        XCA(Q, K, V) = V * Softmax( (K^T * Q) / tau )
    Complexity: O(N * C^2), where N = H * W (spatial tokens) and C = channels.
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0, f"Dimension {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 1x1 point-wise projection followed by 3x3 depth-wise convolution
        self.qkv = nn.Sequential(
            nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False),
            nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=False)
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Reshape to [B, num_heads, head_dim, H*W]
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W)

        # L2-normalize queries and keys across spatial tokens to stabilize dot products
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Cross-covariance matrix: [head_dim x head_dim]
        # Transpose k to [B, num_heads, H*W, head_dim]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)

        # Channel-weighted output: [B, num_heads, head_dim, H*W]
        out = torch.matmul(attn, v)
        out = out.reshape(B, C, H, W)

        return self.project_out(out)


class SelfGatedFeedForward(nn.Module):
    """
    Self-Gated Feed-forward Network (SGFN)
    Incorporates depthwise convolution and element-wise gating:
        SGFN(X) = Conv1x1( (DWConv(W1 * X) * GELU) (x) (W2 * X) )
    """
    def __init__(self, dim: int, expansion_factor: int = 2):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(
            hidden_dim * 2, hidden_dim * 2,
            kernel_size=3, padding=1, groups=hidden_dim * 2, bias=False
        )
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.project_in(x)
        x_dw = self.dwconv(x_proj)
        x1, x2 = torch.chunk(x_dw, 2, dim=1)
        
        # Element-wise gating with GELU activation
        x_gated = F.gelu(x1) * x2
        return self.project_out(x_gated)


class ECRformerBlock(nn.Module):
    """Core ECRformer Transformer Block: XCA + SGFN with residual connections."""
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = CrossCovarianceAttention(dim, num_heads)
        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = SelfGatedFeedForward(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DualStreamECRformerLight(nn.Module):
    """
    True Dual-Stream ECRformer-Light for 10m Cloud Removal.
    
    Inputs:
      - opt: Cloudy optical image [B, 4, H, W] (B2, B3, B4, B8)
      - sar: Sentinel-1 radar backscatter [B, 2, H, W] (VV, VH in dB)
    Output:
      - reconstructed_opt: Clean optical image [B, 4, H, W]
    """
    def __init__(
        self,
        opt_channels: int = 4,
        sar_channels: int = 2,
        out_channels: int = 4,
        base_dim: int = 32
    ):
        super().__init__()

        # =====================================================================
        # 1. OPTICAL STREAM ENCODER (Branch 1)
        # =====================================================================
        self.opt_stem = nn.Sequential(
            nn.Conv2d(opt_channels, base_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.opt_enc1 = ECRformerBlock(base_dim, num_heads=2)
        self.opt_down1 = nn.Conv2d(base_dim, base_dim * 2, kernel_size=2, stride=2)
        self.opt_enc2 = ECRformerBlock(base_dim * 2, num_heads=4)

        # =====================================================================
        # 2. SAR STREAM ENCODER (Branch 2)
        # =====================================================================
        self.sar_stem = nn.Sequential(
            nn.Conv2d(sar_channels, base_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.sar_enc1 = ECRformerBlock(base_dim, num_heads=2)
        self.sar_down1 = nn.Conv2d(base_dim, base_dim * 2, kernel_size=2, stride=2)
        self.sar_enc2 = ECRformerBlock(base_dim * 2, num_heads=4)

        # =====================================================================
        # 3. CROSS-MODAL FUSION BOTTLENECK
        # =====================================================================
        # Fuses Optical (base_dim*2) and SAR (base_dim*2) -> total channels: base_dim*4
        self.fusion_down = nn.Conv2d(base_dim * 4, base_dim * 4, kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(
            ECRformerBlock(base_dim * 4, num_heads=8),
            ECRformerBlock(base_dim * 4, num_heads=8)
        )

        # =====================================================================
        # 4. U-NET DECODER WITH DUAL-STREAM SKIP CONNECTIONS
        # =====================================================================
        # Level 2 Decoder
        self.up2 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, kernel_size=2, stride=2)
        # Skip input: upsampled (base_dim*2) + opt_enc2 (base_dim*2) + sar_enc2 (base_dim*2) = base_dim*6
        self.reduce2 = nn.Conv2d(base_dim * 6, base_dim * 2, kernel_size=1)
        self.dec2 = ECRformerBlock(base_dim * 2, num_heads=4)

        # Level 1 Decoder
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, kernel_size=2, stride=2)
        # Skip input: upsampled (base_dim) + opt_enc1 (base_dim) + sar_enc1 (base_dim) = base_dim*3
        self.reduce1 = nn.Conv2d(base_dim * 3, base_dim, kernel_size=1)
        self.dec1 = ECRformerBlock(base_dim, num_heads=2)

        # =====================================================================
        # 5. TERMINAL 10m RECONSTRUCTION HEAD
        # =====================================================================
        self.head = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_dim, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Surface Reflectance strictly bounded in [0.0, 1.0]
        )

    def forward(self, opt: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            opt: [B, 4, H, W] (Cloudy Sentinel-2 B2, B3, B4, B8)
            sar: [B, 2, H, W] (Sentinel-1 SAR VV, VH in dB)
        Returns:
            reconstructed: [B, 4, H, W] (Clean 10m Optical reflectance)
        """
        # --- Optical Stream Forward ---
        o0 = self.opt_stem(opt)
        o1 = self.opt_enc1(o0)                # [B, base_dim, H, W]
        o2 = self.opt_enc2(self.opt_down1(o1)) # [B, base_dim*2, H/2, W/2]

        # --- SAR Stream Forward ---
        s0 = self.sar_stem(sar)
        s1 = self.sar_enc1(s0)                # [B, base_dim, H, W]
        s2 = self.sar_enc2(self.sar_down1(s1)) # [B, base_dim*2, H/2, W/2]

        # --- Bottleneck Cross-Modal Fusion ---
        fused_intermediate = torch.cat([o2, s2], dim=1) # [B, base_dim*4, H/2, W/2]
        b = self.bottleneck(self.fusion_down(fused_intermediate)) # [B, base_dim*4, H/4, W/4]

        # --- Decoder Stage 2 ---
        d2 = self.up2(b)                                # [B, base_dim*2, H/2, W/2]
        d2_skip = torch.cat([d2, o2, s2], dim=1)       # [B, base_dim*6, H/2, W/2]
        d2 = self.dec2(self.reduce2(d2_skip))          # [B, base_dim*2, H/2, W/2]

        # --- Decoder Stage 1 ---
        d1 = self.up1(d2)                               # [B, base_dim, H, W]
        d1_skip = torch.cat([d1, o1, s1], dim=1)       # [B, base_dim*3, H, W]
        d1 = self.dec1(self.reduce1(d1_skip))          # [B, base_dim, H, W]

        # --- Terminal Reconstruction Head ---
        reconstructed = self.head(d1)                   # [B, 4, H, W]
        return reconstructed


# -----------------------------------------------------------------------------
# Unit Testing & Complexity Profiling
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print(" Dual-Stream ECRformer-Light: Architecture Verification & Profiling")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running model validation on device: {device}")

    # Initialize model
    model = DualStreamECRformerLight(
        opt_channels=4,
        sar_channels=2,
        out_channels=4,
        base_dim=32
    ).to(device)

    # Synthetic batch of size 2, patch size 256x256
    B, H, W = 2, 256, 256
    dummy_opt = torch.rand(B, 4, H, W, device=device)   # 4 optical channels [0, 1]
    dummy_sar = torch.randn(B, 2, H, W, device=device)  # 2 SAR channels (dB norm)

    # Forward pass
    out = model(dummy_opt, dummy_sar)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[OK] Forward Pass Succeeded!")
    print(f"    * Input Optical Tensor Shape : {dummy_opt.shape}")
    print(f"    * Input SAR Tensor Shape     : {dummy_sar.shape}")
    print(f"    * Output Reconstructed Shape : {out.shape}")
    print(f"    * Total Parameters           : {total_params:,} ({total_params / 1e6:.2f} M)")
    print(f"    * Trainable Parameters       : {trainable_params:,}")
    print(f"    * Output Value Range         : min={out.min().item():.4f}, max={out.max().item():.4f}")
    
    # Assertions
    assert out.shape == (B, 4, H, W), "Output shape mismatch!"
    assert out.min() >= 0.0 and out.max() <= 1.0, "Reflectance values not strictly bounded in [0, 1]!"
    print("\n[OK] ALL ARCHITECTURAL ASSERTIONS PASSED WITH ZERO DETACHMENT.")
