import torch
import torch.nn as nn
import torch.nn.functional as F  # 👈 必须引用这个
from .embeddings import apply_rotary_emb

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        
        # 1. 生成 QKV (Fused Linear)
        # (B, L, 3, H, D) -> (3, B, H, L, D)
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. 🔥 RoPE 旋转 (Pre-Attention)
        # 这一步必须在 FlashAttention 之前做
        # q, k: (B, H, L, D)
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # 3. 🚀 FlashAttention (Blackwell 引擎启动！)
        # PyTorch 会自动检测硬件，并在后台调用 FlashAttention CUDA Kernel
        # is_causal=False 因为这是画图 (双向注意力)，不是写小说 (单向)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0, # DiT 通常不用 dropout
            is_causal=False 
        )

        # 4. Reshape 回去
        # (B, H, L, D) -> (B, L, H, D) -> (B, L, C)
        x = x.transpose(1, 2).flatten(2)
        
        # 5. Output Projection
        x = self.proj(x)
        return x

class Mlp(nn.Module):
    """标准的 MLP"""
    def __init__(self, in_features: int, hidden_features: int, out_features: int, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class ConvMlp(nn.Module):
    """
    🔥 混合架构核心：带卷积的 MLP
    替代纯 Linear, 引入 'Inductive Bias' (归纳偏置), 让模型天生懂得 '邻域' 概念。
    """
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        
        # 🔥 Depthwise Convolution (3x3)
        # 这就是 UNet 的灵魂！它强制模型在处理特征时，必须看一眼周围 3x3 的像素。
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features, 
            kernel_size=3, padding=1, groups=hidden_features
        )
        
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        # x: (B, L, C)
        B, L, C = x.shape
        
        # 1. Linear Projection
        x = self.fc1(x) # (B, L, Hidden)
        
        # 2. Reshape & Conv (提取局部特征)
        # 变换为图像格式 (B, Hidden, H, W) 以进行卷积
        x_img = x.transpose(1, 2).view(B, -1, H, W)
        x_img = self.dwconv(x_img)
        # 变回序列格式
        x = x_img.flatten(2).transpose(1, 2) # (B, L, Hidden)
        
        # 3. Activation & Output
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    DiT 核心模块：包含 Self-Attention 和 MLP
    使用 AdaLN-Zero (Adaptive Layer Norm) 进行条件注入
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = ConvMlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            out_features=hidden_size,
        )
        
        # AdaLN 调制参数预测层
        # 输入是 time_emb + label_emb，输出 6 个参数:
        # (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D) - Image Tokens
        c: (B, D) - Conditioning (Time + Label)
        freqs_cis: RoPE position info
        """
        # 预测调制参数
        B, L, D = x.shape
        H = W = int(L ** 0.5)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Attention Block with Modulation
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), freqs_cis)
        
        # MLP Block with Modulation
        mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input, H, W)
        
        return x