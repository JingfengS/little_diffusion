import torch
import torch.nn as nn
from typing import Optional
import math

class TimestepEmbedder(nn.Module):
    """
    标准的正弦时间步编码 (Sinusoidal Positional Embedding)
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :return: an (N, dim) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t: torch.Tensor):
        t_emb = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_emb)

class LabelEmbedder(nn.Module):
    """
    类别标签编码 (用于 Classifier-Free Guidance)
    """
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        训练时随机丢弃 Label 以支持 CFG
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        # 将被丢弃的标签设为 num_classes (即最后一个 embedding 作为 null)
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels
    
    def forward(self, labels: torch.Tensor, is_train: bool = True, force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param labels: (B,)
        :param is_train: 是否训练
        :param force_drop_ids: 强制丢弃的标签
        :return: (B, hidden_size)
        """
        use_dropout = self.dropout_prob > 0.0
        if (is_train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)
    
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    应用 RoPE 旋转
    x: (B, H, L, D)
    freqs_cis: (L, D/2) complex tensor
    """
    # 将 x 转换为复数形式 (B, H, L, D/2)
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_complex = torch.complex(x_real, x_imag)
    
    # 广播 freqs_cis (L, D/2) -> (1, 1, L, D/2)
    freqs_cis = freqs_cis.view(1, 1, *freqs_cis.shape)
    
    # 复数乘法即旋转
    x_out = x_complex * freqs_cis
    
    # 转回实数
    x_out = torch.stack([x_out.real, x_out.imag], dim=-1).flatten(3)
    return x_out.type_as(x)

def precompute_freqs_cis_2d(dim: int, h: int, w: int, theta: float = 10000.0) -> torch.Tensor:
    """
    专为图像生成的 2D RoPE (X 和 Y 轴各占一半维度)
    dim: 注意力头的维度 (比如 Hidden=768, Heads=12, 那么 dim=64)
    h: 序列在高度上的 Patch 数量
    w: 序列在宽度上的 Patch 数量
    """
    half_dim = dim // 2  # 一半给 Y，一半给 X
    
    # 预计算 Y 轴和 X 轴的频率
    freqs_y = 1.0 / (theta ** (torch.arange(0, half_dim, 2).float() / half_dim))
    freqs_x = 1.0 / (theta ** (torch.arange(0, half_dim, 2).float() / half_dim))

    t_y = torch.arange(h, device=freqs_y.device)
    t_x = torch.arange(w, device=freqs_x.device)

    # 计算外积 (H, dim/4) 和 (W, dim/4)
    freqs_y = torch.outer(t_y, freqs_y).float()
    freqs_x = torch.outer(t_x, freqs_x).float()

    # 广播到完整的二维网格大小 (H, W, dim/4)
    freqs_y = freqs_y.unsqueeze(1).expand(h, w, -1)
    freqs_x = freqs_x.unsqueeze(0).expand(h, w, -1)

    # 拼接起来，得到 (H, W, dim/2)，然后展平为 (L, dim/2)
    freqs = torch.cat((freqs_y, freqs_x), dim=-1).reshape(h * w, -1)
    
    # 转为复数张量
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis
        