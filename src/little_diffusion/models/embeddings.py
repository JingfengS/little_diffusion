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

def precompute_freqs_cis(dim, end, theta=10000.0):
    """
    预计算旋转角度 (cis = cos + i*sin)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # (L, D/2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis
        
        