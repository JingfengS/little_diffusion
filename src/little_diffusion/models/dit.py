import torch
import torch.nn as nn

# 导入新的 Config 和 组件
from .config import DiTConfig
from .embeddings import TimestepEmbedder, LabelEmbedder, precompute_freqs_cis_2d
from .blocks import DiTBlock

class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x

class ConvPatchEmbedder(nn.Module):
    """
    使用小型的 CNN 替代线性的 Patch Embedding。
    作用：为 DiT 注入“归纳偏置”，让它在训练初期就能看懂边缘和纹理，
    从而在小样本 (3000张) 上也能快速收敛，避免画出混乱色块。
    """
    def __init__(self, in_channels, hidden_size, patch_size=2):
        super().__init__()
        # 这是一个 2 层的 ResNet-like 结构
        # 假设 patch_size=2，我们需要把分辨率降低 2 倍
        
        mid_channels = max(hidden_size // 4, 32)
        
        self.net = nn.Sequential(
            # 第一层：3x3 卷积，提取局部特征 (纹理/边缘)，不改变尺寸
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=1),
            nn.GroupNorm(4, mid_channels), # GN 比 BN 在小 Batch 下更稳
            nn.SiLU(), # SiLU (Swish) 是 DiT 的标配激活函数
            
            # 第二层：3x3 卷积，负责下采样 (Patchify)
            # stride=patch_size 完成了降维，同时保留了空间相关性
            nn.Conv2d(mid_channels, hidden_size, kernel_size=3, padding=1, stride=patch_size),
            nn.GroupNorm(32, hidden_size),
            nn.SiLU(),
            
            # 第三层：1x1 卷积，特征对齐
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)

class DiT(nn.Module):
    """
    Diffusion Transformer (Config-Driven)
    """
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config # 保存 config 以备后用
        
        # 从 Config 解包参数，方便内部使用
        self.input_size = config.input_size
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.learn_sigma else config.in_channels
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        
        # 1. Patch Embedding
        self.x_embedder = ConvPatchEmbedder(
            in_channels=config.in_channels,
            hidden_size=config.hidden_size,
            patch_size=config.patch_size
        )
        
        # 2. Condition Embedding
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.y_embedder = LabelEmbedder(config.num_classes, config.hidden_size, dropout_prob=config.dropout_prob)
        
        # 3. Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(config.hidden_size, config.num_heads, mlp_ratio=config.mlp_ratio) 
            for _ in range(config.depth)
        ])
        
        # 4. Final Layer
        self.final_layer = FinalLayer(config.hidden_size, config.patch_size, self.out_channels)
        
        # 5. Initialize Weights
        self.initialize_weights()

        grid_h = self.input_size // self.patch_size
        grid_w = self.input_size // self.patch_size
        
        # 6. Precompute RoPE
        head_dim = self.hidden_size // self.num_heads
        self.register_buffer("freqs_cis", precompute_freqs_cis_2d(head_dim, grid_h, grid_w), persistent=False)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        for m in self.x_embedder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Input Check included via Type Hints and Shape assertions if needed
        """
        # (Optional) Runtime Shape Check
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            # 只有在 debug 时才建议开启，为了速度通常假设 dataloader 是对的
            pass 

        x = self.x_embedder(x)
        x = x.flatten(2).transpose(1, 2)
        
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, self.training)
        c = t_emb + y_emb
        
        for block in self.blocks:
            x = block(x, c, self.freqs_cis)
            
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x