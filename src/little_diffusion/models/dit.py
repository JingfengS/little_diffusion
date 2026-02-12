import torch
import torch.nn as nn

# 导入新的 Config 和 组件
from .config import DiTConfig
from .embeddings import TimestepEmbedder, LabelEmbedder, precompute_freqs_cis
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
        self.x_embedder = nn.Conv2d(
            config.in_channels, config.hidden_size, 
            kernel_size=config.patch_size, stride=config.patch_size
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
        
        # 6. Precompute RoPE
        num_patches = (self.input_size // self.patch_size) ** 2
        head_dim = self.hidden_size // self.num_heads
        self.register_buffer("freqs_cis", precompute_freqs_cis(head_dim, num_patches), persistent=False)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

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

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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