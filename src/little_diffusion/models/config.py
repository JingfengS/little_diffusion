from pydantic import BaseModel, Field, model_validator
import json
import os

class DiTConfig(BaseModel):
    """
    DiT 模型配置类 (Single Source of Truth)
    负责所有超参数的校验
    """
    input_size: int = Field(default=128, description="Latent size (e.g. 1024px image -> 128 latent)")
    patch_size: int = Field(default=2, description="Patch size for tokenization")
    in_channels: int = Field(default=4, description="Input latent channels (SDXL=4)")
    hidden_size: int = Field(default=384, description="Transformer hidden dimension")
    depth: int = Field(default=12, description="Number of DiT blocks")
    num_heads: int = Field(default=6, description="Number of attention heads")
    mlp_ratio: float = Field(default=4.0, description="MLP expansion ratio")
    num_classes: int = Field(default=1000, description="Number of classes for conditioning")
    learn_sigma: bool = Field(default=False, description="Whether to learn variance")
    dropout_prob: float = Field(default=0.1, ge=0.0, le=1.0)

    @model_validator(mode='after')
    def check_dimensions(self):
        # 1. 检查 Hidden Size 和 Heads 的匹配性
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
            )
        
        # 2. 检查 Input Size 和 Patch Size 的匹配性
        if self.input_size % self.patch_size != 0:
            raise ValueError(
                f"input_size ({self.input_size}) must be divisible by patch_size ({self.patch_size})"
            )
            
        return self

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def from_pretrained(cls, save_directory: str):
        with open(os.path.join(save_directory, "config.json"), "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)