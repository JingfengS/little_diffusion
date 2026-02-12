import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional, Tuple

# 引入你的组件
from little_diffusion.core import ODE, EulerSimulator
from little_diffusion.processor import VAEProcessor

class LatentDiffusionPipe:
    """
    🌟 工业级推理流水线 (Inference Pipeline)
    负责把 Model, VAE, Scheduler 串起来，实现一键出图。
    """
    def __init__(
        self, 
        model: torch.nn.Module, 
        vae_processor: VAEProcessor, 
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.vae_processor = vae_processor
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def __call__(
        self, 
        steps: int = 50, 
        batch_size: int = 1, 
        seed: Optional[int] = None,
        image_size: int = 704
    ) -> Image.Image:
        """
        🎨 一键生成图片
        Args:
            steps: 采样步数 (越高画质越好，但更慢)
            batch_size: 一次生成几张
            seed: 随机种子 (复现用)
            image_size: 输出分辨率 (pixel)
        Returns:
            PIL Image 对象 (如果 batch_size > 1，返回列表)
        """
        # 1. 设定随机种子 (为了复现那张“梦中情图”)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # 2. 准备 Latent 噪声 (x0)
        # Latent 尺寸 = 图片尺寸 / 8
        latent_dim = image_size // 8
        # Shape: (B, 4, H, W)
        x_init = torch.randn(batch_size, 4, latent_dim, latent_dim).to(self.device)
        
        # 3. 准备时间步 (Time Steps)
        # 从 0 (纯噪声) -> 1 (数据)
        # Shape: (B, Steps, 1) 方便广播计算
        ts = torch.linspace(0, 1, steps, device=self.device).view(1, -1, 1).expand(batch_size, steps, 1)

        # 4. 定义 ODE 求解器
        # 定义 drift 函数: v = model(x, t)
        def drift_func(x, t):
            # 确保 t 的形状适配 model
            if t.dim() == 1: t = t.view(-1, 1)
            # ⚡️ 5070 Ti 混合精度加速
            with torch.amp.autocast('cuda'):
                return self.model(x, t)

        # 这里的 ODE 类我们可以简化为一个 lambda 或者包装器
        # 为了复用你现有的架构，我们动态构建一个简单的 ODE 对象
        class SimpleODE(ODE):
            def drift_coefficient(self, x, t):
                return drift_func(x, t)

        ode = SimpleODE()
        simulator = EulerSimulator(ode)

        # 5. 🚀 执行采样 (也就是你贴的那段代码!)
        # 我们在这里加个 tqdm 进度条，让用户知道还要等多久
        print(f"🎨 Generating {image_size}x{image_size} image with {steps} steps...")
        
        # 直接调用 simulate (只返回最终结果，不存轨迹)
        # 注意：为了让 tqdm 生效，我们需要稍微魔改一下 simulate 或者在这里手动写循环
        # 为了完全复用你的 simulate 代码，我们直接调用它：
        latents = simulator.simulate(x_init, ts)
        
        # 6. ✨ VAE 解码 (Latent -> Pixel)
        print("🧩 Decoding...")
        images = self.vae_processor.decode(latents)
        
        # 如果只有一张图，直接返回对象，而不是列表
        if batch_size == 1:
            return images[0]
        return images