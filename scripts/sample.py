import argparse
import torch
import logging
import time
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))
from little_diffusion.models.config import DiTConfig
from little_diffusion.models.dit import DiT
from little_diffusion.processor import VAEProcessor

# 引入你的 Core 框架！
from little_diffusion.core import ODE, EulerSimulator

# ================= 🔧 配置 =================
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ================= 🧠 Adapter: DiT 专用的 ODE =================
class DiTODE(ODE):
    """适配器：将 DiT 包装成符合 core.py 标准的 ODE 对象"""
    def __init__(self, model, label_id, null_label_id, cfg_scale):
        self.model = model
        self.label_tensor = torch.tensor([label_id], device=next(model.parameters()).device)
        self.null_tensor = torch.tensor([null_label_id], device=self.label_tensor.device)
        self.cfg_scale = cfg_scale

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t 传进来的是形状为 (1, 1) 或 (1) 的浮点数 [0, 1]
        t_float = t.view(-1)
        # 映射给 DiT
        t_int = (t_float * 1000).long()

        # 执行带有 CFG 的前向传播
        eps_cond = self.model(xt, t_int, self.label_tensor)
        if self.cfg_scale > 1.0:
            eps_uncond = self.model(xt, t_int, self.null_tensor)
            v_pred = eps_uncond + self.cfg_scale * (eps_cond - eps_uncond)
        else:
            v_pred = eps_cond
            
        return v_pred

# ================= 🏃 主函数 =================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--label", type=int, default=0)
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda")
    
    # 1. 加载模型 (代码略，与之前一样加载 .pth 和 config)
    checkpoint = torch.load(args.ckpt, map_location=device)
    config = DiTConfig(**checkpoint['config'])
    model = DiT(config).to(device)
    
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    
    vae = VAEProcessor(device=device, use_fp16=True)

    # 2. 准备 ODE 和 采样器
    logger.info("🚀 Starting Generation using Core Framework...")
    null_class = config.num_classes - 1
    
    # 🌟 实例化你的 OOP 框架组件！
    ode = DiTODE(model, args.label, null_class, args.cfg_scale)
    simulator = EulerSimulator(ode)

    # 3. 准备时间轴和初始状态
    # Flow Matching 从 t=0 (纯噪声) 走到 t=1 (原图)
    ts = torch.linspace(0.0, 1.0, args.steps + 1, device=device).view(1, -1, 1)
    x0 = torch.randn(1, config.in_channels, config.input_size, config.input_size, device=device)

    # 4. 执行模拟
    start_time = time.time()
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # 直接调用 simulate，它会返回最后一步的结果
            x_final = simulator.simulate(x0, ts)

    logger.info(f"⚡ Generation took {time.time() - start_time:.2f}s")
    logger.info(f"📊 Latent Stats: Mean={x_final.mean():.2f}, Std={x_final.std():.2f}")

    # 5. 解码并保存
    images = vae.decode(x_final)
    images[0].save(f"images/core_sample_label{args.label}.png")
    logger.info("🎆 Success!")

if __name__ == "__main__":
    main()