import argparse
import torch
import logging
import time
import numpy as np
from pathlib import Path
from PIL import Image

# 复用我们写好的模块
from little_diffusion.models import BabyUNet
from little_diffusion.core import ODE, EulerSimulator
from little_diffusion.processor import VAEProcessor

# ================= 🔧 配置 =================
# 针对 5070 Ti 开启 TF32
torch.set_float32_matmul_precision('high')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ================= 🧠 神经 ODE 包装器 =================
class NeuralODE(ODE):
    def __init__(self, model: torch.nn.Module, cfg_scale: float = 1.0):
        super().__init__()
        self.model = model
        self.cfg_scale = cfg_scale # 虽然现在是单图过拟合，预留 CFG 接口

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 确保 t 的形状正确 (Batch, 1)
        if t.dim() == 0:
            t = t.view(1, 1).expand(xt.shape[0], 1)
        elif t.dim() == 1:
            t = t.view(-1, 1)
            
        # 预测速度场 v
        # 如果训练用了 label embedding 这里可以做 guidance，现在直接预测
        v_pred = self.model(xt, t)
        
        return v_pred

def get_args():
    parser = argparse.ArgumentParser(description="🎨 Industrial Flow Matching Sampler")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--save_dir", type=str, default="./images/save_images", help="Output directory")
    parser.add_argument("--size", type=int, default=704, help="Output image size (pixel)")
    parser.add_argument("--steps", type=int, default=50, help="ODE solver steps (20-100)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--dim", type=int, default=128, help="Model hidden dimension (Must match training!)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device)
    
    # 1. 设置随机种子 (为了复现那张最好的图)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"🌱 Seed set to: {args.seed}")
    
    # 2. 初始化 VAE 处理器 (解码用)
    # 自动使用 FP16 加速
    vae_processor = VAEProcessor(device=args.device, use_fp16=True)

    # 3. 加载 UNet 模型
    logger.info(f"🧠 Loading Model from {args.ckpt}...")
    model = BabyUNet(in_channels=4, out_channels=4, dim=args.dim).to(device)
    
    try:
        checkpoint = torch.load(args.ckpt, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # 🛠️ 鲁棒性修复：自动去除 _orig_mod 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        model.eval()
        # 再次开启编译加速推理 (可选)
        # model = torch.compile(model, mode="max-autotune") 
        logger.info("✅ Model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return

    # 4. 采样流程
    logger.info(f"🎨 Generating image ({args.size}x{args.size})... Steps: {args.steps}")
    
    # 计算 Latent 尺寸 (704 -> 88)
    latent_size = args.size // 8
    
    # 初始化噪声 x0 (Batch=1, Channels=4, H, W)
    x0 = torch.randn(1, 4, latent_size, latent_size).to(device)
    
    # 时间步长 (0 -> 1)
    ts = torch.linspace(0, 1, args.steps, device=device).view(1, -1, 1)
    
    ode = NeuralODE(model)
    simulator = EulerSimulator(ode)

    start_time = time.time()

    with torch.no_grad():
        # 🔥 开启 AMP 混合精度推理 (5070 Ti 核心加速)
        with torch.amp.autocast('cuda'):
            # 执行 ODE 积分
            # 这里的 simulate 会调用 step，从 x0 (噪声) 走到 x1 (数据)
            traj = simulator.simulate_with_trajectory(x0, ts)
            
            # 取最后一步的结果
            x_final = traj[:, -1] # (1, 4, 88, 88)

    gen_time = time.time() - start_time
    logger.info(f"⚡ Generation took {gen_time:.3f}s")

    # 5. 解码 Latent -> Pixel
    logger.info("🧩 Decoding Latent to Image...")
    # VAE Processor 内部会自动处理 scaling factor 和数据类型转换
    images = vae_processor.decode(x_final)

    # 6. 保存结果
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = save_path / f"sample_{timestamp}_s{args.steps}.png"
    
    images[0].save(filename)
    logger.info(f"💾 Saved to: {filename}")
    
    # 顺便打印一下 Latent 的统计信息，看看是不是“炸”了
    logger.info(f"📊 Latent Stats: Mean={x_final.mean():.4f}, Std={x_final.std():.4f}, Min={x_final.min():.4f}, Max={x_final.max():.4f}")
    if x_final.std() > 5.0 or x_final.abs().max() > 10.0:
         logger.warning("⚠️ Warning: Latent values seem very high! The image might be noisy.")

if __name__ == "__main__":
    main()