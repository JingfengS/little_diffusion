import torch
import matplotlib.pyplot as plt
import numpy as np
from little_diffusion.models import BabyUNet
from little_diffusion.core import ODE, EulerSimulator
from pathlib import Path

# ================= 配置区 =================
CKPT_PATH = "checkpoints/baby_unet.pth" # 记得改成那个 14MB 的文件
STEPS = 100 # 总步数
SHOW_FRAMES = 6 # 我们想看几个关键帧
save_path = Path("images/save_images")
save_path.mkdir(exist_ok=True)
# ========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 定义 ODE 包装
class NeuralODE(ODE):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(xt, t)

def main():
    print("🎬 准备生成演变过程...")
    
    # 2. 加载模型
    model = BabyUNet(in_channels=3, out_channels=3, dim=64).to(device)
    try:
        state_dict = torch.load(CKPT_PATH, map_location=device)
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    model.eval()

    # 3. 准备 Simulator
    ode = NeuralODE(model)
    simulator = EulerSimulator(ode)

    # 4. 准备初始噪声
    x0 = torch.randn(1, 3, 704, 704).to(device)
    
    # 5. 定义时间轴 (Batch, Steps, 1)
    # 比如: [0.00, 0.01, ..., 0.99]
    ts = torch.linspace(0, 1, STEPS).to(device).view(1, -1, 1)

    print(f"🚀 开始积分 (记录轨迹)...")
    
    # 6. 使用 simulate_with_trajectory 获取完整历史
    # 返回形状: (Batch, Steps, Channels, H, W)
    with torch.no_grad():
        traj = simulator.simulate_with_trajectory(x0, ts)
    
    print(f"✅ 轨迹生成完毕! Shape: {traj.shape}")

    # 7. 挑选关键帧并画图
    # 比如从 100 步里挑 6 张: [0, 20, 40, 60, 80, 99]
    indices = torch.linspace(0, STEPS - 1, SHOW_FRAMES).long()
    
    fig, axes = plt.subplots(1, SHOW_FRAMES, figsize=(4 * SHOW_FRAMES, 4))
    
    for i, idx in enumerate(indices):
        # 取出那一帧的数据 (1, 3, H, W) -> (3, H, W)
        frame_tensor = traj[0, idx] 
        img = frame_tensor.permute(1, 2, 0).cpu().numpy()
        
        # ✅ 修正 1: 使用标准还原，而不是 Min-Max 拉伸
        # 假设训练数据是 Normalize((-1, -1, -1), (1, 1, 1))
        img_show = (img + 1) / 2
        img_show = np.clip(img_show, 0, 1) # 截断超出范围的值
        
        # 获取当前时间点
        t_val = ts[0, idx, 0].item()
        
        axes[i].imshow(img_show)
        axes[i].set_title(f"t = {t_val:.2f}", fontsize=12)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path / "process_hutao.png")
    plt.show()
    
    print("💾 过程图已保存为 process_hutao.png")

if __name__ == "__main__":
    main()