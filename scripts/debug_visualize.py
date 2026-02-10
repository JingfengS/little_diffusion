import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from little_diffusion.models import BabyUNet

# ================= 配置区 =================
# 确保这里和你刚才跑诊断脚本时用的路径一模一样
IMG_PATH = "images/hutao.jpg"  
CKPT_PATH = "checkpoints/baby_unet.pth" 
# ========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image_tensor(path, size=704):
    """加载图片并归一化到 [-1, 1]"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    try:
        img = Image.open(path).convert('RGB')
        return transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"❌ 图片加载失败: {e}")
        return None

def tensor_to_numpy_img(tensor):
    """把 [-1, 1] 的 Tensor 转回 [0, 1] 的 Numpy 用于画图"""
    # tensor shape: (1, 3, H, W)
    img = (tensor[0] + 1) / 2
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).cpu().detach().numpy()

def main():
    print(f"🎨 开始可视化诊断...")
    
    # 1. 准备模型
    model = BabyUNet(in_channels=3, out_channels=3, dim=64).to(device)
    try:
        state_dict = torch.load(CKPT_PATH, map_location=device)
        # 去前缀逻辑
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_state_dict[k[10:]] = v 
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=True)
        print("✅ 模型权重加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    model.eval()

    # 2. 准备数据
    x1 = load_image_tensor(IMG_PATH) # 目标图 (Target)
    if x1 is None: return
    
    x0 = torch.randn_like(x1).to(device) # 噪声图 (Source)
    
    # 3. 这里的 t 必须是一样的时间点
    # 我们测试 t=0 的情况，模型应该预测从噪声直接走到原图的速度
    t = torch.zeros(1, 1).to(device) 

    print("🖼️ 正在计算...")

    # 4. 让模型预测
    with torch.no_grad():
        # 在 Flow Matching 中，v = x1 - x0
        # 所以理论上 x1 = x0 + v
        pred_v = model(x0, t)
        
        # 【关键】手动一步还原！
        # 如果模型是对的，x0 加上预测的速度，就应该等于 x1 (原图)
        x_reconstructed = x0 + pred_v 

    # 5. 画图对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 图1: 纯噪声
    axes[0].imshow(tensor_to_numpy_img(x0))
    axes[0].set_title("Input: Noise (x0)")
    axes[0].axis('off')

    # 图2: 你的原图 (检查这一步！)
    axes[1].imshow(tensor_to_numpy_img(x1))
    axes[1].set_title("Ground Truth: Reference Image (x1)")
    axes[1].axis('off')

    # 图3: 模型还原结果
    axes[2].imshow(tensor_to_numpy_img(x_reconstructed))
    axes[2].set_title("Model Prediction (x0 + pred_v)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
    
    # 保存结果以防 Notebook 不显示
    plt.savefig("debug_result.png")
    print("✅ 结果已保存为 debug_result.png")

if __name__ == "__main__":
    main()