import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from little_diffusion.models import BabyUNet

# ================= 配置区 =================
# 1. 这里填你真实的图片路径 (必须和训练时一样)
IMG_PATH = "images/hutao.jpg"  
# 2. 这里填你刚才看过的那个 14MB 的模型路径
CKPT_PATH = "checkpoints/baby_unet.pth" 
# ========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(path, size=704):
    """手动加载并归一化图片，模拟 Dataset 的行为"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 归一化到 [-1, 1]
    ])
    try:
        img = Image.open(path).convert('RGB')
        return transform(img).unsqueeze(0).to(device) # (1, 3, 704, 704)
    except Exception as e:
        print(f"❌ 图片加载失败: {e}")
        return None

def main():
    print(f"🕵️‍♂️ 开始模型诊断...")
    print(f"使用设备: {device}")

    # 1. 初始化模型
    model = BabyUNet(in_channels=3, out_channels=3, dim=64).to(device)

    # 2. 加载权重 (带前缀修复逻辑)
    try:
        state_dict = torch.load(CKPT_PATH, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
        
        # strict=True 是关键！如果 key 不匹配它会立刻报错，而不是默默装死
        model.load_state_dict(new_state_dict, strict=True)
        print(f"✅ 权重加载成功: {CKPT_PATH}")
    except Exception as e:
        print(f"❌ 权重加载极其失败: {e}")
        return

    model.eval() # 开启评估模式

    # 3. 准备数据
    x1 = load_image(IMG_PATH) # 目标图片 (真实图片)
    if x1 is None: return
    
    # 构造随机噪声 x0
    x0 = torch.randn_like(x1).to(device)
    
    # 构造时间 t (我们测试 t=0.5 的中间时刻)
    t = torch.tensor([[0.5]]).to(device) # (1, 1)

    # 4. 手动计算 Flow Matching 目标
    # 线性插值: xt = 0.5 * x0 + 0.5 * x1
    xt = (1 - t) * x0 + t * x1
    
    # 真实速度目标: v = x1 - x0
    target_v = x1 - x0

    print("\n📊 --- 诊断报告 ---")
    
    # 5. 模型预测
    with torch.no_grad():
        pred_v = model(xt, t)
    
    # 6. 计算 Loss
    loss = F.mse_loss(pred_v, target_v)
    
    print(f"🔹 目标速度 (Target v) 均值: {target_v.mean().item():.4f}, 标准差: {target_v.std().item():.4f}")
    print(f"🔹 预测速度 (Pred v)   均值: {pred_v.mean().item():.4f}, 标准差: {pred_v.std().item():.4f}")
    print(f"📉 当前 Loss (MSE): {loss.item():.6f}")

    # 7. 判定结果
    if loss.item() < 0.05:
        print("\n✅ 结论: 模型是个天才！权重完全没问题。")
        print("👉 问题出在采样脚本 (sample_hutao.py) 的积分逻辑上，或者可视化代码上。")
    else:
        print("\n❌ 结论: 模型是个笨蛋。权重加载进去了，但它预测全是错的。")
        print("👉 这个 .pth 文件可能是一个没训练过的初始权重，或者训练时保存逻辑有问题。")

if __name__ == "__main__":
    main()