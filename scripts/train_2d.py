import torch
from torch.utils.data import DataLoader 
from little_diffusion.data import SingleImageDataset
from little_diffusion.models import BabyUNet
from little_diffusion.solvers import LinearProbabilityPath, FlowMatchingTrainer
from little_diffusion.core import Gaussian
import os

torch.set_float32_matmul_precision('medium')

# ================= 配置区 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_PATH = "images/hutao.jpg"   # 确保路径对
IMG_SIZE = 64                 # ✅ 降维打击
BATCH_SIZE = 32               # ✅ 加大 Batch
NUM_EPOCHS = 1000             # ✅ 疯狂训练 (小图很快，大约 5-10 分钟)
LR = 1e-3
SAVE_PATH = "checkpoints/hutao_64.pth" # ✅ 新的文件名
# ========================================

def main():
    print(f"🚀 Using device: {DEVICE}")
    print(f"📐 Resolution: {IMG_SIZE}x{IMG_SIZE}, Dim: 128")


    img_path = "images/hutao.jpg"
    dataset = SingleImageDataset(img_path, size=IMG_SIZE, num_samples=1000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = BabyUNet(in_channels=3, out_channels=3, dim=128).to(DEVICE)
    model = torch.compile(model)
    path = LinearProbabilityPath()
    trainer = FlowMatchingTrainer(model, path)
    
    print(f"🔥 Start Training for {NUM_EPOCHS} epochs...")
    trainer.train(
        dataloader=dataloader,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        device=DEVICE
    )
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Training finished. Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()