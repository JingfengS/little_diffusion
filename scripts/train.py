import argparse
import torch
import os
import time
import logging
import signal
import sys
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from little_diffusion.models import BabyUNet
from little_diffusion.solvers import LinearProbabilityPath, FlowMatchingTrainer

# ================= 🚀 5070 Ti 极速模式设置 =================
# 开启 TensorFloat-32 (TF32)，在 Ampere/Hopper 架构上获得 FP32 的精度 + 接近 FP16 的速度
torch.set_float32_matmul_precision('high')
# 屏蔽一些编译时的烦人警告
torch._dynamo.config.suppress_errors = True

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ================= 🛠️ 参数解析 =================
def get_args():
    parser = argparse.ArgumentParser(description="🚀 Industrial Flow Matching Trainer (Latent)")
    
    # 基础配置
    parser.add_argument("--name", type=str, default="run", help="Experiment name")
    parser.add_argument("--data", type=str, required=True, help="Path to .pt latents file")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    
    # 训练超参
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32, help="Try 64 or 128 for 5070 Ti")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dim", type=int, default=128, help="Model width (hidden dimension)")
    
    # 进阶功能
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pth) to resume from")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile (use if errors occur)")
    parser.add_argument("--save_every", type=int, default=500, help="Save checkpoint every X epochs")
    
    return parser.parse_args()

# ================= 🧠 核心训练逻辑 =================
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"🔧 Device: {device} | Experiment: {args.name}")

    # 1. 动态加载 Latent 数据
    if not os.path.exists(args.data):
        logger.error(f"❌ Data file not found: {args.data}")
        return

    logger.info("📦 Loading latents into VRAM...")
    # map_location=device 直接加载进显存，因为 Latent 数据通常很小 (<2GB)
    # 如果数据特别大，请改用 map_location='cpu'
    latents = torch.load(args.data, map_location=device)
    
    # 自动识别尺寸 (N, 4, H, W)
    N, C, H, W = latents.shape
    logger.info(f"📊 Dataset Shape: {latents.shape}")
    logger.info(f"   - Images: {N}")
    logger.info(f"   - Latent Size: {H}x{W} (Equivalent to Pixel {H*8}x{W*8})")

    # 构造 Dataset
    # 如果只有少量图片，repeat 一下让每个 Epoch 多跑几步，避免 tqdm 刷屏太快
    if N < 1000:
        repeat_factor = 1000 // N
        logger.info(f"🔄 Small dataset detected. Repeating {repeat_factor} times per epoch.")
        dataset = TensorDataset(latents.repeat(repeat_factor, 1, 1, 1))
    else:
        dataset = TensorDataset(latents)
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 2. 初始化模型
    # 注意：in/out channels 自动设为 C (通常是 4)
    model = BabyUNet(in_channels=C, out_channels=C, dim=args.dim).to(device)
    
    # 🚀 5070 Ti 加速神器: torch.compile
    # 第一次运行会花 1-2 分钟编译，之后速度提升 30%-50%
    if not args.no_compile:
        logger.info("⚡️ Compiling model with torch.compile (Mode: max-autotune)...")
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception as e:
            logger.warning(f"⚠️ Compile failed: {e}. Fallback to standard mode.")

    # 3. 优化器 & 混合精度 Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda') # 混合精度的大脑

    start_epoch = 0

    # 4. 断点续训逻辑 (Robustness)
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"♻️ Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # 处理 compile 带来的前缀问题
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("_orig_mod."):
                    new_state_dict[k[10:]] = v
                else:
                    new_state_dict[k] = v
            
            # 加载权重
            model.load_state_dict(new_state_dict, strict=False) # strict=False 允许一定的灵活性
            
            # 恢复优化器状态 (重要！否则 LR 会重置)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 恢复 Epoch
            start_epoch = checkpoint.get('epoch', 0) + 1
            logger.info(f"   -> Resuming at Epoch {start_epoch}")
        else:
            logger.warning(f"⚠️ Checkpoint not found: {args.resume}. Starting from scratch.")

    # 5. 准备训练组件
    path = LinearProbabilityPath()
    trainer = FlowMatchingTrainer(model, path)
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 优雅退出处理 (Ctrl+C)
    def signal_handler(sig, frame):
        logger.info("\n🛑 Interrupt received! Saving emergency checkpoint...")
        save_checkpoint(model, optimizer, start_epoch, save_dir / f"{args.name}_interrupted.pth")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # ================= 🔄 训练循环 =================
    logger.info("🔥 Starting Training...")
    model.train()
    
    t0 = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0
        steps = 0
        
        for batch in dataloader:
            x1 = batch[0].to(device) # Target Latents
            
            optimizer.zero_grad()
            
            # ⚡️ 混合精度上下文 (Auto Mixed Precision)
            # 这里的计算会自动转为 FP16，显存减半，速度翻倍
            with torch.amp.autocast('cuda'):
                loss = trainer.get_train_loss(target=x1)
            
            # ⚡️ Scaler 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            steps += 1
            
        avg_loss = epoch_loss / steps
        
        # 打印日志 (每 100 轮)
        if (epoch + 1) % 100 == 0:
            elapsed = time.time() - t0
            speed = (epoch + 1 - start_epoch) / elapsed
            logger.info(f"Epoch {epoch+1:04d} | Loss: {avg_loss:.6f} | Speed: {speed:.1f} epoch/s")

        # 定期保存 (Robust Checkpointing)
        if (epoch + 1) % args.save_every == 0:
            save_path = save_dir / f"{args.name}_ep{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch, save_path)
            
            # 同时也更新一个 latest.pth 方便随时 resume
            save_checkpoint(model, optimizer, epoch, save_dir / f"{args.name}_latest.pth")

    logger.info("✅ Training Finished!")
    save_checkpoint(model, optimizer, args.epochs-1, save_dir / f"{args.name}_final.pth")

def save_checkpoint(model, optimizer, epoch, path):
    """保存完整的训练状态，不仅仅是权重"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': { # 保存一些元数据，防止以后忘了这个模型是啥参数
             'timestamp': time.time(),
        }
    }, path)
    logger.info(f"💾 Saved checkpoint to {path}")

if __name__ == "__main__":
    main()