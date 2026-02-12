import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
import argparse
import sys
import time
import signal
from pathlib import Path

# 引入我们的工业级模块
from little_diffusion.models.config import DiTConfig
from little_diffusion.models.dit import DiT

# ================= 🚀 5070 Ti 极速模式设置 =================
# 开启 TF32 (Ampere/Hopper/Blackwell 专属)
torch.set_float32_matmul_precision('high')
# 抑制编译噪音
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
    parser = argparse.ArgumentParser(description="🚀 Robust DiT Trainer with Resume & Triton")
    
    # 基础配置
    parser.add_argument("--name", type=str, default="dit_test_run", help="Experiment name")
    parser.add_argument("--data", type=str, required=True, help="Path to arknights_latents_1024.pt")
    parser.add_argument("--output_dir", type=str, default="checkpoints/arknights", help="Directory to save checkpoints")
    
    # 训练超参
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16, help="Adjust based on VRAM (16-32 for 5070Ti)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_every", type=int, default=100, help="Save checkpoint every X epochs")
    
    # 续训控制
    parser.add_argument("--resume", type=str, default="latest", help="Path to checkpoint or 'latest' to auto-resume")
    parser.add_argument("--force_restart", action="store_true", help="Ignore existing checkpoints and start over")
    
    # 调试选项
    parser.add_argument("--debug", action="store_true", help="Run with small model for testing")
    
    return parser.parse_args()

# ================= 💾 Checkpoint 管理器 =================
class CheckpointManager:
    def __init__(self, save_dir, experiment_name):
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.name = experiment_name
        self.latest_path = self.save_dir / "arknights_latest_checkpoint.pth"

    def save(self, model, optimizer, epoch, loss, config, is_best=False):
        """保存完整状态"""
        # 如果模型被 compile 过，它的 state_dict key 会带有 "_orig_mod." 前缀
        # 我们需要去除它，以便未来加载时不受 compile 状态影响
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        
        state = {
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': config.model_dump(), # 保存 Pydantic Config
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'timestamp': time.time()
        }
        
        # 1. 保存为 latest (覆盖)
        torch.save(state, self.latest_path)
        
        # 2. 保存为 epoch 历史 (归档)
        epoch_path = self.save_dir / f"epoch_{epoch:04d}.pth"
        torch.save(state, epoch_path)
        
        logger.info(f"💾 Saved Checkpoint: Epoch {epoch} | Loss: {loss:.4f}")

    def load(self, path, model, optimizer=None):
        """加载完整状态"""
        if path == 'latest':
            path = self.latest_path
        
        path = Path(path)
        if not path.exists():
            logger.warning(f"⚠️ Checkpoint not found: {path}")
            return 0 # Start from epoch 0

        logger.info(f"♻️ Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location='cpu') # 先加载到 CPU 省显存
        
        # 加载模型权重
        msg = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info(f"   -> Model Weights Loaded: {msg}")
        
        # 加载优化器
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("   -> Optimizer State Restored")
            
        # 恢复随机种子 (确保复现性)
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])

        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"✅ Successfully Resumed from Epoch {start_epoch}")
        return start_epoch

# ================= 🧠 主程序 =================
def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 准备数据
    logger.info(f"📦 Loading dataset from {args.data}...")
    data_payload = torch.load(args.data, map_location="cpu")
    
    if isinstance(data_payload, dict):
        all_latents = data_payload['latents'] # (N, 4, 128, 128)
        all_labels = data_payload['labels']   # (N,)
        # 自动获取类别数
        num_classes = int(torch.max(all_labels).item()) + 1
    else:
        raise ValueError("Unsupported .pt format")
        
    logger.info(f"📊 Dataset: {len(all_latents)} images, {num_classes} classes")
    
    dataset = TensorDataset(all_latents, all_labels)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )

    # 2. 初始化模型 Config
    if args.debug:
        logger.warning("🐛 DEBUG MODE: Using Tiny DiT")
        config = DiTConfig(
            input_size=128, patch_size=2, hidden_size=64, depth=2, num_heads=4, num_classes=num_classes + 1
        )
    else:
        # 标准 Small 配置
        config = DiTConfig(
            input_size=128, patch_size=2, hidden_size=384, depth=12, num_heads=6, num_classes=num_classes + 1
        )
        
    model = DiT(config).to(device)
    
    # 统计参数
    params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"🧠 Model Initialized ({params:.2f}M params)")

    # 3. 优化器 (启用 Fused)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, fused=True)
    criterion = nn.MSELoss()

    # 4. Checkpoint 管理
    ckpt_manager = CheckpointManager(args.output_dir, args.name)
    start_epoch = 0
    
    # 尝试恢复训练
    if not args.force_restart:
        start_epoch = ckpt_manager.load(args.resume, model, optimizer)
        # 将优化器状态移动到 GPU (因为 load 是在 cpu 做的)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # 5. 编译模型 (Resume 之后再编译)
    logger.info("🔥 Compiling model with Triton (mode='max-autotune')...")
    # max-autotune 可能会慢，如果你觉得卡住太久，可以改成 'default'
    model = torch.compile(model, mode="max-autotune") 

    # 6. 信号捕获 (Ctrl+C)
    def signal_handler(sig, frame):
        logger.info("\n🛑 Interrupt received! Saving emergency checkpoint...")
        ckpt_manager.save(model, optimizer, epoch, avg_loss, config)
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # 7. 训练循环
    logger.info(f"🎬 Training Start: Epoch {start_epoch} -> {args.epochs}")
    model.train()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        steps = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        
        for latents, labels in progress_bar:
            latents = latents.to(device, non_blocking=True).to(torch.bfloat16) # BF16
            labels = labels.to(device, non_blocking=True)
            
            # --- Diffusion Forward (Simple Linear Schedule) ---
            t = torch.randint(0, 1000, (latents.shape[0],), device=device)
            noise = torch.randn_like(latents)
            
            # 简单的加噪 (以后可以换成更复杂的 Scheduler)
            # x_t = (1-alpha) * x + alpha * noise
            alpha = (t.view(-1, 1, 1, 1) / 1000.0) 
            x_t = (1 - alpha) * latents + alpha * noise
            
            target = noise # 预测噪声 (Epsilon-Prediction)
            
            optimizer.zero_grad(set_to_none=True)
            
            # --- Mixed Precision Training (BF16) ---
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(x_t, t, labels)
                loss = criterion(pred, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            steps += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / steps
        
        # --- 定期保存 ---
        if (epoch + 1) % args.save_every == 0:
            ckpt_manager.save(model, optimizer, epoch, avg_loss, config)
            
    # 训练结束保存
    ckpt_manager.save(model, optimizer, args.epochs-1, avg_loss, config)
    logger.info("🏁 Training Finished Successfully!")

if __name__ == "__main__":
    main()