import torch
import argparse
import logging
import sys
import random
from pathlib import Path
from PIL import Image

# 确保能导入 src
sys.path.append(str(Path(__file__).parent.parent / "src"))

from little_diffusion.processor import VAEProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Inspect Latents by Decoding them back to Pixels")
    parser.add_argument("--pt_file", type=str, default="data/processed/latents.pt", help="Path to .pt file")
    parser.add_argument("--output_dir", type=str, default="data/processed/inspection", help="Where to save decoded images")
    parser.add_argument("--num_samples", type=int, default=10, help="How many random samples to check")
    parser.add_argument("--vae", type=str, default="madebyollin/sdxl-vae-fp16-fix", help="VAE model name (Must match preparation!)")
    
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载 Latents 文件
    if not Path(args.pt_file).exists():
        logger.error(f"❌ Latent file not found: {args.pt_file}")
        return

    logger.info(f"📦 Loading {args.pt_file}...")
    data = torch.load(args.pt_file, map_location="cpu")
    
    # 兼容处理：检查是新版 Dict 格式还是旧版 Tensor 格式
    if isinstance(data, dict):
        latents = data['latents']
        labels = data['labels']
        # 优先使用文件里记录的 scaling factor，如果没有则用 SDXL 默认值
        scaling_factor = data.get('scaling_factor', 0.13025)
        logger.info(f"   -> Found metadata. Scaling Factor: {scaling_factor}")
    else:
        latents = data
        labels = None
        scaling_factor = 0.13025 # 假设是 SDXL
        logger.warning("⚠️ Legacy tensor format detected. Assuming SDXL scaling factor.")
        
    total_imgs = len(latents)
    logger.info(f"📊 Dataset Stats: {total_imgs} images, Latent Shape: {latents.shape[1:]}")
    
    # 2. 随机抽样
    indices = random.sample(range(total_imgs), min(args.num_samples, total_imgs))
    selected_latents = latents[indices]
    selected_labels = labels[indices] if labels is not None else None
    
    # 3. 初始化 VAE 解码器
    logger.info(f"🚀 Loading VAE: {args.vae}...")
    processor = VAEProcessor(model_name=args.vae, scaling_factor=scaling_factor)
    
    # 4. 解码 (Latent -> Pixel)
    logger.info("🎨 Decoding latents...")
    decoded_images = processor.decode(selected_latents)
    
    # 5. 保存结果
    logger.info(f"💾 Saving inspections to {out_dir}...")
    for i, (idx, img) in enumerate(zip(indices, decoded_images)):
        label_info = f"_class{selected_labels[i].item()}" if selected_labels is not None else ""
        save_name = f"inspect_{i:02d}_idx{idx}{label_info}.png"
        img.save(out_dir / save_name)
        logger.info(f"   -> Saved {save_name}")
        
    logger.info("✅ Done! Go check the images.")

if __name__ == "__main__":
    main()