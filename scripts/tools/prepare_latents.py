import argparse
import logging
import sys
from pathlib import Path

# 确保能导入 src
sys.path.append(str(Path(__file__).parent.parent / "src"))

from little_diffusion.processor import VAEProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Latent Extraction Pipeline")
    # 输入不再是 folder，而是 metadata json 和 image root
    parser.add_argument("--meta", type=str, default="data/processed/dataset.json", help="Path to dataset.json")
    parser.add_argument("--root", type=str, default="data/processed/images", help="Path to images folder")
    parser.add_argument("--output", "-o", type=str, default="data/processed/latents.pt", help="Output .pt file")
    
    # SDXL VAE 默认是 1024，但你的图是 768，这里设为 768 没问题
    parser.add_argument("--size", type=int, default=1024, help="Target image size") 
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    
    # 默认使用 SDXL VAE
    parser.add_argument("--vae", type=str, default="madebyollin/sdxl-vae-fp16-fix", help="VAE model name")
    
    args = parser.parse_args()
    
    # 1. 初始化引擎
    # SDXL 的 scaling factor 是 0.13025 (代码里默认值已更正，但也可以显式传)
    processor = VAEProcessor(model_name=args.vae, scaling_factor=0.13025)
    
    # 2. 执行任务
    processor.process_dataset(
        metadata_path=args.meta,
        image_root=args.root,
        output_path=args.output,
        image_size=args.size,
        batch_size=args.batch
    )

if __name__ == "__main__":
    main()