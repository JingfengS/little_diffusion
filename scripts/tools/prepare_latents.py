import argparse
import logging
import sys
from pathlib import Path

# 确保能导入 src
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

# 👇 引用变了！
from little_diffusion.preprocessing.processor import VAEProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Latent Extraction Pipeline")
    parser.add_argument("--meta", type=str, default="data/processed/dataset.json", help="Path to dataset.json")
    parser.add_argument("--root", type=str, default="data/processed/", help="Path to images folder")
    parser.add_argument("--output", "-o", type=str, default="data/processed/arknights_latents.pt", help="Output .pt file")
    
    parser.add_argument("--size", type=int, default=1024, help="Target image size") 
    parser.add_argument("--batch", type=int, default=4, help="Batch size (reduce if OOM)")
    parser.add_argument("--vae", type=str, default="madebyollin/sdxl-vae-fp16-fix", help="VAE model name")
    
    args = parser.parse_args()
    
    # 1. 初始化引擎
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