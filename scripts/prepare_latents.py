import argparse
import logging
from little_diffusion.processor import VAEProcessor

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Industrial VAE Pre-processor")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input image folder")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output .pt file")
    parser.add_argument("--size", type=int, default=704, help="Target image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (Try 16/32 for 5070Ti)")
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-mse", help="VAE model name")
    
    args = parser.parse_args()
    
    # 1. 初始化引擎
    processor = VAEProcessor(model_name=args.vae)
    
    # 2. 执行任务
    processor.process_folder(
        input_dir=args.input,
        output_path=args.output,
        image_size=args.size,
        batch_size=args.batch
    )

if __name__ == "__main__":
    main()