import argparse
import logging

from little_diffusion.preprocessing.arknights import ArknightsPreprocessor, PreprocessConfig

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Arknights Data ETL Pipeline")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to the raw data folder (Immutable)")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Path to output processed data")
    parser.add_argument("--whitelist", type=str, default="scripts/tools/operators.txt", help="Path to the character whitelist file")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers")
    
    args = parser.parse_args()

    # 可以在这里修改配置，或者通过 argparse 暴露更多参数
    config = PreprocessConfig(
        target_pixel_area=1024 * 1024, # 针对 2K 图片优化
        face_crop_size=768
    )

    processor = ArknightsPreprocessor(config)
    
    try:
        processor.load_whitelist(args.whitelist)
    except Exception as e:
        logging.error(f"❌ Failed to load whitelist: {e}")
        return

    processor.run(args.raw_dir, args.out_dir, args.workers)

if __name__ == "__main__":
    main()