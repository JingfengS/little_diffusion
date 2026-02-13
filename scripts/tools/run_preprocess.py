import argparse
import logging


# 🌟 切换到新的 AI 流水线
from little_diffusion.dataset_pipeline.main_pipeline import ArknightsPipeline
from little_diffusion.dataset_pipeline.config import PipelineConfig

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="🚀 Arknights AI ETL Pipeline (RMBG-1.4 + Weight Maps)")
    
    # 路径参数
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw images")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Path to output")
    parser.add_argument("--whitelist", type=str, default="scripts/tools/operators.txt", help="Path to whitelist")
    
    # ⚠️ 关于多进程：
    # 因为 RMBG-1.4 模型需要占用显存，且 Python 多进程传大模型很麻烦
    # 为了稳定性，新版 Pipeline 默认在主进程跑 (单卡 4090/5070 跑 Matting 也是秒级，这通常不是瓶颈)
    # parser.add_argument("--workers", type=int, default=1, help="Deprecated in AI mode") 

    args = parser.parse_args()

    # 1. 构建配置对象
    # 如果你想调整权重或者像素面积，可以在这里改
    pipeline_config = PipelineConfig(
        raw_dir=args.raw_dir,
        processed_dir=args.out_dir,
        whitelist_path=args.whitelist,
        target_pixel_area=1024 * 1024, # 1MP
    )
    
    # 2. 打印一下配置确认
    print("==========================================")
    print("🔧 Pipeline Configuration")
    print(f"   Input:  {args.raw_dir}")
    print(f"   Output: {args.out_dir}")
    print(f"   Whitelist: {args.whitelist}")
    print("   AI Engine: Face Detection")
    print("==========================================")

    # 3. 启动引擎
    try:
        pipeline = ArknightsPipeline(pipeline_config)
        pipeline.run()
    except Exception as e:
        logging.error(f"❌ Pipeline failed to initialize: {e}")
        return

if __name__ == "__main__":
    main()