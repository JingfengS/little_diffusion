import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path

# 确保能找到项目根目录

def test_single_image(image_path):
    print(f"\n🔍 Testing Image: {image_path}")
    
    if not os.path.exists(image_path):
        print("❌ Image not found!")
        return

    try:
        from imgutils.segment import segment_rgba_with_isnetis
        print("✅ Successfully imported segment_rgba_with_isnetis")
    except ImportError:
        print("❌ Failed to import imgutils. Please install: pip install dghs-imgutils[gpu]")
        return

    # 1. 加载图片
    raw_img = Image.open(image_path).convert("RGB")
    print(f"   Original Size: {raw_img.size}")

    # 2. 运行模型
    print("🚀 Running AI Inference...")
    # 注意：segment_rgba_with_isnetis 返回 (mask, rgba_image)
    mask_raw, rgba_img = segment_rgba_with_isnetis(raw_img)

    # 3. 深度分析 Mask
    print("\n📊 Mask Analysis:")
    print(f"   Type: {type(mask_raw)}")
    
    if isinstance(mask_raw, np.ndarray):
        print(f"   Shape: {mask_raw.shape}")
        print(f"   Dtype: {mask_raw.dtype}")
        print(f"   Min Value: {mask_raw.min()}")
        print(f"   Max Value: {mask_raw.max()}")
        print(f"   Mean Value: {mask_raw.mean()}")
        
        # 4. 尝试修复并保存
        debug_dir = Path("data/debug_output")
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # 🟢 关键逻辑：如果最大值 <= 1.0，说明是概率图，需要乘 255
        if mask_raw.max() <= 1.0001: 
            print("   ⚠️ Detected float probability (0-1). Scaling to 0-255...")
            mask_fixed = (mask_raw * 255).astype(np.uint8)
        else:
            print("   ✅ Detected uint8 range (0-255).")
            mask_fixed = mask_raw.astype(np.uint8)
            
        # 转为 PIL 并保存
        mask_pil = Image.fromarray(mask_fixed, mode='L')
        save_path = debug_dir / "debug_mask.png"
        mask_pil.save(save_path)
        print(f"   💾 Saved normalized mask to: {save_path}")
        
        # 保存 NumPy 原始数据以便进一步检查
        np.save(debug_dir / "debug_mask.npy", mask_raw)
        print(f"   💾 Saved raw numpy array to: {debug_dir / 'debug_mask.npy'}")
        
        # 保存 RGBA 结果
        rgba_path = debug_dir / "debug_rgba.png"
        rgba_img.save(rgba_path)
        print(f"   💾 Saved matting result to: {rgba_path}")
        
    else:
        print(f"   ❌ Unexpected mask type: {type(mask_raw)}")

if __name__ == "__main__":
    # 在这里填入你的一张测试图片的路径
    # 如果没有参数，默认找一张图
    target_img = "data/raw/21/立绘（公测19年至21年）/立绘（公测19年至21年）/01 基本/古米精英二.png" 
    
    # 支持命令行参数: python scripts/debug_vision.py path/to/image.png
    if len(sys.argv) > 1:
        target_img = sys.argv[1]
        
    test_single_image(target_img)