from PIL import Image, ImageStat
import numpy as np

class WeightMapGenerator:
    """根据你的构想生成 3 阶 Masked Loss 权重图"""
    def __init__(self, fg_weight=1.0, complex_bg_weight=0.1, pure_bg_weight=0.01):
        self.w_fg = int(fg_weight * 255)
        self.w_complex = int(complex_bg_weight * 255)
        self.w_pure = int(pure_bg_weight * 255)

    def generate(self, original_img: Image.Image, alpha_mask: Image.Image) -> Image.Image:
        """
        核心逻辑：
        1. Alpha > 200 -> 主体 (w_fg)
        2. Alpha < 50 -> 检查原图这部分方差。方差大 -> 复杂背景 (w_complex)；方差小 -> 纯白板 (w_pure)
        """
        img_np = np.array(original_img.convert("RGB"))
        alpha_np = np.array(alpha_mask)
        
        # 提取背景区域的像素
        bg_mask = alpha_np < 50
        bg_pixels = img_np[bg_mask]
        
        # 计算背景的色彩方差 (判断是白板还是废墟)
        is_pure_bg = False
        if len(bg_pixels) > 100:
            variance = np.var(bg_pixels)
            if variance < 50: # 方差极小，说明是平滑/纯色背景
                is_pure_bg = True
                
        # 构建权重图
        weight_np = np.full_like(alpha_np, self.w_complex) # 默认填入复杂背景权重
        if is_pure_bg:
            weight_np = np.full_like(alpha_np, self.w_pure) # 替换为白板权重
            
        # 主体权重覆盖 (保留一定的边缘软过渡)
        fg_mask = alpha_np >= 50
        weight_np[fg_mask] = np.maximum(
            weight_np[fg_mask], 
            (alpha_np[fg_mask] / 255.0 * self.w_fg).astype(np.uint8)
        )
        
        return Image.fromarray(weight_np, mode="L")

class SafeCropper:
    """修复超界变黑 Bug 的安全裁剪器"""
    @staticmethod
    def crop_and_pad(img: Image.Image, box: tuple, fill_color=(255, 255, 255)) -> Image.Image:
        """就算 box 超出了图片边界，也能用白色安全填充"""
        left, top, right, bottom = [int(v) for v in box]
        crop_w, crop_h = right - left, bottom - top
        
        # 创建一块安全的白板
        canvas = Image.new(img.mode, (crop_w, crop_h), fill_color)
        
        # 计算在原图上的合法区域
        src_left = max(0, left)
        src_top = max(0, top)
        src_right = min(img.width, right)
        src_bottom = min(img.height, bottom)
        
        if src_right <= src_left or src_bottom <= src_top:
            return canvas # 完全在图外，直接返回白板
            
        # 裁剪合法区域
        valid_crop = img.crop((src_left, src_top, src_right, src_bottom))
        
        # 粘贴到白板的正确位置
        paste_x = src_left - left
        paste_y = src_top - top
        canvas.paste(valid_crop, (paste_x, paste_y))
        
        return canvas