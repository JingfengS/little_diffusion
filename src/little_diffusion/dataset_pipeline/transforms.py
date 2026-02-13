from PIL import Image, ImageStat
import numpy as np

class WeightMapGenerator:
    """根据你的构想生成 3 阶 Masked Loss 权重图"""
    def __init__(self, fg_weight=1.0, complex_bg_weight=0.1, pure_bg_weight=0.01):
        self.w_fg = int(fg_weight * 255)
        self.w_complex = int(complex_bg_weight * 255)
        self.w_pure = int(pure_bg_weight * 255)

    def generate(self, original_img: Image.Image, ai_character_mask: Image.Image) -> Image.Image:
        """
        Args:
            original_img: 原始图像 (可能是 RGBA)
            ai_character_mask: IS-Net 提取的角色蒙版 (L mode)
        """
        # 1. 获取原始内容的范围 (Original Content Mask)
        if original_img.mode == 'RGBA':
            orig_alpha = np.array(original_img.split()[3])
        else:
            # 如果是 JPG，假设全图都是内容（除非 AI 说是背景）
            orig_alpha = np.full(original_img.size[::-1], 255, dtype=np.uint8)

        # 2. 获取 AI 识别的角色范围 (Character Mask)
        char_alpha = np.array(ai_character_mask.resize(original_img.size, Image.Resampling.NEAREST))

        # 3. 初始化权重图 (默认为纯背景)
        weight_map = np.full_like(orig_alpha, self.w_pure)

        # 4. 逻辑分层填充
        
        # Layer 1: 复杂背景/特效 (Complex Background)
        # 逻辑：原图里有东西 (orig_alpha > 128)，但 AI 觉得不是人
        # 这通常就是精二立绘里的：源石结晶、替身、光翼、爆炸特效
        complex_mask = (orig_alpha > 128)
        weight_map[complex_mask] = self.w_complex

        # Layer 2: 角色主体 (Foreground)
        # 逻辑：AI 觉得是人的地方
        char_mask = (char_alpha > 128)
        weight_map[char_mask] = self.w_fg
        
        return Image.fromarray(weight_map, mode="L")

class SafeCropper:
    """修复超界变黑 Bug 的安全裁剪器"""
    @staticmethod
    def crop_and_pad(img: Image.Image, box: tuple, fill_color=(255, 255, 255)) -> Image.Image:
        """就算 box 超出了图片边界，也能用白色安全填充"""
        left, top, right, bottom = [int(v) for v in box]
        crop_w, crop_h = right - left, bottom - top

        # 1. 确保 fill_color 格式正确
        # 如果图像是灰度图(L)，fill_color 必须是 int
        if img.mode == 'L' and isinstance(fill_color, tuple):
            fill_color = fill_color[0] # 取第一个通道的值
        
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