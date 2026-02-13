from PIL import Image
from typing import Tuple

class SquarePadResize:
    """
    智能缩放填充 (支持 Image 和 Mask 同步处理)：
    1. 保持比例缩放，让长边 = target_size
    2. 短边用指定颜色填充 (Pad) 到 target_size
    """

    def __init__(
        self,
        target_size: int,
        img_fill: tuple = (255, 255, 255),
        mask_fill: float = 0.01,
    ):
        self.target_size = target_size
        self.img_fill = img_fill
        # 0.01 是我们为纯色白边分配的极低权重
        self.mask_fill = int(mask_fill * 255)

    def __call__(
        self, img: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        w, h = img.size

        # 1. 计算缩放比例 (基于长边)
        ratio = self.target_size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)

        # 2. 同步缩放
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        mask = mask.resize(
            (new_w, new_h), Image.Resampling.NEAREST
        )  # Mask 建议用 NEAREST 防止边缘模糊

        # 3. 创建正方形画布
        new_img = Image.new("RGB", (self.target_size, self.target_size), self.img_fill)
        new_mask = Image.new("L", (self.target_size, self.target_size), self.mask_fill)

        # 4. 居中粘贴
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2

        new_img.paste(img, (paste_x, paste_y))
        new_mask.paste(mask, (paste_x, paste_y))

        return new_img, new_mask