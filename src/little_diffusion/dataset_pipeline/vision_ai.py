import numpy as np
from PIL import Image
import logging
import traceback

# 尝试导入 imgutils
try:
    from imgutils.segment import segment_rgba_with_isnetis
    from imgutils.detect import detect_faces

    HAS_IMGUTILS = True
except ImportError:
    HAS_IMGUTILS = False

logger = logging.getLogger(__name__)


class MattingEngine:
    """专为二次元立绘打造的抠图引擎 (基于 dghs-imgutils 的 IS-Net)"""

    def __init__(self, device=None):
        if not HAS_IMGUTILS:
            logger.error(
                "❌ dghs-imgutils not installed! Please run: pip install dghs-imgutils[gpu]"
            )
            raise ImportError("dghs-imgutils is required.")

        # 绑定函数引用，不需要额外初始化
        self.segment_func = segment_rgba_with_isnetis
        self.has_backend = True
        logger.info(
            "🚀 dghs-imgutils IS-Net (Anime Character Matting) loaded successfully!"
        )

    def get_alpha_mask(self, img: Image.Image) -> Image.Image:
        """输入 RGB/RGBA 图像，返回只包含【人物主体】的 Alpha 蒙版"""
        if not self.has_backend:
            raise RuntimeError("Anime Matting Engine is not initialized.")

        try:
            # 0. 预处理：转为 RGB 以获得最佳兼容性
            if img.mode != 'RGB':
                input_img = img.convert("RGB")
            else:
                input_img = img

            # 1. 核心调用：(mask, rgba_image)
            # mask 是 numpy.ndarray, rgba_image 是 PIL.Image
            mask, _ = self.segment_func(input_img)
            
            mask = (mask * 255).astype(np.uint8) 
            mask = Image.fromarray(mask)
            # 3. 格式统一化
            if mask.mode != 'L':
                mask = mask.convert("L")
                
            # 4. 尺寸安全检查
            if mask.size != img.size:
                mask = mask.resize(img.size, Image.Resampling.NEAREST)
                
            return mask

        except Exception as e:
            logger.error(f"Error during anime matting: {e}")
            traceback.print_exc()
            # 容错：如果失败，返回全白 Mask
            return Image.new("L", img.size, 255)


class FaceDetector:
    """封装 dghs-imgutils 的人脸检测"""

    def __init__(self):
        self.has_backend = HAS_IMGUTILS
        if not HAS_IMGUTILS:
            logger.warning("⚠️ imgutils not found. Face detection will fail.")

    def get_best_face_box(self, img: Image.Image, confidence=0.5):
        if not self.has_backend:
            return None
        try:
            detections = detect_faces(img)
            if not detections:
                return None
            best_face = max(detections, key=lambda x: x[2])
            box, _, score = best_face
            return box if score > confidence else None
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return None
