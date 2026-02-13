import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)

class MattingEngine:
    """RMBG-1.4 高级背景去除引擎"""
    def __init__(self, device="cuda"):
        self.device = device
        logger.info("🚀 Loading RMBG-1.4 Matting Model...")
        # Bria AI 的 RMBG-1.4 是目前开源最强抠图模型
        self.model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-1.4", trust_remote_code=True
        ).to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def get_alpha_mask(self, img: Image.Image) -> Image.Image:
        """输入 RGB 图像，返回灰度 Alpha 蒙版"""
        orig_size = img.size
        input_tensor = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
        
        # 模型预测
        preds = self.model(input_tensor)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        
        # 转回 PIL 并恢复原图大小
        mask_np = (pred.numpy() * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np, mode="L").resize(orig_size, Image.Resampling.LANCZOS)
        return mask_pil

class FaceDetector:
    """封装 dghs-imgutils 的人脸检测"""
    def __init__(self):
        try:
            from imgutils.detect import detect_faces
            self.detect_faces = detect_faces
            self.has_backend = True
        except ImportError:
            self.has_backend = False
            logger.warning("⚠️ imgutils not found. Face detection will fallback to center crop.")

    def get_best_face_box(self, img: Image.Image, confidence=0.5):
        if not self.has_backend:
            return None
        try:
            detections = self.detect_faces(img)
            if not detections:
                return None
            best_face = max(detections, key=lambda x: x[2])
            box, _, score = best_face
            return box if score > confidence else None
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return None