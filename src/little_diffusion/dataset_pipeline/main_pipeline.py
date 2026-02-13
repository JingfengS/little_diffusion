import os
import json
import hashlib
import logging
import torch
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image
from tqdm import tqdm

# 引入项目组件
from little_diffusion.dataset_pipeline.config import PipelineConfig, ImageMeta, CropType
from little_diffusion.dataset_pipeline.vision_ai import MattingEngine, FaceDetector
from little_diffusion.dataset_pipeline.transforms import WeightMapGenerator, SafeCropper

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class ArknightsPipeline:
    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config

        # 1. 准备目录
        self.out_root = Path(config.processed_dir)
        self.img_dir = self.out_root / "images"
        self.mask_dir = self.out_root / "masks"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)

        # 2. 加载白名单
        self.whitelist = []
        self.char_to_id = {}
        self._load_whitelist()

        # 3. 初始化 AI 引擎
        # 注意：MattingEngine 比较吃显存，建议单进程运行
        logger.info("🔧 Initializing AI Engines...")
        self.matting = MattingEngine(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.face_detector = FaceDetector()
        self.weight_gen = WeightMapGenerator(
            fg_weight=config.weights.fg_weight,
            complex_bg_weight=config.weights.bg_complex_weight,
            pure_bg_weight=config.weights.bg_pure_weight,
        )
        logger.info("✅ Engines Ready.")

    def _load_whitelist(self):
        path = Path(self.config.whitelist_path)
        if not path.exists():
            raise FileNotFoundError(f"Whitelist not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            self.whitelist = [line.strip() for line in f if line.strip()]
        # 按长度降序排列，防止"陈"匹配到"假日威龙陈"
        self.whitelist.sort(key=len, reverse=True)
        self.char_to_id = {name: idx for idx, name in enumerate(self.whitelist)}
        logger.info(f"📋 Loaded {len(self.whitelist)} operators.")

    def _match_character(self, filename: str) -> Optional[str]:
        for kw in self.config.blacklist_keywords:
            if kw.upper() in filename.upper():
                return None
        for name in self.whitelist:
            if name in filename:
                return name
        return None

    def _save_pair(
        self, img: Image.Image, mask: Image.Image, base_name: str, suffix: str
    ) -> Tuple[str, str]:
        """保存 RGB 和 Mask 对，返回相对路径"""
        img_name = f"{base_name}_{suffix}.jpg"
        mask_name = f"{base_name}_{suffix}.png"  # Mask 存为 PNG 无损灰度

        img.save(self.img_dir / img_name, quality=95)
        mask.save(self.mask_dir / mask_name)

        return f"images/{img_name}", f"masks/{mask_name}"

    def process_single_file(self, file_path: Path, char_name: str) -> List[ImageMeta]:
        results = []
        label_id = self.char_to_id[char_name]

        try:
            # 1. 加载原图
            with Image.open(file_path) as raw_img:
                raw_img = raw_img.convert("RGBA")

                # 🔙 回滚逻辑：使用原图的 bbox (保留特效/背景)，而不是 AI Mask 的 bbox
                # 这样 Full Body 就会包含光翼、替身、废墟等元素，保持多样性
                bbox = raw_img.getbbox()
                if not bbox:
                    return []
                img_trimmed = raw_img.crop(bbox)

            # 2. AI 介入
            # 虽然我们保留了背景，但我们依然需要 AI 告诉我们“哪里是人”
            # 用于生成 Weight Map (人=1.0, 背景特效=0.1)
            ai_mask = self.matting.get_alpha_mask(img_trimmed)
            weight_map = self.weight_gen.generate(img_trimmed, ai_mask)

            # 3. 合成白底图
            rgb_white_bg = Image.new("RGB", img_trimmed.size, (255, 255, 255))
            rgb_white_bg.paste(img_trimmed, mask=img_trimmed.split()[3])

            # 准备基础信息
            file_hash = hashlib.md5(str(file_path).encode("utf-8")).hexdigest()[:6]
            base_name = f"{label_id}_{char_name}_{file_hash}"
            w, h = rgb_white_bg.size

            # 🔥 预先检测人脸 (服务于 Face 和 Half 策略)
            face_box = self.face_detector.get_best_face_box(rgb_white_bg)
            if face_box:
                fx1, fy1, fx2, fy2 = face_box
                face_cx = (fx1 + fx2) / 2
                face_cy = (fy1 + fy2) / 2
                face_h = fy2 - fy1
            else:
                face_cx = w / 2
                face_cy = h * 0.2
                face_h = h * 0.1

            # ================= 策略 A: Full Body (等比缩放) =================
            scale = (self.config.target_pixel_area / (w * h)) ** 0.5
            if scale < 1.0:
                target_w, target_h = int(w * scale), int(h * scale)
                img_full = rgb_white_bg.resize(
                    (target_w, target_h), Image.Resampling.LANCZOS
                )
                # ⚠️ Mask 必须同步缩放
                mask_full = weight_map.resize(
                    (target_w, target_h), Image.Resampling.NEAREST
                )
            else:
                img_full = rgb_white_bg
                mask_full = weight_map

            p_img, p_mask = self._save_pair(
                img_full, mask_full, base_name, CropType.FULL.value
            )
            results.append(
                ImageMeta(
                    p_img,
                    p_mask,
                    char_name,
                    label_id,
                    CropType.FULL.value,
                    str(file_path),
                )
            )

            # ================= 策略 B: Face Crop (智能大头照) =================
            # 使用 FaceDetector 在 RGB 图上找脸
            if face_box:
                # 扩图逻辑 (2.0倍，包含头发)
                cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                span = max(fx2 - fx1, fy2 - fy1) * 2.0
                span = max(span, 512)  # 最小尺寸限制
                half_span = span / 2
                crop_box = (
                    cx - half_span,
                    cy - half_span,
                    cx + half_span,
                    cy + half_span,
                )  # 可能超界

                # 使用 SafeCropper 同步裁剪 RGB 和 Weight Map
                img_face = SafeCropper.crop_and_pad(
                    rgb_white_bg, crop_box, fill_color=(255, 255, 255)
                )
                mask_face = SafeCropper.crop_and_pad(
                    weight_map,
                    crop_box,
                    fill_color=int(self.config.weights.bg_complex_weight * 255),
                )
                # Mask 填充默认给复杂背景权重，比较安全

                # 统一 Resize 到 768x768
                target_s = self.config.face_crop_size
                img_face = img_face.resize(
                    (target_s, target_s), Image.Resampling.LANCZOS
                )
                mask_face = mask_face.resize(
                    (target_s, target_s), Image.Resampling.NEAREST
                )

                p_img, p_mask = self._save_pair(
                    img_face, mask_face, base_name, CropType.FACE.value
                )
                results.append(
                    ImageMeta(
                        p_img,
                        p_mask,
                        char_name,
                        label_id,
                        CropType.FACE.value,
                        str(file_path),
                    )
                )

            # ================= 策略 C: Half Body (Letterbox) =================
            target_w_final, target_h_final = self.config.half_body_size
            target_ar = target_w_final / target_h_final  # 0.75

            if face_box:
                # 1. 确定 Top: 头顶向上留 1.2 倍脸长 (够放光环/耳朵)
                #    max(0, ...) 防止切出上边界
                crop_top = int(max(0, fy1 - face_h * 1.2))

                # 2. 确定 Bottom: 头底向下延伸 6.5 倍脸长 (Head + Body + Thighs)
                #    min(h, ...) 防止切出下边界
                #    注意：这里我们实际上是在定义一个“理想的半身高度”
                ideal_bottom = int(fy2 + face_h * 6.5)
                crop_bottom = int(min(h, ideal_bottom))

                # 3. 确定最终高度
                crop_h = crop_bottom - crop_top

                # 如果脸太小或者计算出的高度太小，强行保底 (防止切出极小的图)
                if crop_h < 512:
                    crop_h = int(min(h, 1024))
                    crop_bottom = crop_top + crop_h
            else:
                # Fallback: 如果没脸，回退到原来的逻辑 (Top 0, Height 60%)
                crop_top = 0
                crop_h = int(h * 0.6)
                crop_bottom = crop_h

            # 4. 根据高度和比例 (3:4) 反推宽度
            crop_w = int(crop_h * target_ar)

            # 5. 确定左右边界 (以人脸/中心为轴)
            half_crop_w = crop_w / 2
            left = int(face_cx - half_crop_w)
            right = int(face_cx + half_crop_w)

            # 6. 边界修正 (Shift & Clamp)
            if left < 0:
                right -= left
                left = 0
            if right > w:
                left -= right - w
                right = w
                if left < 0:
                    left = 0

            # 7. 执行裁剪
            img_half = rgb_white_bg.crop((left, crop_top, right, crop_bottom))
            mask_half = weight_map.crop((left, crop_top, right, crop_bottom))

            # 8. Resize (保持纵横比，宽度不足补白边)
            scale_h = target_h_final / img_half.height
            new_w_res = int(img_half.width * scale_h)

            img_half_res = img_half.resize(
                (new_w_res, target_h_final), Image.Resampling.LANCZOS
            )
            mask_half_res = mask_half.resize(
                (new_w_res, target_h_final), Image.Resampling.NEAREST
            )

            final_img = Image.new(
                "RGB", (target_w_final, target_h_final), (255, 255, 255)
            )
            final_mask = Image.new(
                "L",
                (target_w_final, target_h_final),
                int(self.config.weights.bg_pure_weight * 255),
            )

            paste_x = (target_w_final - new_w_res) // 2
            final_img.paste(img_half_res, (paste_x, 0))
            final_mask.paste(mask_half_res, (paste_x, 0))

            p_img, p_mask = self._save_pair(
                final_img, final_mask, base_name, CropType.HALF.value
            )
            results.append(
                ImageMeta(
                    p_img,
                    p_mask,
                    char_name,
                    label_id,
                    CropType.HALF.value,
                    str(file_path),
                )
            )

        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            # 出错时不中断，直接返回空
            return []

        return results

    def run(self):
        logger.info(f"🚀 Starting Data Pipeline")
        logger.info(f"   Source: {self.config.raw_dir}")
        logger.info(f"   Target: {self.config.processed_dir}")

        # 1. 扫描文件
        tasks = []
        raw_path = Path(self.config.raw_dir)
        for root, _, files in os.walk(raw_path):
            if any(kw in root for kw in self.config.blacklist_keywords):
                continue

            for file in files:
                if Path(file).suffix not in self.config.allowed_extensions:
                    continue
                char_name = self._match_character(file)
                if char_name:
                    tasks.append((Path(root) / file, char_name))

        logger.info(f"🔍 Found {len(tasks)} valid images.")

        # 2. 顺序执行 (GPU Matting 难以并行，且 Python GIL 限制)
        # 如果需要加速，可以考虑使用 PyTorch DataLoader 的多进程模式，但这里简单起见用循环
        all_meta = []
        for file_path, char_name in tqdm(tasks, desc="Processing"):
            metas = self.process_single_file(file_path, char_name)
            all_meta.extend(metas)

        # 3. 保存索引
        logger.info("💾 Saving Dataset Metadata...")
        meta_dicts = [m.__dict__ for m in all_meta]
        with open(self.out_root / "dataset.json", "w", encoding="utf-8") as f:
            json.dump(meta_dicts, f, indent=2, ensure_ascii=False)

        with open(self.out_root / "id_map.json", "w", encoding="utf-8") as f:
            json.dump(self.char_to_id, f, indent=2, ensure_ascii=False)

        logger.info("✨ Data Pipeline Completed Successfully!")


if __name__ == "__main__":
    pipeline = ArknightsPipeline()
    pipeline.run()
