import os
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Union, Set, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

from PIL import Image
from tqdm import tqdm

# 配置 Logger
logger = logging.getLogger(__name__)

# =========================================================
# 🔄 依赖切换: dghs-imgutils
# =========================================================
try:
    from imgutils.detect import detect_faces

    HAS_IMGUTILS = True
except ImportError:
    HAS_IMGUTILS = False
    logger.warning(
        "⚠️ dghs-imgutils not found. Install it for SOTA cropping: pip install dghs-imgutils[opencv]"
    )


class CropType(Enum):
    """定义裁剪类型的枚举，防止魔法字符串"""

    FULL = "full"
    FACE = "face"
    HALF = "half"


@dataclass
class PreprocessConfig:
    """
    配置类：将所有硬编码参数提取出来
    """

    target_pixel_area: int = 1024 * 1024  # 目标像素量 (1MP)
    face_crop_size: int = 768  # 大头照尺寸
    half_body_size: Tuple[int, int] = (768, 1024)  # 半身照尺寸

    # 黑名单：遇到这些词直接跳过
    blacklist_keywords: List[str] = field(
        default_factory=lambda: [
            "NPC",
            "IMG",
            "avg",
            "怪物",
            "敌方",
            "token",
            "trap",
            "整合运动",
            "龙门士兵",
            "路人",
            "黑帮",
            "保镖",
            "游客",
            "人物介绍",
            "小车",
        ]
    )

    # 允许的文件扩展名
    allowed_extensions: Set[str] = field(
        default_factory=lambda: {".png", ".jpg", ".jpeg", ".PNG", ".JPG"}
    )


@dataclass
class ImageMeta:
    """
    数据传输对象 (DTO)，确保 dataset.json 的结构稳定
    """

    file_path: str  # 相对路径
    character: str  # 角色名
    class_id: int  # 数字 ID
    type: str  # 裁剪类型
    original_path: str  # 溯源路径


class ArknightsPreprocessor:
    """
    明日方舟立绘专用 ETL 处理器
    Extract: 递归扫描
    Transform: 智能裁剪、缩放、去背
    Load: 保存为扁平化数据集
    """

    def __init__(self, config: PreprocessConfig = PreprocessConfig()):
        self.config = config
        self.whitelist: List[str] = []
        self.char_to_id: Dict[str, int] = {}

    def load_whitelist(self, txt_path: Union[str, Path]) -> None:
        """加载白名单并构建ID映射"""
        path = Path(txt_path)
        if not path.exists():
            raise FileNotFoundError(f"Whitelist file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
            self.whitelist = sorted(list(set(names)), key=len, reverse=True)

        self.char_to_id = {name: idx for idx, name in enumerate(self.whitelist)}
        logger.info(f"📋 Loaded {len(self.whitelist)} operators from whitelist.")

    def _is_blacklisted(self, image_path_str: str) -> bool:
        """检查路径是否包含黑名单关键词"""
        # 统一转大写比较，忽略大小写差异
        path_upper = image_path_str.upper()
        return any(kw.upper() in path_upper for kw in self.config.blacklist_keywords)

    def _match_character_name(self, filename: str) -> Optional[str]:
        """核心匹配逻辑：从文件名中提取角色名"""
        for name in self.whitelist:
            if name in filename:
                return name
        return None

    def _get_smart_face_crop(self, img_pil: Image.Image) -> Image.Image:
        """
        AI 智能裁剪核心逻辑 (dghs-imgutils 版本)
        """
        w, h = img_pil.size
        face_box = None

        if HAS_IMGUTILS:
            # detect_faces 返回列表: [((x0, y0, x1, y1), label, score), ...]
            # 我们只需要检测人脸 (label通常不需要过滤，默认就是脸)
            try:
                detections = detect_faces(img_pil)

                if detections:
                    # 1. 排序：取置信度(score)最高的那张脸
                    # dghs的返回格式是 ((x0, y0, x1, y1), label, score)
                    best_face = max(detections, key=lambda x: x[2])
                    (fx1, fy1, fx2, fy2), _, score = best_face

                    # 只有置信度够高才信它 (比如 > 0.5)
                    if score > 0.5:
                        face_w = fx2 - fx1
                        face_h = fy2 - fy1

                        cx = fx1 + face_w / 2
                        cy = fy1 + face_h / 2

                        # 🌟 扩图倍数 (Zoom Out Factor)
                        # 2.2x - 2.5x 能包含头发和脖子
                        crop_span = max(face_w, face_h) * 2.0

                        # 限制最小尺寸，防止切太小放大后模糊
                        crop_span = max(crop_span, 512)

                        half_span = crop_span / 2

                        left = max(0, cx - half_span)
                        top = max(0, cy - half_span)
                        right = min(w, cx + half_span)
                        bottom = min(h, cy + half_span)

                        # 边界修正：尽量保持正方形
                        if right - left < crop_span:  # 宽不够
                            if left == 0:
                                right = min(w, crop_span)
                            else:
                                left = max(0, w - crop_span)
                        if bottom - top < crop_span:  # 高不够
                            if top == 0:
                                bottom = min(h, crop_span)
                            else:
                                top = max(0, h - crop_span)

                        face_box = (left, top, right, bottom)
            except Exception as e:
                logger.warning(f"Face detection failed: {e}")

        # Fallback: 如果没装库，或者没检测到脸，使用规则裁剪
        if face_box is None:
            crop_size = min(int(h * 0.45), w)
            center_x = w // 2
            left = max(0, center_x - crop_size // 2)
            top = int(h * 0.05)
            face_box = (left, top, left + crop_size, top + crop_size)

        return img_pil.crop(face_box)

    def _process_single_image(
        self, img_path: Path, output_dir: Path, char_name: str
    ) -> List[ImageMeta]:
        # ... (这部分逻辑与之前完全一致，保持 Letterbox 和 3种裁剪策略不变) ...
        # ... 为了节省篇幅，这里可以直接复制上一次回答中的 _process_single_image 代码 ...
        # ... 唯一的区别是 _process_single_image 内部调用的是 self._get_smart_face_crop ...

        label_id = self.char_to_id[char_name]
        results = []

        try:
            # 1. Load & Trim
            with Image.open(img_path) as img:
                img = img.convert("RGBA")
                bbox = img.getbbox()
                if bbox is None:
                    return []
                img_trimmed = img.crop(bbox)

            # 2. Composite White Background
            full_bg = Image.new("RGB", img_trimmed.size, (255, 255, 255))
            full_bg.paste(img_trimmed, mask=img_trimmed.split()[3])

            file_hash = hashlib.md5(str(img_path).encode("utf-8")).hexdigest()[:6]
            base_name = f"{label_id}_{char_name}_{file_hash}"
            w, h = full_bg.size

            # === A. Full Body ===
            scale = (self.config.target_pixel_area / (w * h)) ** 0.5
            if scale < 1.0:
                img_full = full_bg.resize(
                    (int(w * scale), int(h * scale)), Image.Resampling.LANCZOS
                )
            else:
                img_full = full_bg
            full_name = f"{base_name}_{CropType.FULL.value}.jpg"
            img_full.save(output_dir / full_name, quality=95)
            results.append(
                ImageMeta(
                    full_name, char_name, label_id, CropType.FULL.value, str(img_path)
                )
            )

            # === B. Face Crop (使用新版 dghs 逻辑) ===
            img_face_raw = self._get_smart_face_crop(full_bg)
            img_face = img_face_raw.resize(
                (self.config.face_crop_size, self.config.face_crop_size),
                Image.Resampling.LANCZOS,
            )
            face_name = f"{base_name}_{CropType.FACE.value}.jpg"
            img_face.save(output_dir / face_name, quality=95)
            results.append(
                ImageMeta(
                    face_name, char_name, label_id, CropType.FACE.value, str(img_path)
                )
            )

            # === C. Half Body (使用 Letterbox 逻辑) ===
            half_crop_h = int(h * 0.55)
            target_ar = 3 / 4
            current_ar = w / half_crop_h

            crop_w = w
            crop_h = half_crop_h

            if current_ar > target_ar:
                target_crop_w = int(crop_h * target_ar)
                left = (w - target_crop_w) // 2
                crop_w = target_crop_w
            else:
                left = 0

            img_half_raw = full_bg.crop((left, 0, left + crop_w, crop_h))

            # Letterbox Resize
            target_w, target_h = self.config.half_body_size
            ratio = min(target_w / crop_w, target_h / crop_h)
            new_w = int(crop_w * ratio)
            new_h = int(crop_h * ratio)
            img_half_resized = img_half_raw.resize(
                (new_w, new_h), Image.Resampling.LANCZOS
            )
            final_half = Image.new("RGB", (target_w, target_h), (255, 255, 255))
            final_half.paste(
                img_half_resized, ((target_w - new_w) // 2, (target_h - new_h) // 2)
            )

            half_name = f"{base_name}_{CropType.HALF.value}.jpg"
            final_half.save(output_dir / half_name, quality=95)
            results.append(
                ImageMeta(
                    half_name, char_name, label_id, CropType.HALF.value, str(img_path)
                )
            )

        except Exception as e:
            logger.error(f"❌ Failed processing {img_path}: {e}")

        return results

    def run(
        self,
        raw_root: Union[str, Path],
        output_root: Union[str, Path],
        num_workers: int = 8,
    ) -> None:
        """
        主执行入口 (并行版)
        num_workers: 并行数量，建议设置为 CPU 核心数 (你的 9700X 可以设为 8 或 16)
        """
        raw_path = Path(raw_root)
        out_path = Path(output_root)

        # 准备输出目录
        img_out_path = out_path / "images"
        img_out_path.mkdir(parents=True, exist_ok=True)

        if not self.whitelist:
            raise ValueError("Whitelist is empty! Please call load_whitelist() first.")

        # 1. Scanning (扫描阶段依然是很快的，单线程即可)
        logger.info(f"🔍 Scanning directory: {raw_path}...")
        tasks: List[Tuple[Path, str]] = []

        for root, _, files in os.walk(raw_path):
            if self._is_blacklisted(root):
                continue
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix not in self.config.allowed_extensions:
                    continue
                if self._is_blacklisted(file):
                    continue

                char_name = self._match_character_name(file)
                if char_name:
                    tasks.append((file_path, char_name))

        logger.info(
            f"🚀 Found {len(tasks)} valid images. Starting parallel processing with {num_workers} workers..."
        )

        # 2. Parallel Processing (并行处理)
        all_meta: List[ImageMeta] = []

        # 使用 ProcessPoolExecutor 启动多进程
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            # 注意：这里我们传递 self._process_single_image，Python 会自动序列化对象
            future_to_file = {
                executor.submit(self._process_single_image, p, img_out_path, name): p
                for p, name in tasks
            }

            # 使用 tqdm 显示进度，as_completed 会在任务完成时立刻返回
            for future in tqdm(
                as_completed(future_to_file), total=len(tasks), desc="Processing"
            ):
                try:
                    meta_list = future.result()
                    all_meta.extend(meta_list)
                except Exception as e:
                    file_p = future_to_file[future]
                    logger.error(f"❌ Worker failed processing {file_p}: {e}")

        # 3. Saving Metadata
        logger.info("💾 Saving metadata...")
        meta_dicts = [asdict(m) for m in all_meta]

        with open(out_path / "dataset.json", "w", encoding="utf-8") as f:
            json.dump(meta_dicts, f, indent=2, ensure_ascii=False)

        with open(out_path / "id_map.json", "w", encoding="utf-8") as f:
            json.dump(self.char_to_id, f, indent=2, ensure_ascii=False)

        logger.info(f"✨ Preprocessing complete. Processed {len(all_meta)} items.")
