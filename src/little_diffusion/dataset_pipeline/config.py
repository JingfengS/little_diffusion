import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Set, Dict

# ================= 数据结构定义 =================

class CropType(Enum):
    FULL = "full"   # 全身/全身缩放
    FACE = "face"   # 大头照
    HALF = "half"   # 半身/膝盖以上

@dataclass
class WeightConfig:
    """Masked Loss 的核心权重配置"""
    fg_weight: float = 1.0          # 主体人物 (Alpha > 200)
    bg_complex_weight: float = 0.1  # 复杂背景 (废墟、特效) -> 弱学习
    bg_pure_weight: float = 0.01    # 纯色/白背景 -> 几乎不学习

@dataclass
class PipelineConfig:
    """ETL 流水线全局配置"""
    # 路径配置
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    whitelist_path: str = "scripts/tools/operators.txt"
    
    # 图像参数
    target_pixel_area: int = 1024 * 1024  # 1MP (用于 Full 模式的缩放基准)
    face_crop_size: int = 768             # Face 模式固定尺寸
    half_body_size: Tuple[int, int] = (768, 1024) # Half 模式固定尺寸
    
    # 权重配置
    weights: WeightConfig = field(default_factory=WeightConfig)

    # 过滤配置
    blacklist_keywords: List[str] = field(
        default_factory=lambda: [
            "NPC", "IMG", "avg", "怪物", "敌方", "token", "trap", 
            "整合运动", "龙门士兵", "路人", "黑帮", "保镖", "小车"
        ]
    )
    allowed_extensions: Set[str] = field(
        default_factory=lambda: {".png", ".jpg", ".jpeg", ".PNG", ".JPG"}
    )

@dataclass
class ImageMeta:
    """数据元数据 (写入 dataset.json)"""
    file_path: str      # 相对路径 (images/xxx.jpg)
    mask_path: str      # 相对路径 (masks/xxx.png)
    character: str      # 角色名
    class_id: int       # 类别 ID
    type: str           # CropType
    original_path: str  # 原始文件路径 (用于溯源)