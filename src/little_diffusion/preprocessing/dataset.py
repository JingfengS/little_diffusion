import json
import logging
import torch
from pathlib import Path
from typing import Union
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from little_diffusion.preprocessing.utils import SquarePadResize

logger = logging.getLogger(__name__)

class JSONImageDataset(Dataset):
    """
    读取 Image 和对应的 Weight Mask
    """

    def __init__(
        self,
        metadata_path: Union[str, Path],
        image_root: Union[str, Path],
        image_size: int = 1024,
    ):
        self.image_root = Path(image_root)
        self.image_size = image_size

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        logger.info(f"📚 Loaded dataset index with {len(self.metadata)} items.")

        self.smart_resize = SquarePadResize(
            image_size, img_fill=(255, 255, 255), mask_fill=0.01
        )
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5], [0.5])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        image_path = self.image_root / item["file_path"]

        # 自动推断 Mask 路径：假设在同级目录 ../masks/ 下
        mask_path = (
            self.image_root.parent
            / "masks"
            / Path(item["file_path"]).with_suffix(".png").name
        )
        class_id = item["class_id"]

        try:
            img = Image.open(image_path).convert("RGB")

            # 如果 Mask 存在则读取，否则创建全白(权重1.0) Mask
            if mask_path.exists():
                mask = Image.open(mask_path).convert("L")
            else:
                mask = Image.new("L", img.size, 255)

            # 同步 Resize 和 Pad
            img_padded, mask_padded = self.smart_resize(img, mask)

            # 转 Tensor
            # ToTensor 会自动把 0~255 转为 0.0~1.0
            img_tensor = self.normalize(self.to_tensor(img_padded))
            mask_tensor = self.to_tensor(mask_padded) 

            return img_tensor, mask_tensor, class_id

        except Exception as e:
            logger.error(f"❌ Corrupted image {image_path}: {e}")
            # 返回全零 tensor 以防崩坏
            return (
                torch.zeros(3, self.image_size, self.image_size),
                torch.zeros(1, self.image_size, self.image_size),
                -1,
            )