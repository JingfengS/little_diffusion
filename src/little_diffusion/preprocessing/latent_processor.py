import os
import json
import torch
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Tuple
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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


class JSONImageDataset(Dataset):
    """
    升级版数据集：读取 Image 和对应的 Weight Mask
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

        # 假设 mask 与 image 同名，但后缀是 .png，且存在于 masks 文件夹下
        # 这里需要与你下一步的 vision_processor 产出对齐
        mask_path = (
            self.image_root.parent
            / "masks"
            / Path(item["file_path"]).with_suffix(".png").name
        )
        class_id = item["class_id"]

        try:
            img = Image.open(image_path).convert("RGB")

            # 如果 Mask 不存在 (兼容老数据)，就建一个全为 1.0 的 Dummy Mask
            if mask_path.exists():
                mask = Image.open(mask_path).convert("L")
            else:
                mask = Image.new("L", img.size, 255)

            # 同步 Resize 和 Pad
            img_padded, mask_padded = self.smart_resize(img, mask)

            # 转 Tensor (ToTensor 会自动把 0~255 转为 0.0~1.0)
            img_tensor = self.normalize(self.to_tensor(img_padded))
            mask_tensor = self.to_tensor(mask_padded)  # 形状: (1, 1024, 1024)

            return img_tensor, mask_tensor, class_id

        except Exception as e:
            logger.error(f"❌ Corrupted image {image_path}: {e}")
            return (
                torch.zeros(3, self.image_size, self.image_size),
                torch.zeros(1, self.image_size, self.image_size),
                -1,
            )


class VAEProcessor:
    # __init__ 保持不变，略 ...
    def __init__(
        self,
        model_name: str = "madebyollin/sdxl-vae-fp16-fix",
        device: Optional[str] = None,
        scaling_factor: float = 0.13025,
        use_fp16: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaling_factor = scaling_factor
        if use_fp16 and self.device != "cpu":
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        logger.info(f"🚀 Loading VAE: {model_name}")
        try:
            self.vae = AutoencoderKL.from_pretrained(
                model_name, torch_dtype=self.dtype
            ).to(self.device)
            self.vae.eval()
            self.vae.requires_grad_(False)
        except Exception as e:
            logger.error(f"❌ VAE Load Failed: {e}")
            raise e

    @torch.no_grad()
    def process_dataset(
        self,
        metadata_path: str,
        image_root: str,
        output_path: str,
        image_size: int = 1024,
        batch_size: int = 4,
        num_workers: int = 4,
    ):
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        dataset = JSONImageDataset(metadata_path, image_root, image_size)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        all_latents = []
        all_masks = []  # 👈 新增：存储处理好的 Mask
        all_labels = []

        logger.info("📸 Starting VAE Encoding & Mask Downsampling...")

        for batch_imgs, batch_masks, batch_labels in tqdm(dataloader, desc="Encoding"):
            pixel_values = batch_imgs.to(self.device, dtype=self.dtype)

            # 1. VAE 编码 RGB 图像
            dist = self.vae.encode(pixel_values).latent_dist
            latents = dist.sample() * self.scaling_factor

            # 2. Mask 下采样！(1024x1024 -> 128x128)
            # 使用 Average Pooling，这样边缘的权重会平滑过渡
            masks_downsampled = F.avg_pool2d(batch_masks, kernel_size=8, stride=8)

            # 3. 搬回 CPU
            all_latents.append(latents.float().cpu())
            all_masks.append(masks_downsampled.float().cpu())  # 👈 收集 Mask
            all_labels.append(batch_labels.long().cpu())

        final_latents = torch.cat(all_latents, dim=0)
        final_masks = torch.cat(all_masks, dim=0)
        final_labels = torch.cat(all_labels, dim=0)

        # 打包保存
        payload = {
            "latents": final_latents,
            "masks": final_masks,  # 👈 新增：Mask 一并打包！
            "labels": final_labels,
            "scaling_factor": self.scaling_factor,
            "image_size": image_size,
            "latent_size": final_latents.shape[-1],
        }

        torch.save(payload, output_path)

        logger.info(f"✅ Saved processed data to {output_path}")
        logger.info(f"📊 Latents Shape: {final_latents.shape}")
        logger.info(f"🎭 Masks Shape: {final_masks.shape}")  # 应该是 (N, 1, 128, 128)
        logger.info(f"🏷️ Labels Shape: {final_labels.shape}")

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> List[Image.Image]:
        """
        解码工具 (用于 Sample 阶段)
        """
        latents = latents.to(self.device, dtype=self.dtype)
        latents = latents / self.scaling_factor

        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)

        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        output_images = []
        for i in range(image.shape[0]):
            img_np = (image[i] * 255).round().astype(np.uint8)
            output_images.append(Image.fromarray(img_np))

        return output_images
