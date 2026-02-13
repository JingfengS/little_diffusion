import os
import logging
import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Optional
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL

from little_diffusion.preprocessing.dataset import JSONImageDataset

logger = logging.getLogger(__name__)

class VAEProcessor:
    def __init__(
        self,
        model_name: str = "madebyollin/sdxl-vae-fp16-fix",
        device: Optional[str] = None,
        scaling_factor: float = 0.13025,
        use_fp16: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaling_factor = scaling_factor
        
        # 自动选择最佳精度
        if use_fp16 and self.device != "cpu":
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        logger.info(f"🚀 Loading VAE: {model_name} ({self.dtype})")
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
        all_masks = [] 
        all_labels = []

        logger.info("📸 Starting VAE Encoding & Mask Downsampling...")

        for batch_imgs, batch_masks, batch_labels in tqdm(dataloader, desc="Encoding"):
            # Move to GPU
            pixel_values = batch_imgs.to(self.device, dtype=self.dtype)
            
            # Mask 不需要转 fp16，保持 float32 精度更好，或者跟 latent 保持一致也可以
            # 这里为了 avg_pool 计算准确，建议用 float32 计算，最后存的时候可以压缩
            batch_masks = batch_masks.to(self.device)

            # 1. VAE 编码 RGB 图像
            dist = self.vae.encode(pixel_values).latent_dist
            latents = dist.sample() * self.scaling_factor

            # 2. Mask 下采样！(1024x1024 -> 128x128)
            # 使用 Average Pooling 获得平滑边缘
            masks_downsampled = F.avg_pool2d(batch_masks, kernel_size=8, stride=8)

            # 3. 搬回 CPU
            all_latents.append(latents.float().cpu())
            all_masks.append(masks_downsampled.float().cpu())
            all_labels.append(batch_labels.long().cpu())

        final_latents = torch.cat(all_latents, dim=0)
        final_masks = torch.cat(all_masks, dim=0)
        final_labels = torch.cat(all_labels, dim=0)

        # 打包保存
        payload = {
            "latents": final_latents,
            "masks": final_masks,
            "labels": final_labels,
            "scaling_factor": self.scaling_factor,
            "image_size": image_size,
            "latent_size": final_latents.shape[-1],
        }

        torch.save(payload, output_path)

        logger.info(f"✅ Saved processed data to {output_path}")
        logger.info(f"📊 Latents Shape: {final_latents.shape}")
        logger.info(f"🎭 Masks Shape: {final_masks.shape}")
        logger.info(f"🏷️ Labels Shape: {final_labels.shape}")

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> List[Image.Image]:
        """解码工具 (用于 Sample 阶段)"""
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