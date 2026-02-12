import os
import torch
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """
    工业级图片数据集：
    - 递归搜索
    - 自动过滤非图片文件
    - 统一预处理流水线
    - 错误文件自动跳过 (返回空)
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        image_size: int = 512,
        ext: List[str] = [".jpg", ".jpeg", ".png", ".webp", ".bmp"],
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.files = sorted(
            [p for p in self.root_dir.rglob("*") if p.suffix.lower() in ext]
        )
        if len(self.files) == 0:
            logger.warning(f"⚠️ No images found in {root_dir}")
        # VAE 标准预处理: Resize -> Crop -> Normalize [-1, 1]
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.LANCZOS
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img)
        except Exception as e:
            logger.error(f"❌ Corrupted image {path}: {e}")
            # 返回 None，要在 collate_fn 里处理，或者这里返回全黑图
            # 为了简单起见，我们返回一个标记，让 DataLoader 过滤（需要自定义 collate）
            # 这里简单返回全 0 Tensor 占位，避免崩溃
            return torch.zeros(3, self.image_size, self.image_size)


class VAEProcessor:
    """
    VAE Engine for reuse
    """

    def __init__(self,
                 model_name: str = "stabilityai/sd-vae-ft-mse",
                 device: Optional[str] = None,
                 scaling_factor: float = 0.18215,
                 use_fp16: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaling_factor = scaling_factor
        self.use_fp16 = use_fp16 and (self.device != 'cpu')
        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        logger.info(f"🚀 Loading VAE: {model_name} (FP16: {self.use_fp16})")
        try:
            self.vae = AutoencoderKL.from_pretrained(model_name, torch_dtype=self.dtype).to(self.device)
            self.vae.eval()
            self.vae.requires_grad_(False) # 冻结权重，节省显存

            # self.vae.enable_tiling()
        except Exception as e:
            logger.error(f"❌ VAE Load Failed: {e}")
            raise e
    
    @torch.no_grad()
    def process_folder(self, input_dir: str, output_path: str, image_size: int = 512, batch_size: int = 4, num_workers: int = 4):
        """
        Process all images in a folder and save the latent representations.
        Save to .pt files
        """
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        dataset = ImageDataset(input_dir, image_size)
        if len(dataset) == 0:
            logger.warning(f"⚠️ No images found in {input_dir}")
            return None
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        
        all_latents = []
        logger.info(f"📸 Processing {len(dataset)} images from {input_dir}...")
        for batch in tqdm(dataloader, desc="Encoding"):
            pixel_values = batch.to(self.device, dtype=self.dtype)
            
            # Encode -> Sample -> Scale
            dist = self.vae.encode(pixel_values).latent_dist
            latents = dist.sample() * self.scaling_factor
            
            # 立即转回 CPU 释放显存
            all_latents.append(latents.float().cpu())
            
        # 拼接并保存
        final_tensor = torch.cat(all_latents, dim=0)
        torch.save(final_tensor, output_path)
        
        logger.info(f"✅ Saved latents to {output_path}")
        logger.info(f"📊 Shape: {final_tensor.shape} (N, 4, {image_size//8}, {image_size//8})")
        return final_tensor
    
    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> List[Image.Image]:
        """
        解码 Latent Tensor 回 PIL Images
        args:
            latents: (B, 4, H, W) Tensor
        returns:
            List of PIL Images
        """
        latents = latents.to(self.device, dtype=self.dtype)
        latents = latents / self.scaling_factor
        
        # VAE decoder
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        
        # Convert to PIL Images
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        
        output_images = []
        for i in range(image.shape[0]):
            img_np = (image[i] * 255).round().astype(np.uint8)
            output_images.append(Image.fromarray(img_np))
        
        return output_images