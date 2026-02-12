import os
import json
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

class SquarePadResize:
    """
    智能缩放填充：
    1. 保持比例缩放，让长边 = target_size
    2. 短边用白色填充 (Pad) 到 target_size
    结果：一张不失真、不被裁剪的正方形图片
    """
    def __init__(self, target_size: int, fill_color: tuple = (255, 255, 255)):
        self.target_size = target_size
        self.fill_color = fill_color

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        
        # 1. 计算缩放比例 (基于长边)
        ratio = self.target_size / max(w, h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        # 2. 缩放
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 3. 创建正方形画布
        new_img = Image.new("RGB", (self.target_size, self.target_size), self.fill_color)
        
        # 4. 居中粘贴
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img
class JSONImageDataset(Dataset):
    """
    升级版数据集：读取 dataset.json 索引，支持 Label
    """
    def __init__(
        self,
        metadata_path: Union[str, Path],
        image_root: Union[str, Path],
        image_size: int = 1024,
    ):
        self.image_root = Path(image_root)
        self.image_size = image_size
        
        # 加载索引文件
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        logger.info(f"📚 Loaded dataset index with {len(self.metadata)} images.")

        # VAE 预处理: Resize (LANCZOS) -> CenterCrop -> Normalize
        # 注意：因为我们之前的预处理已经做过 Letterbox 和 Resize，
        # 这里的 CenterCrop 主要是为了防御性编程，防止有漏网之鱼
        self.transform = transforms.Compose([
            SquarePadResize(image_size, fill_color=(255, 255, 255)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        image_path = self.image_root / item['file_path']
        class_id = item['class_id']
        
        try:
            img = Image.open(image_path).convert("RGB")
            tensor = self.transform(img)
            # 返回: (Pixel_Tensor, Label_ID)
            return tensor, class_id
        except Exception as e:
            logger.error(f"❌ Corrupted image {image_path}: {e}")
            # 返回全0占位，并在 Label 设为 -1 (需要在 Collate 时过滤，或者简单点直接忽略错误)
            return torch.zeros(3, self.image_size, self.image_size), -1

class VAEProcessor:
    """
    VAE 处理引擎：负责将图片转为 Latents 并打包保存
    """
    def __init__(self,
                 model_name: str = "madebyollin/sdxl-vae-fp16-fix", # 👈 升级为 SDXL VAE
                 device: Optional[str] = None,
                 scaling_factor: float = 0.13025, # 👈 SDXL 的缩放因子是 0.13025
                 use_fp16: bool = True):
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaling_factor = scaling_factor
        
        # 5070 Ti 优先使用 BF16，如果不支持则回退到 FP16
        if use_fp16 and self.device != 'cpu':
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
                logger.info("🚀 Using BFloat16 for VAE processing (Ampere/Hopper Optimized)")
            else:
                self.dtype = torch.float16
                logger.info("🚀 Using Float16 for VAE processing")
        else:
            self.dtype = torch.float32

        logger.info(f"🚀 Loading VAE: {model_name}")
        try:
            self.vae = AutoencoderKL.from_pretrained(model_name, torch_dtype=self.dtype).to(self.device)
            self.vae.eval()
            self.vae.requires_grad_(False)
        except Exception as e:
            logger.error(f"❌ VAE Load Failed: {e}")
            raise e
    
    @torch.no_grad()
    def process_dataset(self, 
                        metadata_path: str, 
                        image_root: str, 
                        output_path: str, 
                        image_size: int = 512, 
                        batch_size: int = 4, 
                        num_workers: int = 4):
        """
        读取 dataset.json -> VAE Encode -> 保存 Latents + Labels
        """
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        dataset = JSONImageDataset(metadata_path, image_root, image_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        
        all_latents = []
        all_labels = []
        
        logger.info("📸 Starting VAE Encoding...")
        
        for batch_imgs, batch_labels in tqdm(dataloader, desc="Encoding"):
            # 1. 搬运图片到 GPU
            pixel_values = batch_imgs.to(self.device, dtype=self.dtype)
            
            # 2. VAE 编码
            # SDXL VAE 输出分布，采样并缩放
            dist = self.vae.encode(pixel_values).latent_dist
            latents = dist.sample() * self.scaling_factor
            
            # 3. 搬回 CPU (省显存)
            all_latents.append(latents.float().cpu()) # 存为 FP32 保证精度，训练时再转 BF16
            all_labels.append(batch_labels.long().cpu())
            
        # 4. 拼接大张量
        final_latents = torch.cat(all_latents, dim=0)
        final_labels = torch.cat(all_labels, dim=0)
        
        # 5. 打包保存
        payload = {
            "latents": final_latents,
            "labels": final_labels,
            "scaling_factor": self.scaling_factor,
            "image_size": image_size,
            "latent_size": final_latents.shape[-1]
        }
        
        torch.save(payload, output_path)
        
        logger.info(f"✅ Saved processed data to {output_path}")
        logger.info(f"📊 Latents Shape: {final_latents.shape}")
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