from little_diffusion.processor import VAEProcessor
import torch

# 初始化处理器
processor = VAEProcessor()

# 加载之前保存的 Latents
latents = torch.load("images/train_latents/hutao_fp16.pt")

# 取第一张图对应的 Latent (1, 4, 88, 88)
single_latent = latents[0].unsqueeze(0) 

# 解码！
images = processor.decode(single_latent)

# 保存看看
images[0].save("images/save_images/restored_hutao_fp16.png")
print("✅ 还原成功！快去打开 restored_hutao.png 看看清晰度！")