import torch
import matplotlib.pyplot as plt
from little_diffusion.models import BabyUNet
from little_diffusion.core import ODE, EulerSimulator
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = "checkpoints/hutao_64.pth"
save_path = Path("images/save_images")
save_path.mkdir(exist_ok=True)
IMG_SIZE = 64

class NeuralODE(ODE):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
    
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(xt, t)

def main():
    print(f"Using device: {device}")
    model = BabyUNet(in_channels=3, out_channels=3, dim=128).to(device)
    try:
        state_dict = torch.load(ckpt_path, map_location=device)
        # 👇【新增】修复 torch.compile 带来的前缀问题
        new_state_dict = {}
        for k, v in state_dict.items():
            # 如果键名以 _orig_mod. 开头，就把前缀去掉
            if k.startswith("_orig_mod."):
                new_state_dict[k[10:]] = v 
            else:
                new_state_dict[k] = v
        
        # 使用修复后的字典加载
        model.load_state_dict(new_state_dict)
        print("Successfully loaded checkpoint")
    except FileNotFoundError:
        print(f"Checkpoint not found at {ckpt_path}")
        return
    except RuntimeError as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()

    ode = NeuralODE(model)
    simulator = EulerSimulator(ode)

    x0 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

    steps = 100
    ts = torch.linspace(0, 1, steps, dtype=torch.float32).to(device).view(1, -1, 1) # (1, steps, 1)

    print("Generating image ...")

    with torch.no_grad():
        trajectory = simulator.simulate_with_trajectory(x0, ts)
        x_final = trajectory[0, -1]
    
    img_tensor = (x_final + 1) / 2
    img_tensor = img_tensor.clamp(0, 1)

    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(img_np)
    plt.axis('off')
    plt.title("Generated Image via Flow Matching")
    plt.show()

    plt.imsave(save_path / "generated_hutao.png", img_np)
    print("Generated image saved to generated_hutao.png")
    
if __name__ == "__main__":
    main()