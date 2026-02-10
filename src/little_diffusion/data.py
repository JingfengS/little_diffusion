from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os

class SingleImageDataset(Dataset):
    def __init__(self, image_path: str, size=704, num_samples=10000):
        """
        args: 
            image_path: path to the single image
            size: target size (704 by default)
            num_samples: number of samples to generate (10000 by default)
        """
        super().__init__()
        self.num_samples = num_samples
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        img = Image.open(image_path).convert("RGB")
        self.img_tensor = self.transform(img) # (3, 704, 704)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.img_tensor