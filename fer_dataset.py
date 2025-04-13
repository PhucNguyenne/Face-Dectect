import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FERDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        # Giả sử cấu trúc: root_dir/0, root_dir/1, ..., root_dir/6
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for file in os.listdir(label_dir):
                    if file.endswith('.jpg') or file.endswith('.png'):
                        self.image_paths.append(os.path.join(label_dir, file))
                        self.labels.append(int(label))
                        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
