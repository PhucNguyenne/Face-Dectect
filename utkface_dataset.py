import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Lấy tất cả các file ảnh .jpg trong thư mục
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        # Tách nhãn từ tên file: age_gender_race_date.jpg
        try:
            age_str, gender_str, _, _ = img_name.split('_')
            age = int(age_str)
            gender = int(gender_str)  # 0: Female, 1: Male
        except Exception as e:
            age = 0
            gender = 0
        
        if self.transform:
            image = self.transform(image)
        return image, age, gender
