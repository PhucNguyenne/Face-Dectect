# dataset.py
import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def get_transforms(mode="train"):
    """
    Định nghĩa các biến đổi (transforms) cho ảnh.
    
    Args:
        mode (str): "train" hoặc "val"
    
    Returns:
        transforms.Compose: Các biến đổi cho ảnh
    """
    if mode == "train":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class CelebADataset(Dataset):
    """
    Dataset cho tập dữ liệu CelebA.
    """
    def __init__(self, df, data_dir, transforms=None):
        """
        Args:
            df (pd.DataFrame): DataFrame chứa image_id, partition, và bounding box
            data_dir (str): Đường dẫn đến thư mục chứa dữ liệu (Data/)
            transforms: Các biến đổi áp dụng cho ảnh
        """
        self.df = df
        self.data_dir = data_dir
        self.transforms = transforms
        self.image_dir = os.path.join(data_dir, "img_align_celeba")
        print("Đường dẫn thư mục ảnh:", self.image_dir)
        if len(self.df) > 0:
            sample_img = self.df.iloc[0]["image_id"]
            sample_path = os.path.join(self.image_dir, sample_img)
            print("Đường dẫn ảnh mẫu:", sample_path)
            if not os.path.exists(sample_path):
                print(f"Cảnh báo: Không tìm thấy ảnh {sample_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Lấy thông tin từ DataFrame
        img_name = self.df.iloc[idx]["image_id"]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Đọc ảnh
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Lấy bounding box
        bbox = self.df.iloc[idx][["x_1", "y_1", "width", "height"]].values.astype(float)
        bbox = torch.tensor(bbox, dtype=torch.float32)

        # Áp dụng biến đổi
        if self.transforms:
            image = self.transforms(image)

        return image, bbox