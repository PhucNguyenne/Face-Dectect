# transform_utils.py
import os
import torch
import torchvision.transforms as T
from PIL import Image
from facenet_pytorch import MTCNN

def get_detector_transform(image_size=(224, 224)):
    """
    Pipeline transform dành cho face detection.
    Resize ảnh về kích thước image_size và chuyển thành tensor.
    Dùng khi đưa ảnh vào mô hình phát hiện khuôn mặt.
    """
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor()
    ])

def get_inference_transform(image_size=(224, 224)):
    """
    Pipeline transform dành cho inference trên khuôn mặt đã crop.
    Resize ảnh, RandomHorizontalFlip, chuyển thành tensor và normalize theo chuẩn ImageNet.
    Dùng khi đưa khuôn mặt đã crop vào các mô hình phân loại (UTKFace, FER).
    """
    return T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

def get_camera_transform(image_size=(224, 224)):
    """
    Pipeline transform cho ảnh từ camera.
    Áp dụng ColorJitter để điều chỉnh ánh sáng, sau đó resize, chuyển thành tensor và normalize.
    """
    return T.Compose([
        T.Resize(image_size),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

def get_inference_transform_no_norm(image_size=(224, 224)):
    """
    Pipeline transform cho inference trên khuôn mặt đã crop, KHÔNG áp dụng Normalize.
    Dùng khi bạn muốn lưu ảnh gốc (không bị thay đổi màu sắc).
    """
    return T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor()
    ])

def process_and_save_faces(image_path, output_dir="processed_img", output_size=224):
    """
    Sử dụng MTCNN để phát hiện, căn chỉnh và crop khuôn mặt từ ảnh đầu vào,
    sau đó lưu từng khuôn mặt đã crop vào thư mục output_dir.
    
    Args:
        image_path (str): Đường dẫn đến ảnh đầu vào.
        output_dir (str): Thư mục lưu ảnh đã xử lý.
        output_size (int): Kích thước đầu ra của khuôn mặt (mỗi khuôn mặt được resize về kích thước này).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ảnh và chuyển sang RGB
    img = Image.open(image_path).convert("RGB")
    
    # Khởi tạo MTCNN, lưu ý image_size ở đây là một số nguyên
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(image_size=output_size, margin=20, post_process=True, device=device)
    
    # Phát hiện và crop khuôn mặt
    faces = mtcnn(img)
    if faces is None:
        print("Không phát hiện được khuôn mặt nào!")
        return
    
    if isinstance(faces, torch.Tensor):
        faces = [faces]
    
    # Sử dụng ToPILImage để chuyển tensor về PIL Image (không có Normalize)
    to_pil = T.ToPILImage()
    saved_paths = []
    for i, face in enumerate(faces):
        face_img = to_pil(face)
        output_path = os.path.join(output_dir, f"face_{i+1}.jpg")
        face_img.save(output_path)
        saved_paths.append(output_path)
        print(f"Đã lưu khuôn mặt {i+1} tại: {output_path}")
    return saved_paths

# Ví dụ sử dụng:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Xử lý và lưu ảnh khuôn mặt đã căn chỉnh từ ảnh đầu vào")
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn đến ảnh đầu vào")
    parser.add_argument("--output_dir", type=str, default="processed_img", help="Thư mục lưu ảnh đã xử lý")
    args = parser.parse_args()
    
    process_and_save_faces(args.image, args.output_dir)
