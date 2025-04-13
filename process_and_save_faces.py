import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN
import torchvision.transforms as T

def process_and_save_faces(image_path, output_dir="processed_img", image_size=(224,224)):
    """
    Phát hiện và căn chỉnh khuôn mặt trong ảnh bằng MTCNN, sau đó lưu từng khuôn mặt đã crop vào thư mục output_dir.
    
    Args:
        image_path (str): Đường dẫn đến ảnh đầu vào.
        output_dir (str): Thư mục để lưu ảnh đã xử lý.
        image_size (tuple): Kích thước mong muốn sau khi crop (chỉ dùng cho bước resize, tuy nhiên MTCNN cần một số nguyên cho image_size).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ảnh và chuyển sang RGB
    img = Image.open(image_path).convert("RGB")
    
    # Khởi tạo MTCNN để phát hiện khuôn mặt
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # MTCNN yêu cầu image_size là số nguyên, ví dụ 224
    output_size = 224
    mtcnn = MTCNN(image_size=output_size, margin=20, post_process=True, device=device)
    
    # Phát hiện và crop khuôn mặt
    faces = mtcnn(img)
    
    if faces is None:
        print("Không phát hiện được khuôn mặt nào!")
        return
    
    # Nếu chỉ có 1 khuôn mặt, đảm bảo faces là list
    if isinstance(faces, torch.Tensor):
        faces = [faces]
    
    # Chuyển đổi từ tensor về ảnh gốc mà không áp dụng hiệu ứng bổ sung
    to_pil = T.ToPILImage()
    for i, face in enumerate(faces):
        face_img = to_pil(face)
        output_path = os.path.join(output_dir, f"face_{i+1}.jpg")
        face_img.save(output_path)
        print(f"Đã lưu khuôn mặt {i+1} tại: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Xử lý và lưu ảnh khuôn mặt đã căn chỉnh từ ảnh đầu vào")
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn đến ảnh đầu vào")
    parser.add_argument("--output_dir", type=str, default="process_img", help="Thư mục lưu ảnh đã xử lý")
    args = parser.parse_args()
    
    process_and_save_faces(args.image, args.output_dir)
