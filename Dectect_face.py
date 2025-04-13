import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms as T

#########################
# 1) UTKFace Model
#########################
from train_utkface import UTKFaceModel

#########################
# 2) FER Model
#########################
import torch.nn as nn
import torchvision.models as models

# Mapping nhãn cảm xúc FER (7 lớp)
EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Mapping nhóm tuổi (4 nhóm)
AGE_LABELS = {
    0: "Under 20",
    1: "20-39",
    2: "40-59",
    3: "60+"
}

# Mapping giới tính
GENDER_LABELS = {
    0: "Female",
    1: "Male"
}

def load_utk_model(weight_path="utkface_model.pth"):
    model = UTKFaceModel()
    state_dict = torch.load(weight_path, map_location="cpu")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print("Error loading state_dict normally:", e)
        print("Trying with strict=False. Vui lòng fine-tune lại mô hình sau khi load!")
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def load_fer_model(weight_path="fer_model.pth"):
    fer_model = models.resnet18(pretrained=False)
    fer_model.fc = nn.Linear(fer_model.fc.in_features, 7)  # 7 lớp cảm xúc
    fer_model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    fer_model.eval()
    return fer_model

#########################
# Transform dùng cho inference (cho UTKFace & FER)
#########################
face_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

#########################
# Transform dùng cho detection (cho MTCNN) – không Normalize
#########################
detector_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])

#########################
# Hàm xử lý ảnh bằng MTCNN và lưu vào thư mục "processed_img"
#########################
def process_and_save_face(image_path, output_dir="processed_img", output_size=224, margin=20):
    """
    Sử dụng MTCNN để phát hiện, căn chỉnh và crop khuôn mặt từ ảnh đầu vào.
    Lưu khuôn mặt đầu tiên vào output_dir và trả về đường dẫn ảnh đó.
    """
    from facenet_pytorch import MTCNN
    os.makedirs(output_dir, exist_ok=True)
    
    img = Image.open(image_path).convert("RGB")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(image_size=output_size, margin=margin, post_process=True, device=device)
    
    faces = mtcnn(img)
    if faces is None:
        print("Không phát hiện được khuôn mặt nào!")
        return None
    if isinstance(faces, torch.Tensor):
        faces = [faces]
    
    # Chuyển tensor về PIL Image (không áp dụng Normalize)
    to_pil = T.ToPILImage()
    face_img = to_pil(faces[0])
    
    # Kiểm tra kích thước crop để đảm bảo khuôn mặt đủ lớn
    if face_img.size[0] < 50 or face_img.size[1] < 50:
        print("Khuôn mặt crop được quá nhỏ, có thể crop sai.")
        return None
    
    output_path = os.path.join(output_dir, "face_1.jpg")
    face_img.save(output_path)
    print(f"Đã lưu khuôn mặt tại: {output_path}")
    return output_path

#########################
# Pipeline inference
#########################
def inference_on_image(image_path):
    """
    Nếu ảnh đầu vào chưa nằm trong thư mục "processed_img", tự động xử lý ảnh bằng MTCNN để crop khuôn mặt.
    Sau đó, dùng ảnh đã crop cho inference với mô hình UTKFace và FER.
    """
    # Nếu ảnh không nằm trong folder "processed_img", tự động xử lý
    if "processed_img" not in os.path.abspath(image_path):
        print("Ảnh chưa được xử lý, chạy pipeline xử lý ảnh...")
        processed_path = process_and_save_face(image_path, output_dir="processed_img")
        if processed_path is None:
            print("Crop khuôn mặt thất bại.")
            return
        image_path = processed_path

    # Load ảnh đã crop
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    face_crop = img  # Ảnh đã được crop/căn chỉnh
    
    # Chuẩn bị tensor cho inference
    face_tensor = face_transform(face_crop).unsqueeze(0)
    
    # Dự đoán với mô hình UTKFace
    utk_model = load_utk_model("utkface_model.pth")
    with torch.no_grad():
        age_logits, gender_logits = utk_model(face_tensor)
    pred_age_bin = age_logits.argmax(dim=1).item()
    pred_gender = gender_logits.argmax(dim=1).item()
    age_label = AGE_LABELS[pred_age_bin]
    gender_label = GENDER_LABELS[pred_gender]
    
    # Dự đoán với mô hình FER
    fer_model = load_fer_model("fer_model.pth")
    with torch.no_grad():
        fer_out = fer_model(face_tensor)
    pred_emotion = fer_out.argmax(dim=1).item()
    emotion_label = EMOTION_LABELS[pred_emotion]
    
    # Hiển thị ảnh đã crop và kết quả dự đoán
    plt.figure()
    plt.imshow(img_np)
    plt.title(f"{gender_label}, {age_label}, {emotion_label}")
    plt.axis("off")
    plt.show()
    
    print("Dự đoán:")
    print(" - Giới tính:", gender_label)
    print(" - Nhóm tuổi:", age_label)
    print(" - Cảm xúc:", emotion_label)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference với ảnh đã được xử lý tự động bằng MTCNN")
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn đến ảnh đầu vào")
    args = parser.parse_args()
    
    inference_on_image(args.image)
