# process_image.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn
import torchvision.models as models

# Import mô hình
from train_celeba import FaceDetector
from train_utkface import UTKFaceModel

# Mapping nhãn
EMOTION_LABELS = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Sad", 5: "Surprise", 6: "Neutral"
}
AGE_LABELS = {
    0: "Under 20", 1: "20-39", 2: "40-59", 3: "60+"
}
GENDER_LABELS = {
    0: "Female", 1: "Male"
}

def load_face_detector(weight_path="face_detector.pth"):
    model = FaceDetector()
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model

def load_utk_model(weight_path="utkface_model.pth"):
    model = UTKFaceModel()
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model

def load_fer_model(weight_path="fer_model.pth"):
    fer_model = models.resnet18(pretrained=False)
    fer_model.fc = nn.Linear(fer_model.fc.in_features, 7)  # 7 lớp cảm xúc
    fer_model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    fer_model.eval()
    return fer_model

# Transform dùng cho khuôn mặt đã crop
face_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def process_image(image_path, output_path):
    # 1. Load ảnh
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    
    # 2. Phát hiện bounding box bằng FaceDetector
    face_detector = load_face_detector("face_detector.pth")
    transform_detector = T.Compose([
        T.Resize((224,224)),
        T.ToTensor()
    ])
    det_input = transform_detector(img).unsqueeze(0)
    with torch.no_grad():
        bbox_pred = face_detector(det_input)  # [1,4] trong không gian 224x224
    x, y, w, h = bbox_pred[0].numpy()
    # Debug: in ra giá trị raw bbox
    # print("Raw bbox (224x224 space):", x, y, w, h)
    
    orig_w, orig_h = img.size
    scale_x = orig_w / 224.0
    scale_y = orig_h / 224.0
    x = x * scale_x
    y = y * scale_y
    w = w * scale_x
    h = h * scale_y
    # print("Scaled bbox:", x, y, w, h)
    
    x1 = int(x)
    x2 = int(x + w)
    if x2 < x1:
        x1, x2 = x2, x1
    y1 = int(y)
    y2 = int(y + h)
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, orig_w)
    y2 = min(y2, orig_h)
    
    # Nếu bounding box không hợp lệ, sử dụng toàn bộ ảnh
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = 0, 0, orig_w, orig_h

    # 3. Crop khuôn mặt
    face_crop = img.crop((x1, y1, x2, y2))
    
    # 4. Dự đoán tuổi và giới tính với UTKFace
    utk_model = load_utk_model("utkface_model.pth")
    face_tensor = face_transform(face_crop).unsqueeze(0)
    with torch.no_grad():
        age_logits, gender_logits = utk_model(face_tensor)
    pred_age_bin = age_logits.argmax(dim=1).item()
    pred_gender = gender_logits.argmax(dim=1).item()
    age_label = AGE_LABELS[pred_age_bin]
    gender_label = GENDER_LABELS[pred_gender]
    
    # 5. Dự đoán cảm xúc với FER
    fer_model = load_fer_model("fer_model.pth")
    with torch.no_grad():
        fer_out = fer_model(face_tensor)
    pred_emotion = fer_out.argmax(dim=1).item()
    emotion_label = EMOTION_LABELS[pred_emotion]
    
    # 6. Vẽ bounding box và overlay thông tin dự đoán
    fig, ax = plt.subplots(1)
    ax.imshow(img_np)
    rect = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1),
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    info_text = f"{gender_label}, {age_label}, {emotion_label}"
    ax.text(x1, y1-10, info_text, color='red', fontsize=12, backgroundcolor='white')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    
    return output_path
