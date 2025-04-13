import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn
import torchvision.models as models

#########################
# Mapping nhãn
#########################
EMOTION_LABELS = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Sad", 5: "Surprise", 6: "Neutral"
}
AGE_LABELS = {0: "Under 20", 1: "20-39", 2: "40-59", 3: "60+"}
GENDER_LABELS = {0: "Female", 1: "Male"}

#########################
# Load mô hình
#########################
# Face Detector Model
from train_celeba import FaceDetector
def load_face_detector(weight_path="face_detector.pth"):
    model = FaceDetector()
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model

# UTKFace Model
from train_utkface import UTKFaceModel
def load_utk_model(weight_path="utkface_model.pth"):
    model = UTKFaceModel()
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model

# FER Model (sử dụng ResNet18 thay đổi fc cho 7 lớp cảm xúc)
def load_fer_model(weight_path="fer_model.pth"):
    fer_model = models.resnet18(pretrained=False)
    fer_model.fc = nn.Linear(fer_model.fc.in_features, 7)
    fer_model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    fer_model.eval()
    return fer_model

#########################
# Các transform tiền xử lý
#########################
def get_detector_transform(image_size=(224,224)):
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor()
    ])

def get_inference_transform(image_size=(224,224)):
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

det_transform = get_detector_transform()
inference_transform = get_inference_transform()

# Load mô hình (nạp trọng số từ file .pth)
face_detector = load_face_detector("face_detector.pth")
utk_model = load_utk_model("utkface_model.pth")
fer_model = load_fer_model("fer_model.pth")

#########################
# Hàm xử lý 1 khung hình (frame)
#########################
def process_frame(frame):
    # Chuyển frame từ BGR (OpenCV) sang RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    orig_w, orig_h = pil_img.size

    # 1. Phát hiện bounding box khuôn mặt
    det_input = det_transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        bbox_pred = face_detector(det_input)  # [1,4] trong không gian 224x224
    x, y, w, h = bbox_pred[0].numpy()
    # Scale về kích thước gốc
    scale_x = orig_w / 224.0
    scale_y = orig_h / 224.0
    x = x * scale_x
    y = y * scale_y
    w = w * scale_x
    h = h * scale_y
    x1 = int(x)
    x2 = int(x + w)
    if x2 < x1:  # Nếu width âm, đảo chỗ
        x1, x2 = x2, x1
    y1 = int(y)
    y2 = int(y + h)
    if y2 < y1:
        y1, y2 = y2, y1
    # Clamp bounding box trong kích thước ảnh gốc
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, orig_w)
    y2 = min(y2, orig_h)
    # Nếu bounding box không hợp lệ, sử dụng toàn bộ ảnh
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = 0, 0, orig_w, orig_h

    # 2. Crop khuôn mặt từ ảnh
    face_crop = pil_img.crop((x1, y1, x2, y2))
    
    # 3. Chuyển khuôn mặt crop về tensor cho inference
    face_tensor = inference_transform(face_crop).unsqueeze(0)
    
    # 4. Dự đoán tuổi, giới tính và cảm xúc
    with torch.no_grad():
        age_logits, gender_logits = utk_model(face_tensor)
        fer_out = fer_model(face_tensor)
    pred_age = age_logits.argmax(dim=1).item()    # 0..3
    pred_gender = gender_logits.argmax(dim=1).item()  # 0..1
    pred_emotion = fer_out.argmax(dim=1).item()       # 0..6
    
    age_label = AGE_LABELS[pred_age]
    gender_label = GENDER_LABELS[pred_gender]
    emotion_label = EMOTION_LABELS[pred_emotion]
    
    # 5. Vẽ bounding box và thông tin lên frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    label_text = f"{gender_label}, {age_label}, {emotion_label}"
    cv2.putText(frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0,255,0), 2)
    return frame

#########################
# Main: xử lý video từ camera
#########################
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera!")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        cv2.imshow("Camera Inference", processed_frame)
        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
