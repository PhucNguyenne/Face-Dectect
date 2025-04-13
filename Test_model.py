import os
import torch
import numpy as np
import pandas as pd
import json
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn
import torchvision.models as models

from train_utkface import UTKFaceModel

# Mapping nhãn
EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

AGE_LABELS = {
    0: "Under 20",
    1: "20-39",
    2: "40-59",
    3: "60+"
}

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
# Transform dùng cho inference (ảnh trong Test_IMG được giả định đã crop/căn chỉnh)
#########################
face_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def inference_on_single_image(image_path, utk_model, fer_model):
    # Load ảnh đã crop (ảnh được lưu trong Test_IMG)
    img = Image.open(image_path).convert("RGB")
    face_tensor = face_transform(img).unsqueeze(0)
    
    with torch.no_grad():
        age_logits, gender_logits = utk_model(face_tensor)
        fer_out = fer_model(face_tensor)
    
    pred_age_bin = age_logits.argmax(dim=1).item()
    pred_gender = gender_logits.argmax(dim=1).item()
    pred_emotion = fer_out.argmax(dim=1).item()
    
    return AGE_LABELS[pred_age_bin], GENDER_LABELS[pred_gender], EMOTION_LABELS[pred_emotion]

def main():
    test_img_dir = "Test_IMG"  # Thư mục chứa ảnh test đã được crop/căn chỉnh
    output_csv = "Test_IMG_predictions.csv"
    output_json = "Test_IMG_predictions.json"
    
    # Lấy danh sách các file ảnh trong Test_IMG (jpg, jpeg, png)
    image_files = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    predictions = []
    
    # Load mô hình UTKFace và FER một lần
    utk_model = load_utk_model("utkface_model.pth")
    fer_model = load_fer_model("fer_model.pth")
    
    for img_file in image_files:
        image_path = os.path.join(test_img_dir, img_file)
        try:
            age_label, gender_label, emotion_label = inference_on_single_image(image_path, utk_model, fer_model)
            predictions.append({
                "image_id": img_file,
                "age": age_label,
                "gender": gender_label,
                "emotion": emotion_label
            })
            print(f"{img_file}: {gender_label}, {age_label}, {emotion_label}")
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    # Lưu kết quả dự đoán vào CSV
    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)
    print(f"Đã lưu kết quả dự đoán vào file CSV: {output_csv}")
    
    # Lưu kết quả dự đoán vào JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
    print(f"Đã lưu kết quả dự đoán vào file JSON: {output_json}")

if __name__ == "__main__":
    main()
