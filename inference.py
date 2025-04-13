# src/inference.py
import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms
import torch.nn as nn

def load_model(model_path):
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def predict_face(model, image):
    """
    :param model: mô hình PyTorch
    :param image: ảnh BGR (OpenCV) hoặc RGB (nếu đã chuyển)
    :return: class_id, confidence
    """
    # Chuyển sang RGB nếu cần
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Chuẩn bị transform
    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    input_tensor = t(image).unsqueeze(0)  # shape: (1, 3, 224, 224)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        class_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0, class_id].item()

    return class_id, confidence

if __name__ == "__main__":
    model_path = "face_recog_resnet50.pth"
    if not os.path.exists(model_path):
        print("Model file not found. Please train the model first.")
    else:
        model = load_model(model_path)
        print("Model loaded.")

        # Ví dụ chạy inference trên 1 ảnh
        img_path = "../data/img_align_celeba/000001.jpg"
        image = cv2.imread(img_path)
        class_id, conf = predict_face(model, image)
        print(f"Predicted class: {class_id}, confidence: {conf:.4f}")
