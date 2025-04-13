import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Face_security.train_celeba import FaceDetector  # Giả sử FaceDetector được định nghĩa trong train.py
from dataset import get_transforms

# Thiết lập device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo mô hình và load trọng số
model = FaceDetector().to(device)
model.load_state_dict(torch.load("face_detector.pth", map_location=device))
model.eval()

# Đường dẫn đến ảnh ngoài tập dữ liệu
img_path = "./Test_img_out/*.jpg"  # Thay bằng đường dẫn ảnh của bạn

# Đọc ảnh và chuyển đổi sang RGB
image = Image.open(img_path).convert("RGB")

# Áp dụng transform giống như khi training (ví dụ: transform "val")
transform = get_transforms("val")
input_tensor = transform(image).unsqueeze(0).to(device)

# Thực hiện inference
with torch.no_grad():
    output = model(input_tensor)

# Lấy dự đoán bounding box (giả sử output có 4 giá trị: x, y, w, h)
bbox = output[0].cpu().numpy()
print("Predicted bounding box:", bbox)

# Vẽ bounding box lên ảnh
fig, ax = plt.subplots(1)
ax.imshow(image)
rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                         linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.title("Predicted Bounding Box")
plt.show()
