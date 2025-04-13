import os
import shutil
import random

# Đường dẫn thư mục chứa ảnh gốc của CelebA
source_dir = "Data/img_align_celeba"  # Điều chỉnh đường dẫn nếu cần

# Thư mục đích để lưu ảnh cho test
dest_dir = "Test_IMG"

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(dest_dir, exist_ok=True)

# Lấy danh sách tất cả các file ảnh (jpg, jpeg, png)
image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Nếu có nhiều hơn 100 ảnh, chọn ngẫu nhiên 100 ảnh; nếu ít hơn, copy tất cả
selected_files = random.sample(image_files, 100) if len(image_files) >= 100 else image_files

# Copy các file ảnh đã chọn sang thư mục Test_IMG
for filename in selected_files:
    src_path = os.path.join(source_dir, filename)
    dest_path = os.path.join(dest_dir, filename)
    shutil.copy2(src_path, dest_path)

print(f"Đã copy {len(selected_files)} hình ảnh vào thư mục {dest_dir}.")
