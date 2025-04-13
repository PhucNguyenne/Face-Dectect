import torch
import argparse
import os
import pandas as pd
from dataset import CelebADataset, get_transforms
from utils import load_partition_bbox
from train import FaceDetector  # Giả sử FaceDetector được định nghĩa trong train.py

def test_model(data_dir, batch_size=32):
    # Load partition: train_df, val_df, test_df (giả sử load_partition_bbox trả về 3 dataframe)
    _, _, test_df = load_partition_bbox(data_dir)
    print("Số mẫu trong test dataset:", len(test_df))
    
    # Tạo dataset cho tập test
    # Lưu ý: các transform 'val' phải giống với khi training
    test_dataset = CelebADataset(test_df, data_dir, transforms=get_transforms("val"))
    
    # DataLoader cho tập test, không cần shuffle
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # Khởi tạo mô hình và load trọng số đã lưu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceDetector().to(device)
    model.load_state_dict(torch.load("face_detector.pth", map_location=device))
    model.eval()
    
    predictions = []  # Danh sách lưu kết quả dự đoán (x, y, w, h)
    
    # Vòng lặp inference trên tập test
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            # Lưu kết quả từ batch (chuyển về numpy)
            predictions.extend(outputs.cpu().numpy())
    
    # Giả sử test_df có cột "image_id" để định danh ảnh, ta thêm các cột dự đoán vào DataFrame
    test_df = test_df.reset_index(drop=True)
    test_df["pred_x"] = [pred[0] for pred in predictions]
    test_df["pred_y"] = [pred[1] for pred in predictions]
    test_df["pred_w"] = [pred[2] for pred in predictions]
    test_df["pred_h"] = [pred[3] for pred in predictions]
    
    output_csv = "test_predictions.csv"
    test_df.to_csv(output_csv, index=False)
    print("Đã lưu dự đoán của test dataset vào file", output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test mô hình trên tập test của CelebA")
    parser.add_argument("--data-dir", type=str, required=True, help="Đường dẫn đến thư mục chứa dữ liệu CelebA")
    parser.add_argument("--batch-size", type=int, default=32, help="Kích thước batch cho việc test")
    args = parser.parse_args()
    
    test_model(args.data_dir, args.batch_size)
