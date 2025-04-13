import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from dataset import CelebADataset, get_transforms
from utils_celeba import load_partition_bbox
from tqdm import tqdm  # Thêm tqdm để có progress bar
import matplotlib.pyplot as plt

class FaceDetector(nn.Module):
    def __init__(self):
        super(FaceDetector, self).__init__()
        # Sử dụng ResNet18 làm backbone với trọng số từ ImageNet
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")  
        # Thêm Dropout và thay đổi lớp fully connected cuối cùng để dự đoán 4 giá trị (x, y, width, height)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 4)



    def forward(self, x):
        return self.backbone(x)

def compute_iou(boxes_pred, boxes_gt):
    """
    Tính IoU giữa bounding boxes dự đoán và ground truth.
    Các boxes có định dạng [x, y, width, height].
    """
    # Chuyển sang định dạng [x1, y1, x2, y2]
    boxes_pred_x1 = boxes_pred[:, 0]
    boxes_pred_y1 = boxes_pred[:, 1]
    boxes_pred_x2 = boxes_pred[:, 0] + boxes_pred[:, 2]
    boxes_pred_y2 = boxes_pred[:, 1] + boxes_pred[:, 3]

    boxes_gt_x1 = boxes_gt[:, 0]
    boxes_gt_y1 = boxes_gt[:, 1]
    boxes_gt_x2 = boxes_gt[:, 0] + boxes_gt[:, 2]
    boxes_gt_y2 = boxes_gt[:, 1] + boxes_gt[:, 3]

    inter_x1 = torch.max(boxes_pred_x1, boxes_gt_x1)
    inter_y1 = torch.max(boxes_pred_y1, boxes_gt_y1)
    inter_x2 = torch.min(boxes_pred_x2, boxes_gt_x2)
    inter_y2 = torch.min(boxes_pred_y2, boxes_gt_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    area_pred = (boxes_pred_x2 - boxes_pred_x1) * (boxes_pred_y2 - boxes_pred_y1)
    area_gt = (boxes_gt_x2 - boxes_gt_x1) * (boxes_gt_y2 - boxes_gt_y1)
    union_area = area_pred + area_gt - inter_area + 1e-6
    iou = inter_area / union_area
    return iou

def train_model(data_dir, batch_size=32, num_epochs=2, lr=0.001):
    """
    Huấn luyện mô hình và vẽ biểu đồ Loss và Accuracy (IoU).
    
    Args:
        data_dir (str): Đường dẫn đến thư mục chứa dữ liệu
        batch_size (int): Kích thước batch
        num_epochs (int): Số epoch huấn luyện
        lr (float): Tốc độ học
    """
    # Tải dữ liệu
    train_df, val_df, _ = load_partition_bbox(data_dir)
    print("Số mẫu trong train_df:", len(train_df))
    print("Số mẫu trong val_df:", len(val_df))

    # Tạo dataset
    train_dataset = CelebADataset(train_df, data_dir, transforms=get_transforms("train"))
    val_dataset = CelebADataset(val_df, data_dir, transforms=get_transforms("val"))
    print("Số mẫu trong train_dataset:", len(train_dataset))
    print("Số mẫu trong val_dataset:", len(val_dataset))

    # Tạo DataLoader (nâng cao num_workers nếu hệ thống cho phép)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=32)

    # Khởi tạo mô hình, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceDetector().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Scheduler giảm learning rate khi validation loss không giảm
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Khởi tạo biến lưu trữ loss và accuracy theo epoch
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Early stopping parameters
    best_loss = float('inf')
    trigger_times = 0
    patience = 10

    start_time = time.time()

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        train_batches = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] (Train)", leave=False)
        for images, bboxes in train_loader_tqdm:
            images, bboxes = images.to(device), bboxes.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, bboxes)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_batches += 1

            # Tính IoU cho batch và tính accuracy (đếm mẫu có IoU >= 0.5)
            iou = compute_iou(outputs, bboxes)
            batch_accuracy = (iou >= 0.5).float().mean()
            running_accuracy += batch_accuracy.item()

            train_loader_tqdm.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_accuracy.item():.4f}")

        avg_train_loss = running_loss / train_batches
        avg_train_accuracy = running_accuracy / train_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)

        # Validation
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_batches = 0
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] (Val)", leave=False)
        with torch.no_grad():
            for images, bboxes in val_loader_tqdm:
                images, bboxes = images.to(device), bboxes.to(device)
                outputs = model(images)
                loss = criterion(outputs, bboxes)
                val_loss += loss.item()
                val_batches += 1

                iou = compute_iou(outputs, bboxes)
                batch_accuracy = (iou >= 0.5).float().mean()
                val_accuracy += batch_accuracy.item()

                val_loader_tqdm.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_accuracy.item():.4f}")

        avg_val_loss = val_loss / val_batches
        avg_val_accuracy = val_accuracy / val_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}")
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered")
                break

    total_time = time.time() - start_time
    print(f"Thời gian huấn luyện: {total_time/60:.2f} phút")

    # Lưu mô hình
    torch.save(model.state_dict(), "face_detector.pth")
    print("Đã lưu mô hình face_detector.pth")

    # Vẽ biểu đồ Loss
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("./loss_accuracy/celeba_loss_plot.png")
    plt.show()

    # Vẽ biểu đồ Accuracy
    plt.figure()
    plt.plot(epochs, train_accuracies, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("./loss_accuracy/celeba_accuracy_plot.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình nhận diện khuôn mặt")
    parser.add_argument("--data-dir", type=str, required=True, help="Đường dẫn đến thư mục chứa dữ liệu")
    parser.add_argument("--batch-size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--num-epochs", type=int, default=2, help="Số epoch huấn luyện")
    parser.add_argument("--lr", type=float, default=0.001, help="Tốc độ học")
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr
    )
