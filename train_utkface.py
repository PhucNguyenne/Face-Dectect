import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utkface_dataset import UTKFaceDataset
from torchvision import models, transforms
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# Hàm chuyển tuổi thành nhóm tuổi (bin)
def get_age_bin(age):
    if age < 20:
        return 0
    elif age < 40:
        return 1
    elif age < 60:
        return 2
    else:
        return 3

# Mô hình với hai đầu ra: một cho tuổi (phân loại nhóm tuổi) và một cho giới tính
class UTKFaceModel(nn.Module):
    def __init__(self, num_age_bins=4, num_gender_classes=2):
        super(UTKFaceModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Loại bỏ lớp fc gốc
        feature_dim = 512

        # Thêm dropout trước khi linear
        self.age_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feature_dim, num_age_bins)
        )
        self.gender_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feature_dim, num_gender_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        age_logits = self.age_head(features)
        gender_logits = self.gender_head(features)
        return age_logits, gender_logits


def train_utkface(data_dir, batch_size=32, num_epochs=20, lr=0.001):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    dataset = UTKFaceDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UTKFaceModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Lưu lại các metric cho từng epoch
    train_losses = []
    train_age_accuracies = []
    train_gender_accuracies = []
    
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_age_correct = 0
        running_gender_correct = 0
        total_samples = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for images, ages, genders in pbar:
            images = images.to(device)
            # Chuyển đổi tuổi thành nhóm tuổi (bin)
            age_bins = torch.tensor([get_age_bin(age) for age in ages], dtype=torch.long).to(device)
            genders = torch.tensor(genders, dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            age_logits, gender_logits = model(images)
            loss_age = criterion(age_logits, age_bins)
            loss_gender = criterion(gender_logits, genders)
            loss = loss_age + loss_gender
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            total_samples += images.size(0)
            
            _, age_preds = torch.max(age_logits, 1)
            running_age_correct += torch.sum(age_preds == age_bins).item()
            
            _, gender_preds = torch.max(gender_logits, 1)
            running_gender_correct += torch.sum(gender_preds == genders).item()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        epoch_loss = running_loss / len(dataloader)
        epoch_age_acc = running_age_correct / total_samples
        epoch_gender_acc = running_gender_correct / total_samples
        
        train_losses.append(epoch_loss)
        train_age_accuracies.append(epoch_age_acc)
        train_gender_accuracies.append(epoch_gender_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Age Acc: {epoch_age_acc:.4f}, Gender Acc: {epoch_gender_acc:.4f}")
    
    total_time = time.time() - start_time
    print(f"Thời gian huấn luyện: {total_time/60:.2f} phút")
    torch.save(model.state_dict(), "utkface_model.pth")
    print("Đã lưu mô hình utkface_model.pth")
    
    # Vẽ biểu đồ Loss
    epochs = range(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./loss_accuracy/utkface_loss_plot.png")
    plt.show()
    
    # Vẽ biểu đồ Accuracy: vẽ Age và Gender trên cùng 1 plot
    plt.figure()
    plt.plot(epochs, train_age_accuracies, 'b-', label='Age Accuracy')
    plt.plot(epochs, train_gender_accuracies, 'r-', label='Gender Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("./loss_accuracy/utkface_accuracy_plot.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UTKFace for Age and Gender")
    parser.add_argument("--data-dir", type=str, required=True, help="Đường dẫn đến thư mục chứa ảnh UTKFace")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    train_utkface(args.data_dir, args.batch_size, args.num_epochs, args.lr)
