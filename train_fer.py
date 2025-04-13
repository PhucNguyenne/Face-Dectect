import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from fer_dataset import FERDataset
from torchvision import models, transforms
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

def train_fer(data_dir, batch_size=32, num_epochs=20, lr=0.001):
    # Định nghĩa transform cho ảnh
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Tạo dataset và DataLoader
    dataset = FERDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    
    num_emotions = 7  # Số lớp cảm xúc trong FER2013
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_emotions)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Lưu lại các metric cho từng epoch
    train_losses = []
    train_accuracies = []
    
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = running_corrects / total_samples
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    
    total_time = time.time() - start_time
    print(f"Thời gian huấn luyện: {total_time/60:.2f} phút")
    
    torch.save(model.state_dict(), "fer_model.pth")
    print("Đã lưu mô hình fer_model.pth")
    
    # Vẽ biểu đồ Loss
    epochs = range(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./loss_accuracy/fer_loss_plot.png")
    plt.show()
    
    # Vẽ biểu đồ Accuracy
    plt.figure()
    plt.plot(epochs, train_accuracies, 'b-', label='Train Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("./loss_accuracy/fer_accuracy_plot.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FER model for Emotion Recognition")
    parser.add_argument("--data-dir", type=str, required=True, help="Đường dẫn đến thư mục dữ liệu FER (cấu trúc: mỗi lớp một thư mục)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    train_fer(args.data_dir, args.batch_size, args.num_epochs, args.lr)
