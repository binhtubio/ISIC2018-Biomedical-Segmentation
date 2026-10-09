import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from dataset import ISIC_Task2_Dataset
from model import get_model

def train_task2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang khởi động huấn luyện Task 2 trên: {device}")

    # 1. Nạp phễu dữ liệu (Sẽ trỏ về Google Drive khi lên Colab)
    dataset = ISIC_Task2_Dataset(
        image_dir='data/images', 
        mask_dir='data/ISIC2018_Task2_Training_GroundTruth_v3'
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    # 2. Khởi tạo U-Net (Đã cấu hình classes=5 ở model.py)
    model = get_model().to(device)

    # 3. Tối ưu Loss Đa nhãn (Multi-label)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = smp.losses.DiceLoss(mode='multilabel')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4. Vòng lặp học tập
    epochs = 10
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Ép AI học đồng thời 5 kênh độc lập
            loss = criterion_bce(outputs, masks) + criterion_dice(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f" Epoch [{epoch+1}/{epochs}] - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        avg_loss = train_loss / len(train_loader)
        print(f"==> Tổng kết Epoch {epoch+1} | Trung bình Loss: {avg_loss:.4f}")
        
        # Lưu lại "bộ não" tốt nhất
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_unet_task2.pth')
            print("🌟 Đã lưu trọng số Task 2 xuất sắc nhất!")

if __name__ == '__main__':
    train_task2()