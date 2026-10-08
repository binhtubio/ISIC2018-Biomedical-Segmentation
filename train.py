import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
import time

from dataset import ISICDataset
from model import get_model

# 1. Hàm tính Dice Score để đo lường phần trăm diện tích giao nhau
def dice_score(pred, target, threshold=0.5):
    # Ép xác suất về 0 hoặc 1
    pred = (torch.sigmoid(pred) > threshold).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    if union == 0:
        return 1.0 # Nếu cả dự đoán và thực tế đều không có u (toàn màu đen) thì coi như đúng 100%
    return (2. * intersection) / union

# 2. Tích hợp Hàm Loss: 50% BCE + 50% Dice Loss
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, y_pred, y_true):
        return 0.5 * self.bce(y_pred, y_true) + 0.5 * self.dice(y_pred, y_true)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")
    if device.type == 'cpu':
        print("⚠️ CẢNH BÁO: Đang train toàn bộ dữ liệu trên CPU. Quá trình này sẽ mất nhiều giờ. Hãy cắm sạc laptop!")

    # 3. Nạp và chia dữ liệu (80% Train, 20% Validation)
    print("Đang nạp dữ liệu...")
    full_dataset = ISICDataset('data/images', 'data/masks', img_size=256)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    print(f"Số batch Train: {len(train_loader)} | Số batch Validation: {len(val_loader)}")

    # 4. Khởi tạo
    model = get_model().to(device)
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50 # Chạy trước 10 epochs để đánh giá
    best_val_dice = 0.0
    
    print("BẮT ĐẦU HUẤN LUYỆN CHÍNH THỨC...")
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # --- PHASE A: HỌC (TRAIN) ---
        model.train()
        train_loss, train_dice = 0.0, 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_dice += dice_score(outputs, masks).item()
            
            # In tiến độ mỗi 50 batch để không bị trôi màn hình
            if (batch_idx + 1) % 50 == 0:
                print(f"   [Train] Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        
        # --- PHASE B: THI THỬ (VALIDATION) ---
        model.eval()
        val_loss, val_dice = 0.0, 0.0
        
        with torch.no_grad(): # Tắt đạo hàm để tăng tốc và tiết kiệm RAM
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_dice += dice_score(outputs, masks).item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        
        end_time = time.time()
        
        # --- PHASE C: TỔNG KẾT & LƯU CHỐT ---
        print(f"\n==> EPOCH {epoch+1} hoàn tất trong {(end_time - start_time):.0f} giây")
        print(f"    Train | Loss: {avg_train_loss:.4f} | Dice Score: {avg_train_dice:.4f}")
        print(f"    Valid | Loss: {avg_val_loss:.4f} | Dice Score: {avg_val_dice:.4f}")
        
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), 'best_unet_model.pth')
            print(f"    [⭐] Đã lưu model tốt nhất mới (Dice: {best_val_dice:.4f}) -> best_unet_model.pth\n")
        else:
            print("\n")

if __name__ == "__main__":
    train()