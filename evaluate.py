import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import ISICDataset
from model import get_model

# 1. Các hàm tính toán chỉ số Y khoa
def calculate_metrics(pred, target, threshold=0.5):
    # Ép về nhị phân: 1 (Có u), 0 (Không u)
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()

    # Tính TP, FP, TN, FN
    TP = (pred * target).sum().item()
    FP = (pred * (1 - target)).sum().item()
    FN = ((1 - pred) * target).sum().item()
    TN = ((1 - pred) * (1 - target)).sum().item()

    # Tính các chỉ số
    smooth = 1e-6 # Tránh lỗi chia cho 0
    dice = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)
    iou = (TP + smooth) / (TP + FP + FN + smooth)
    sensitivity = (TP + smooth) / (TP + FN + smooth) # Khả năng không bỏ sót
    specificity = (TN + smooth) / (TN + FP + smooth) # Khả năng không báo động giả

    return dice, iou, sensitivity, specificity

def evaluate():
    # 2. Xử lý ép chạy trên CPU
    device = torch.device('cpu')
    print("Đang khởi tạo hệ thống trên CPU...")

    # Nạp mô hình & Ép trọng số GPU sang CPU
    model = get_model()
    model.load_state_dict(torch.load('best_unet_model.pth', map_location=device))
    model.to(device)
    model.eval() # Bật chế độ thi cử (tắt dropout, batchnorm)

    # 3. Nạp dữ liệu Test 1000 ảnh (Nhớ đổi tên thư mục cho đúng)
    # Lưu ý: batch_size=4 để tránh lỗi Insufficient memory như hôm qua
    print("Đang nạp 1000 ảnh Test...")
    test_dataset = ISICDataset('data/Test_Images', 'data/Test_Masks', img_size=256)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    total_dice, total_iou, total_sens, total_spec = 0, 0, 0, 0
    num_batches = len(test_loader)

    print("BẮT ĐẦU ĐÁNH GIÁ...")
    with torch.no_grad(): # Tắt tính đạo hàm để tiết kiệm 100% RAM
        for batch_idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            dice, iou, sens, spec = calculate_metrics(outputs, masks)
            
            total_dice += dice
            total_iou += iou
            total_sens += sens
            total_spec += spec

            # In tiến độ
            if (batch_idx + 1) % 50 == 0:
                print(f" Đã quét {batch_idx + 1}/{num_batches} batch...")

    # 4. Tổng kết điểm trung bình
    print("\n" + "="*40)
    print("BÁO CÁO NGHIỆM THU TRÊN TẬP TEST ISIC 2018")
    print("="*40)
    print(f"1. Dice Score       : {total_dice/num_batches:.4f}")
    print(f"2. Jaccard (IoU)    : {total_iou/num_batches:.4f}")
    print(f"3. Độ nhạy (Sens)   : {total_sens/num_batches:.4f}")
    print(f"4. Độ đặc hiệu (Spec): {total_spec/num_batches:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate()