import torch
import segmentation_models_pytorch as smp

# --- KHỞI TẠO MÔ HÌNH U-NET ---
def get_model():
    model = smp.Unet(
        encoder_name="resnet18",        # Sử dụng sườn nén là mạng ResNet18
        encoder_weights="imagenet",     # Sử dụng trọng số đã học trước từ bộ dữ liệu ImageNet (giúp AI hội tụ nhanh hơn)
        in_channels=3,                  # Ảnh đầu vào có 3 kênh màu (RGB)
        classes=1,                      # Đầu ra là 1 kênh duy nhất (Mặt nạ nhị phân đen/trắng)
    )
    return model

# ==========================================
# TEST THỬ KIẾN TRÚC MÔ HÌNH
# ==========================================
if __name__ == "__main__":
    print("Đang tải mạng U-Net (ResNet18 backbone)...")
    model = get_model()
    
    # AI trong PyTorch xử lý dữ liệu theo lô (Batch). 
    # Shape yêu cầu là: [Batch_size, Channels, Height, Width]
    # Ta tạo một tensor giả lập đại diện cho 1 bức ảnh RGB 256x256
    dummy_input = torch.randn(1, 3, 256, 256)
    
    print(f"Hình dáng dữ liệu ĐẦU VÀO: {dummy_input.shape}")
    
    # Cho bức ảnh đi qua mô hình (Forward pass)
    # LƯU Ý: Ở bước test này, mô hình sẽ trả ra kết quả ngẫu nhiên vì chưa được huấn luyện (Train)
    output = model(dummy_input)
    
    print(f"Hình dáng dữ liệu ĐẦU RA: {output.shape}")
    print("Thành công! Mô hình đã sẵn sàng.")