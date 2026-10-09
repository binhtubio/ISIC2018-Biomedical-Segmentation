from dataset import ISIC_Task2_Dataset

# Trỏ đúng tên thư mục bạn vừa giải nén
image_dir = 'data/images'
mask_dir = 'data/Task2_Masks_Train'

try:
    dataset = ISIC_Task2_Dataset(image_dir, mask_dir)
    img, masks = dataset[0]
    print(f"Kích thước ảnh gốc: {img.shape}")
    print(f"Kích thước khối mặt nạ 5 lớp: {masks.shape}")
    print("✅ THÀNH CÔNG! Dữ liệu đã khớp nhau và sẵn sàng cho Task 2.")
except Exception as e:
    print(f"❌ Có lỗi xảy ra: {e}")