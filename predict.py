import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import get_model

# 1. Khởi tạo mô hình
model = get_model()

# --- NẠP TRỌNG SỐ TỐT NHẤT TỪ EPOCH 7 ---
model.load_state_dict(torch.load('best_unet_model.pth', map_location=torch.device('cpu')))

model.eval()

# 2. Đọc và tiền xử lý ảnh (Chọn 1 ảnh từ thư mục data/images của bạn)
image_path = 'data/images/ISIC_0000000.jpg' # Cập nhật tên file cho đúng
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (256, 256))

# Chuẩn hóa ma trận y như trong file dataset.py
img_tensor = img_resized.astype(np.float32) / 255.0
img_tensor = np.transpose(img_tensor, (2, 0, 1))

# PyTorch luôn yêu cầu chiều Batch. Ta thêm 1 chiều ảo ở đầu để thành [1, 3, 256, 256]
img_tensor = torch.tensor(img_tensor).unsqueeze(0) 

# 3. Dự đoán
# torch.no_grad() giúp tắt tính toán đạo hàm, tiết kiệm bộ nhớ khi test
with torch.no_grad():
    output = model(img_tensor)
    
# 4. Hậu xử lý (Post-processing)
# Output đang là giá trị thô (logits). Cần ép qua hàm Sigmoid để biến nó thành xác suất [0 đến 1]
prob_mask = torch.sigmoid(output)

# Phân ngưỡng (Thresholding): Pixel nào có xác suất > 50% là khối u thì cho thành 1 (màu trắng), còn lại là 0 (màu đen)
binary_mask = (prob_mask > 0.5).float()

# Lột bỏ các chiều ảo và chuyển lại về NumPy để matplotlib vẽ được (từ [1, 1, 256, 256] thành [256, 256])
binary_mask = binary_mask.squeeze().numpy()

# 5. Hiển thị
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img_resized)
axes[0].set_title('Ảnh gốc (Input)')
axes[0].axis('off')

axes[1].imshow(binary_mask, cmap='gray')
axes[1].set_title('AI Dự đoán (Prediction)')
axes[1].axis('off')

plt.tight_layout()
plt.show()