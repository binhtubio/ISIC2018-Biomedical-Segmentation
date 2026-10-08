import cv2
import matplotlib.pyplot as plt
import os

# Đường dẫn đến 1 cặp ảnh - mask cụ thể (thay đổi tên file cho khớp với dữ liệu bạn tải)
image_path = 'data/images/ISIC_0012255.jpg'
mask_path = 'data/masks/ISIC_0012255_segmentation.png'

# 1. Đọc ảnh gốc bằng OpenCV
# OpenCV mặc định đọc ảnh màu theo không gian BGR. Cần chuyển sang RGB để matplotlib hiển thị đúng màu da thực tế.
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Đọc ảnh mặt nạ (Ground Truth)
# Mặt nạ là ảnh nhị phân (chỉ có màu đen/trắng), nên ta đọc trực tiếp ở chế độ ảnh xám (Grayscale).
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
# --- BƯỚC 2.1: RESIZE ẢNH ---
IMG_SIZE = 256

# Resize ảnh gốc
image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))

# Resize mặt nạ 
# LƯU Ý QUAN TRỌNG: Với mask, bắt buộc dùng nội suy INTER_NEAREST để các pixel 
# chỉ giữ lại đúng giá trị 0 (đen) hoặc 255 (trắng), không tạo ra các mức xám mờ ở đường viền.
mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

# Hiển thị đối chiếu ảnh sau khi resize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(image_resized)
axes[0].set_title(f'Ảnh gốc đã Resize ({IMG_SIZE}x{IMG_SIZE})')
axes[0].axis('off')

axes[1].imshow(mask_resized, cmap='gray')
axes[1].set_title(f'Mặt nạ đã Resize ({IMG_SIZE}x{IMG_SIZE})')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# Kiểm tra kích thước của hai ma trận
print(f"Kích thước ảnh gốc: {image_rgb.shape}") # Ví dụ: (2000, 3000, 3) - Chiều cao, chiều rộng, 3 kênh màu
print(f"Kích thước mặt nạ: {mask.shape}")       # Ví dụ: (2000, 3000) - Chỉ có chiều cao và chiều rộng

# 3. Hiển thị đối chiếu trên cùng một khung hình
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(image_rgb)
axes[0].set_title('Ảnh gốc (Input)')
axes[0].axis('off')

axes[1].imshow(mask, cmap='gray')
axes[1].set_title('Mặt nạ tổn thương (Ground Truth)')
axes[1].axis('off')

plt.tight_layout()
plt.show()
# =========================================================
# --- BƯỚC 2.2: THUẬT TOÁN DULLRAZOR XÓA LÔNG ---
# =========================================================

# 1. Chuyển ảnh resize về ảnh xám để xử lý cường độ sáng dễ hơn
gray_resized = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)

# Khởi tạo Kernel (khối ma trận cấu trúc) hình chữ nhật kích thước 17x17
# Kích thước kernel này tương đương với độ dài/dày tối đa của sợi lông cần tìm
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))

# Áp dụng phép toán BlackHat (Phát hiện chi tiết tối trên nền sáng)
blackhat = cv2.morphologyEx(gray_resized, cv2.MORPH_BLACKHAT, kernel)

# 2. Phân ngưỡng (Thresholding) để tạo mặt nạ sợi lông
# Mọi pixel trong blackhat có giá trị độ xám > 10 sẽ biến thành 255 (trắng), còn lại thành 0 (đen)
_, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

# 3. Inpainting (Lấp đầy)
# Lưu ý: Hàm cv2.inpaint hoạt động chuẩn nhất trên hệ màu BGR gốc của OpenCV
image_bgr_resized = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)

# Thực hiện Inpainting sử dụng thuật toán Telea (cv2.INPAINT_TELEA)
# inpaintRadius=3 là bán kính số pixel xung quanh sợi lông được dùng để nội suy màu lấp vào
inpainted_bgr = cv2.inpaint(image_bgr_resized, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# Chuyển lại về RGB để hiển thị bằng Matplotlib
image_dullrazor_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

# =========================================================
# --- HIỂN THỊ KẾT QUẢ DULLRAZOR ---
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image_resized)
axes[0].set_title('1. Ảnh gốc sau Resize')
axes[0].axis('off')

axes[1].imshow(hair_mask, cmap='gray')
axes[1].set_title('2. Mặt nạ phát hiện lông (BlackHat)')
axes[1].axis('off')

axes[2].imshow(image_dullrazor_rgb)
axes[2].set_title('3. Ảnh sau Inpainting (Đã xóa lông)')
axes[2].axis('off')

plt.tight_layout()
plt.show()