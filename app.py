import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from model import get_model

st.set_page_config(page_title="AI Phân Vùng Khối U Da", layout="wide")
st.title("Phần Mềm Phân Vùng Tổn Thương Da Liễu (ISIC 2018)")

@st.cache_resource
def load_ai_model():
    device = torch.device("cpu")
    model = get_model()
    # Sửa dòng load model trong app.py thành như sau:
    model.load_state_dict(torch.load('best_unet_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_ai_model()

# ==========================================
# THUẬT TOÁN DULLRAZOR - XÓA LÔNG BỀ MẶT
# ==========================================
def apply_dullrazor(img):
    # 1. Chuyển sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2. Bộ lọc BlackHat để tìm các sợi lông tối màu
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # 3. Làm mờ nhẹ và nhị phân hóa để tạo mặt nạ định vị sợi lông
    bhg = cv2.GaussianBlur(blackhat, (3, 3), 0)
    _, hair_mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)
    
    # 4. Nội suy (Inpaint) lấp đầy vùng lông bằng màu da/khối u lân cận
    img_cleaned = cv2.inpaint(img, hair_mask, 6, cv2.INPAINT_TELEA)
    return img_cleaned

uploaded_file = st.file_uploader("Tải ảnh da liễu lên đây (JPG/PNG)...", type=["jpg", "png", "jpeg"])

# Thêm nút gạt (Checkbox) lên giao diện
use_dullrazor = st.checkbox("Bật bộ lọc xóa lông tiền xử lý (DullRazor)")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    
    img_resized = cv2.resize(image_np, (256, 256))
    
    # QUYẾT ĐỊNH CÓ DÙNG DULLRAZOR HAY KHÔNG
    if use_dullrazor:
        img_processed = apply_dullrazor(img_resized)
    else:
        img_processed = img_resized.copy()

    # Tiền xử lý để đưa vào AI
    img_tensor = img_processed.astype(np.float32) / 255.0
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    img_tensor = torch.tensor(img_tensor).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        
    prob_mask = torch.sigmoid(output)
    binary_mask = (prob_mask > 0.5).float().squeeze().numpy()
    
    total_pixels = 256 * 256
    tumor_pixels = np.sum(binary_mask == 1)
    tumor_ratio = (tumor_pixels / total_pixels) * 100

    red_mask = np.zeros((256, 256, 3), dtype=np.uint8)
    red_mask[:, :, 0] = 255 
    red_mask_applied = cv2.bitwise_and(red_mask, red_mask, mask=(binary_mask * 255).astype(np.uint8))
    
    # Trộn ảnh (sau khi đã lọc lông hoặc không) với mặt nạ AI
    overlay_img = cv2.addWeighted(img_processed, 1.0, red_mask_applied, 0.4, 0)

    st.markdown(f"### 📊 Phân tích định lượng: Tổn thương chiếm **{tumor_ratio:.2f}%** diện tích bề mặt ảnh soi.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Đổi tiêu đề tự động dựa vào việc có bật DullRazor hay không
        st.header("1. Ảnh Đầu Vào" + (" (Đã Xóa Lông)" if use_dullrazor else ""))
        st.image(img_processed, use_container_width=True)
        
    with col2:
        st.header("2. Mặt Nạ AI")
        st.image(binary_mask, use_container_width=True, clamp=True)
        
    with col3:
        st.header("3. Lớp Phủ Lâm Sàng")
        st.image(overlay_img, use_container_width=True)
        
    st.success("Hệ thống đã hoàn tất phân tích!")