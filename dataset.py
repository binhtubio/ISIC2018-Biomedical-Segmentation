import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size  # <--- BỔ SUNG DÒNG NÀY ĐỂ TRÁNH LỖI
        
        # Lọc chỉ lấy các file .jpg
        all_files = os.listdir(image_dir)
        self.image_names = [f for f in all_files if f.endswith('.jpg')]

        # Cài đặt Albumentations
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5), # 50% cơ hội lật ngang
            A.VerticalFlip(p=0.5),   # 50% cơ hội lật dọc
            # Chuẩn hóa ảnh màu
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.jpg', '_segmentation.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        # --- CHỐT CHẶN BẢO VỆ 2.0 ---
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Nếu không tìm thấy file ảnh gốc hoặc file mask, thay vì báo lỗi làm sập chương trình,
        # ta trả về một ma trận toàn số 0 để lấp chỗ trống.
        if image is None or mask is None:
            # In ra cảnh báo để bạn biết file nào đang bị lỗi
            print(f"\n[BỎ QUA] Lỗi đọc file: {img_name} hoặc {mask_name}")
            return torch.zeros((3, self.img_size, self.img_size)), torch.zeros((1, self.img_size, self.img_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Đưa cả ảnh và mask qua pipeline Albumentations
        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed['image']
        mask_tensor = transformed['mask']
        
        # Mask đang có giá trị 0 hoặc 255 và shape [256, 256]. 
        # Cần chuyển về [0, 1] và thêm chiều channel thành [1, 256, 256]
        mask_tensor = (mask_tensor.unsqueeze(0).float() / 255.0)
        
        return image_tensor, mask_tensor

if __name__ == "__main__":
    my_dataset = ISICDataset('data/images', 'data/masks')
    img, msk = my_dataset[0]
    print(f"Test Augmentation -> Ảnh: {img.shape}, Mask: {msk.shape}")
class ISIC_Task2_Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        
        # 5 thuộc tính lâm sàng của Task 2
        self.attributes = [
            'pigment_network', 
            'negative_network', 
            'streaks', 
            'milia_like_cyst', 
            'globules'
        ]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        base_name = img_name.replace('.jpg', '')
        
        # 1. Nạp ảnh gốc
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # 2. Quét và gom 5 chiếc mặt nạ
        masks = []
        for attr in self.attributes:
            # Ban tổ chức ISIC đặt tên file mặt nạ theo chuẩn này:
            mask_name = f"{base_name}_attribute_{attr}.png"
            mask_path = os.path.join(self.mask_dir, mask_name)
            
            if os.path.exists(mask_path):
                # Nếu có mặt nạ -> Bác sĩ xác nhận có thuộc tính này
                mask = Image.open(mask_path).convert('L')
                mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
                mask_np = np.array(mask) / 255.0
            else:
                # Nếu không có file -> Vùng da đó không có thuộc tính này (mặt nạ đen toàn tập)
                mask_np = np.zeros((self.img_size, self.img_size))
                
            masks.append(mask_np)

        # 3. Ép 5 mặt nạ thành một khối Tensor [5, 256, 256]
        masks_tensor = torch.tensor(np.stack(masks, axis=0), dtype=torch.float32)
        
        return image, masks_tensor