import os
import cv2

# Thư mục chứa ảnh gốc và nhãn
image_folder = "demo/images"
label_folder = "demo/labels"

# Thư mục để lưu ảnh đã cắt
output_image_folder = 'demo/cropped_images'

os.makedirs(output_image_folder, exist_ok=True)

# Duyệt qua tất cả các tệp tin label và cắt ảnh
for label_file in os.listdir(label_folder):
    if label_file.endswith('.txt'):
        # Đọc nội dung từ tệp tin label
        with open(os.path.join(label_folder, label_file), 'r') as file:
            content = file.readlines()
        
        if len(content) > 0:
            # Trích xuất thông tin từ label (class_id, x, y, w, h)
            values = content[0].split()
            class_id = int(values[0])
            x, y, w, h = map(float, values[1:])

            
            # Đường dẫn đến ảnh gốc
            image_path = os.path.join(image_folder, os.path.splitext(label_file)[0] + '.jpg')
            
            # Đường dẫn đến ảnh đã cắt
            cropped_image_path = os.path.join(output_image_folder, label_file.replace('.txt', '.jpg'))
            
            # Cắt ảnh
            image = cv2.imread(image_path)
            h_img, w_img, _ = image.shape
            x, y, w, h = int(x * w_img), int(y * h_img), int(w * w_img), int(h * h_img)
            x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
            cropped_image = image[y1:y2, x1:x2]
            
            # Lưu ảnh đã cắt
            cv2.imwrite(cropped_image_path, cropped_image)