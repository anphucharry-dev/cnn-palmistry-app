import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# 1. Cấu hình trang web
st.set_page_config(page_title="Xem Chỉ Tay AI", page_icon="🔮", layout="centered")
st.title("🔮 Đồ Án AI: Phân Tích Chỉ Tay Bằng Mạng CNN")
st.write("Ứng dụng sử dụng Transfer Learning (MobileNetV2) để phân tích các đường nét trên lòng bàn tay của bạn.")

# 2. Tải mô hình (Dùng st.cache_resource để không phải tải lại model mỗi lần bấm nút)
@st.cache_resource
def load_model():
    # Tên file model phải khớp với file bạn tải về
    return tf.keras.models.load_model('Palmistry_Model.keras')

model = load_model()
class_names = ['M shape palm real photo', 'messy palm lines real photo', 'normal palm real photo']

fortune_dict = {
    'M shape palm real photo': "🌟 **Bàn Tay Chữ M:** Bạn có trực giác nhạy bén, khả năng lãnh đạo xuất chúng và hậu vận vô cùng rực rỡ, giàu sang!",
    'messy palm lines real photo': "🌊 **Chỉ Tay Rối/Đứt Đoạn:** Cuộc sống của bạn có thể trải qua nhiều thử thách. Tuy nhiên, chỉ cần giữ vững lý trí, mọi khó khăn đều sẽ qua.",
    'normal palm real photo': "⚖️ **Bàn Tay Phổ Thông:** Cuộc sống của bạn bình yên, ổn định. Bạn là người thực tế, chăm chỉ và sẽ đạt được thành công nhờ sự kiên trì."
}

# 3. Tạo Tabs cho 2 lựa chọn: Upload ảnh hoặc Dùng Webcam
tab1, tab2 = st.tabs(["📁 Tải ảnh lên", "📸 Chụp từ Webcam"])

img_file = None

with tab1:
    upload_file = st.file_uploader("Chọn một bức ảnh lòng bàn tay rõ nét", type=['jpg', 'jpeg', 'png'])
    if upload_file is not None:
        img_file = upload_file

with tab2:
    camera_file = st.camera_input("Đưa lòng bàn tay vào khung hình và chụp")
    if camera_file is not None:
        img_file = camera_file

# 4. Xử lý và Dự đoán
if img_file is not None:
    # Hiển thị ảnh
    image = Image.open(img_file)
    st.image(image, caption="Ảnh đang phân tích...", use_column_width=True)
    
    if st.button("Bắt đầu xem chỉ tay 🔮"):
        with st.spinner("AI đang 'bắt mạch' chỉ tay..."):
            try:
                # Tiền xử lý ảnh cho CNN
                img_resized = image.resize((224, 224))
                
                # Chuyển RGBA sang RGB nếu có (đối với ảnh png)
                if img_resized.mode != "RGB":
                    img_resized = img_resized.convert("RGB")
                    
                img_array = tf.keras.utils.img_to_array(img_resized)
                img_array = tf.expand_dims(img_array, 0) # Thêm batch dimension
                
                # Dự đoán
                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])
                class_idx = np.argmax(predictions[0])
                
                predicted_class = class_names[class_idx]
                confidence = 100 * np.max(predictions[0])
                
                # In kết quả
                st.success(f"**Kết quả nhận diện:** {predicted_class} (Độ tự tin: {confidence:.2f}%)")
                st.info(fortune_dict[predicted_class])
                
            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {e}")