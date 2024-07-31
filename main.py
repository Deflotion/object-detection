# Python In-built packages
from pathlib import Path
import PIL
# External packages
import streamlit as st
# Local Modules
import configs
import func

# Load Pre-trained ML Model
model_image = func.load_model(configs.IMAGE_DETECTION_MODEL)
model_live = func.load_model(configs.IMAGE_DETECTION_MODEL)

# Setting page layout
st.set_page_config(
    page_title="Real Time Object Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Live and Real Time Object Detection")

# Sidebar
st.sidebar.header("Masih Pemula")



st.write('Selamat datang di project aplikasi web untuk mendeteksi objek menggunakan Machine Learning')
st.sidebar.write("Silahkan Pilih Metode deteksi dibawah")
source_radio = st.sidebar.radio(
    "Pilih:", configs.SOURCES_LIST)

# If image is selected
source_img = None
if source_radio == configs.IMAGE:
    source_img = st.file_uploader(
        "Pilih Gambar...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    col1, col2 = st.columns(2)
    detect_btn = st.button('Detect Objects')
    
    with col1:
        try:
            if source_img is None:
                st.write("Gambar Tidak ada!")
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image", use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            st.write("Tidak ada gambar yang dideteksi!")
        else:
            if detect_btn:
                res = model_image.predict(uploaded_image, conf=0.7)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image', use_column_width=True)
                # try:
                #     with st.expander("Detection Results"):
                #         for box in boxes:
                #             st.write(box.data)
                # except Exception as ex:
                #     # st.write(ex)
                #     st.write("No image is uploaded yet!")

elif source_radio == configs.WEBCAM:
    func.play_webcam(0.7, model_live)

else:
    st.error("Please select a valid source type!")