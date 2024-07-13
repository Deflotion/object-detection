# Python In-built packages
from pathlib import Path
import PIL
# External packages
import streamlit as st
# Local Modules
import configs
import func


## Function
def load_model(model_path):
    model = YOLO(model_path)
    return model


def display_tracker():
    is_display_tracker = True
    if is_display_tracker:
        tracker_type = "bytetrack.yaml"
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

def play_webcam(conf, model):
    source_webcam = configs.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker()
    if st.button('Start Camera'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

## configs
from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
WEBCAM = 'Webcam'

SOURCES_LIST = [IMAGE, WEBCAM]

# Images config
IMAGES_DIR = ROOT / 'images'

# ML Model config
MODEL_DIR = ROOT / 'models'
DETECTION_MODEL = MODEL_DIR / 'yolov8l.pt'

# Webcam
WEBCAM_PATH = 0



## Main page
# Load Pre-trained ML Model
model = func.load_model(configs.DETECTION_MODEL)

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
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            st.write("Tidak ada gambar yang dideteksi!")
        else:
            if detect_btn:
                res = model.predict(uploaded_image,
                                    conf=0.7
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == configs.WEBCAM:
    func.play_webcam(0.7, model)

else:
    st.error("Please select a valid source type!")