from flask import Flask, render_template, request
from ultralytics import YOLO
import configs
import streamlit as st
import cv2
import main
import func

app = Flask(__name__ ,static_url_path='/models')
model_image = func.load_model(configs.IMAGE_DETECTION_MODEL)
model_live = func.load_model(configs.LIVE_DETECTION_MODEL)

def load_model(model_path):
    model = YOLO(model_path)
    return model


def display_tracker():
    is_display_tracker = True
    if is_display_tracker:
        tracker_type = "bytetrack.yaml"
        return is_display_tracker, tracker_type
    return is_display_tracker, None


@app.route('/')
def index():
    return 'connect Succesfull'

@app.route('/live', methods=['POST'])
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
    st_frame.image(res_plotted,caption='Detected Video',channels="BGR", use_column_width=True)

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
                    _display_detected_frames(conf,model,st_frame,image,is_display_tracker,tracker,)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

@app.route('/detect', methods=['POST'])
def detect():
    res = model_image.predict(main.uploaded_image,conf=0.7)
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
    return res.imgs[0]

if __name__ == '__main__':
    app.run(debug=True)
