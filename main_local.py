import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np

# Size Window
ZONE_POLYGON = np.array([
  [0,0],
  [1280 // 2,0],
  [1280 // 2,720],
  [0,720]
])

# Set Resolution Camera
def parse_argument() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Object Detection Live")
  parser.add_argument(
    "--webcam-resolution", 
    default=[1280,720], 
    nargs=2, 
    type=int
    )
  args = parser.parse_args()
  return args

def main():
  args = parse_argument()
  frame_width, frame_height = args.webcam_resolution
  # Capture Camera
  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
  
  # Modeling
  model = YOLO("yolov8l.pt")
  
  # Use GPU
  # model.to('cuda')
  
  box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
  )
  
  # Custom Zone area Window
  zone = sv.PolygonZone(polygon=ZONE_POLYGON, frame_resolution_wh=tuple(args.webcam_resolution))
  
  # Custom Frame Red(Optional)
  # zone_annotator = sv.PolygonZoneAnnotator(
  #   zone=zone, 
  #   color=sv.Color.red(),
  #   thickness=2,
  #   text_thickness=2,
  #   text_scale=2
  #   )

  while(True):
    ret, frame = cap.read()
    # Result
    result = model(frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)
    
    # Ignore person
    # detections = detections[detections.class_id != 0]
    
    # Labels Predict
    labels = [
      f"{model.model.names[class_id]} {confidence:0.2f}"
      for _,confidence, class_id, _
      in detections
    ]
    
    # Shape Box
    frame = box_annotator.annotate(
      scene=frame, 
      detections=detections, 
      labels=labels
      )
    
    zone.trigger(detections=detections)
    # frame = zone_annotator.annotate(scene=frame)
    
    cv2.imshow("Object Detection", frame)
    
    if(cv2.waitKey(30)==27):
      break

if __name__ == "__main__":
  main()