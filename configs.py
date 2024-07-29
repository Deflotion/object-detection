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
LIVE_DETECTION_MODEL = 'http://127.0.0.1:5000/live'
IMAGE_DETECTION_MODEL = 'http://127.0.0.1:5000/predict'

# Webcam
WEBCAM_PATH = 0
