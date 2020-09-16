from imutils.face_utils import FaceAligner
from PIL import Image
import numpy as np
import imutils
import dlib
import cv2

from ..utils import downloads

fa_loaded = False


def setup_face_aligner():
    global detector, predictor, fa, fa_loaded
    detector = dlib.get_frontal_face_detector()
    predictor_file = downloads.download_data_file(
        url='https://storage.googleapis.com/glow-demo/shape_predictor_68_face_landmarks.dat', 
        output_path='face/shape_predictor_68_face_landmarks.dat')
    predictor = dlib.shape_predictor(predictor_file)
    fa = FaceAligner(predictor, desiredFaceWidth=256, desiredLeftEye=(0.371, 0.480))
    fa_loaded = True

    
def align_face(img, width=800):
    global fa_loaded
    if not fa_loaded:
        setup_face_aligner()
    
    img = np.array(img)
    img = img[:, :, ::-1]  # Convert from RGB to BGR format
    img = imutils.resize(img, width=width)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 2)

    if len(rects) > 0:
        align_img = fa.align(img, gray, rects[0])[:, :, ::-1]
        align_img = np.array(Image.fromarray(align_img).convert('RGB'))
        return align_img, True
    else:
        print('Warning: No face found!')
        return None, False

    
def align_face_from_path(img_path, width=800):
    img = Image.open(img_path).convert('RGB') 
    x, face_found = align_face(img, width)
    return x, face_found