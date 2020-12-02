from imutils.face_utils import FaceAligner
from PIL import Image
import numpy as np
import imutils
import dlib
import cv2

from ..utils import downloads

detector = None
predictor = None


def setup_face_detection():
    global detector, predictor
    predictor_file = downloads.download_data_file(
        url='https://storage.googleapis.com/glow-demo/shape_predictor_68_face_landmarks.dat', 
        output_path='face_recognition/shape_predictor_68_face_landmarks.dat')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_file)

    
def align_face(img, face_width=256, resize_width=800):
    if not predictor:
        setup_face_detection()

    face_aligner = FaceAligner(
        predictor, 
        desiredFaceWidth=face_width, 
        desiredLeftEye=(0.371, 0.480)
    )

    img = np.array(img)
    img = img[:, :, ::-1]  # Convert from RGB to BGR format
    img = imutils.resize(img, width=resize_width)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 2)

    if len(rects) > 0:
        align_img = face_aligner.align(img, gray, rects[0])[:, :, ::-1]
        align_img = np.array(Image.fromarray(align_img).convert('RGB'))
        return align_img, True
    else:
        print('Warning: No face found!')
        return None, False

    
def align_face_from_path(img_path, face_width=256, resize_width=800):
    img = Image.open(img_path).convert('RGB') 
    x, face_found = align_face(img, face_width, resize_width)
    return x, face_found