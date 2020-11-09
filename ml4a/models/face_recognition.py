from PIL import Image, ImageDraw
from random import random, sample
import numpy as np
import cv2
import dlib
import face_recognition

from ..utils import downloads


detector = None
predictor = None


def initialize_face_processing():
    global detector, predictor    
    landmarks_path = downloads.download_data_file(
        'https://raw.githubusercontent.com/ageitgey/face_recognition_models/master/face_recognition_models/models/shape_predictor_68_face_landmarks.dat', 
        'face_recognition/shape_predictor_68_face_landmarks.dat')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(landmarks_path)
    
    
def get_encodings(target_face_img):
    if predictor is None:
        initialize_face_processing()
    #target_face_img = face_recognition.load_image_file(filename)
    target_face_img = np.array(target_face_img)
    target_encodings = face_recognition.face_encodings(target_face_img)
    return target_encodings


def get_face(img, target_encodings=None):
    if predictor is None:
        initialize_face_processing()
    img = np.array(img)
    locations = face_recognition.face_locations(img, model="cnn")
    if len(locations) == 0:
        return None, None, None, None, None
    encodings = face_recognition.face_encodings(img, locations)
    landmarks = face_recognition.face_landmarks(img, locations)
    if target_encodings is not None:
        distances = [face_recognition.face_distance([target_encodings], encoding) for encoding in encodings]
        idx_closest = distances.index(min(distances))
        target_face, target_landmarks = locations[idx_closest], landmarks[idx_closest]
    else:
        target_face, target_landmarks = locations[0], landmarks[0]
    top, right, bottom, left = target_face
    x, y, w, h = left, top, right-left, bottom-top
    return x, y, w, h, target_landmarks


def draw_landmarks(img, landmarks, color=(255,255,255,255), width=1):
    img = Image.fromarray(np.copy(img))
    d = ImageDraw.Draw(img, 'RGBA')
    whole_face = landmarks['chin'] + list(reversed(landmarks['right_eyebrow'])) + list(reversed(landmarks['left_eyebrow'])) + [landmarks['chin'][0]]
    #d.line(landmarks['left_eyebrow'], fill=color, width=width)
    #d.line(landmarks['right_eyebrow'], fill=color, width=width)
    d.line(landmarks['left_eye'], fill=color, width=width)
    d.line(landmarks['right_eye'], fill=color, width=width)
    d.line(landmarks['top_lip'], fill=color, width=width)
    d.line(landmarks['bottom_lip'], fill=color, width=width)
    d.line(landmarks['nose_bridge'], fill=color, width=width)
    d.line(landmarks['nose_tip'], fill=color, width=width)
    #d.line(landmarks['chin'], fill=color, width=width)
    d.line(whole_face, fill=color, width=width)
    return img

