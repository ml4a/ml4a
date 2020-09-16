import numpy as np
from PIL import Image, ImageDraw
from random import random, sample
import cv2
import dlib
import face_recognition


# model
detector = None
predictor = None
jx0, jy0, jw0, jh0 = None, None, None, None


def initialize_face_processing(landmarks_path):
    face_landmark_shape_file = landmarks_path
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_shape_file)
    
    
def get_encodings(filename):
    target_face_img = face_recognition.load_image_file(filename)
    target_encodings = face_recognition.face_encodings(target_face_img)[0]
    return target_encodings


def get_face(img, target_encodings):
    img = np.array(img)
    locations = face_recognition.face_locations(img, model="cnn")
    encodings = face_recognition.face_encodings(img, locations)
    landmarks = face_recognition.face_landmarks(img, locations)
    if len(locations) == 0:
        return None, None, None, None, None
    if target_encodings is not None:
        distances = [ face_recognition.face_distance([target_encodings], encoding) for encoding in encodings ]
        idx_closest = distances.index(min(distances))
        target_face, target_landmarks = locations[idx_closest], landmarks[idx_closest]
    else:
        target_face, target_landmarks = locations[0], landmarks[0]
    top, right, bottom, left = target_face
    x, y, w, h = left, top, right-left, bottom-top
    return x, y, w, h, target_landmarks


def get_crop_around_face(img, target_encodings, aspect_ratio, face_crop, face_crop_lerp):
    global jx0, jy0, jw0, jh0
    ix, iy, iw, ih, ilandmarks = get_face(img, target_encodings)
    if ilandmarks is None:
        return None, None, None, None
    if aspect_ratio > iw/ih:
        jw, jh = ih * aspect_ratio, ih
    else:
        jw, jh = iw, ih / aspect_ratio
    jw, jh = jw / face_crop, jh / face_crop
    jx, jy = ix - 0.5 * (jw - iw), iy - 0.5 * (jh - ih)
    if jx0 is None:
        jx0, jy0, jw0, jh0 = jx, jy, jw, jh
    jx0 = jx0 * (1.0 - face_crop_lerp) + jx * face_crop_lerp
    jy0 = jy0 * (1.0 - face_crop_lerp) + jy * face_crop_lerp
    jw0 = jw0 * (1.0 - face_crop_lerp) + jw * face_crop_lerp
    jh0 = jh0 * (1.0 - face_crop_lerp) + jh * face_crop_lerp
    return jx0, jy0, jw0, jh0
        

def draw_landmarks(img_, landmarks, color, width):
    img = Image.fromarray(np.copy(img_))
    d = ImageDraw.Draw(img, 'RGBA')

    # Make the eyebrows into a nightmare
    whole_face = landmarks['chin'] + list(reversed(landmarks['right_eyebrow'])) + list(reversed(landmarks['left_eyebrow'])) + [landmarks['chin'][0]]
#    d.line(landmarks['left_eyebrow'], fill=color, width=width)
#    d.line(landmarks['right_eyebrow'], fill=color, width=width)
    d.line(landmarks['left_eye'], fill=color, width=width)
    d.line(landmarks['right_eye'], fill=color, width=width)
    d.line(landmarks['top_lip'], fill=color, width=width)
    d.line(landmarks['bottom_lip'], fill=color, width=width)
    d.line(landmarks['nose_bridge'], fill=color, width=width)
    d.line(landmarks['nose_tip'], fill=color, width=width)
#    d.line(landmarks['chin'], fill=color, width=width)
    d.line(whole_face, fill=color, width=width)
    
    return img


def extract_face(img, target_encodings):
    x, y, w, h, landmarks = get_face(img, target_encodings)
    color, width = (255, 255, 255, 255), 1
    blank_img = Image.new('RGB', (img.width, img.height))
    img = draw_landmarks(blank_img, landmarks, color, width)
    return img

