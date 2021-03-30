from imutils.face_utils import FaceAligner
from PIL import Image, ImageDraw
import numpy as np
import imutils
import dlib
import cv2
import torch
import torchvision.transforms as transforms
import face_recognition

from ..utils import downloads
from ..models import submodules

#with submodules.localimport('submodules/face-parsing.PyTorch') as _importer:
with submodules.import_from('face-parsing-PyTorch'):  # localimport fails here    
    from model import BiSeNet

detector = None
predictor = None
parser = None

parsing_labels = {
    'background': 0, 'skin': 1, 'l_brow': 2, 'r_brow': 3,
    'l_eye': 4, 'r_eye': 5, 'eye_g': 6, 'l_ear': 7, 'r_ear': 8, 
    'ear_r': 9, 'nose': 10, 'mouth': 11, 'u_lip': 12, 'l_lip': 13,
    'neck': 14, 'neck_l': 15, 'cloth': 16, 'hair': 17, 'hat': 18}

parsing_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                  [255, 0, 85], [255, 0, 170],
                  [0, 255, 0], [85, 255, 0], [170, 255, 0],
                  [0, 255, 85], [0, 255, 170],
                  [0, 0, 255], [85, 0, 255], [170, 0, 255],
                  [0, 85, 255], [0, 170, 255],
                  [255, 255, 0], [255, 255, 85], [255, 255, 170],
                  [255, 0, 255], [255, 85, 255], [255, 170, 255],
                  [0, 255, 255], [85, 255, 255], [170, 255, 255]]


def setup_face_detection():
    global detector, predictor
    predictor_file = downloads.download_data_file(
        url='https://github.com/ageitgey/face_recognition_models/raw/master/face_recognition_models/models/shape_predictor_68_face_landmarks.dat', 
        output_path='face_recognition/shape_predictor_68_face_landmarks.dat')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_file)

    
def setup_face_parsing():
    global parser
    parser_file = downloads.download_from_gdrive(
        gdrive_fileid='154JgKpzCPW82qINcVieuPH3fZ2e0P812', 
        output_path='face-parsing/79999_iter.pth')
    n_classes = 19
    parser = BiSeNet(n_classes=n_classes)
    parser.cuda()
    parser.load_state_dict(torch.load(parser_file))
    parser.eval()


def get_parsing_labels():
    return list(parsing_labels.keys())


def align_face(img, 
               face_width=256, 
               resize_width=800):    
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

    
def get_face_parts(parsing, labels):
    labels = labels if isinstance(labels, list) else [labels]
    labels_found = [label in get_parsing_labels() for label in labels]
    assert False not in labels_found, 'Error: labels not understood'
    part_masks = [np.equal(parsing, parsing_labels[label]) for label in labels]
    full_mask = np.sum(part_masks, axis=0)
    full_mask = 255 * np.tile(full_mask[:, :, None], [1, 1, 3])
    return full_mask


def parse_face(img):   
    if not parser:
        setup_face_parsing()
        
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        img = to_tensor(img)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = parser(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        return parsing
    
    
def visualize_parse(parsing, overlay=None):
    w1, w2, stride = 0.4, 0.6, 1
    if overlay is not None:
        vis_im = np.array(overlay).copy().astype(np.uint8)
    else:
        vis_im = np.zeros(parsing.shape).astype(np.uint8)
        w1, w2 = 0, 1
    
    vis_parsing_anno = parsing.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = parsing_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), w1, vis_parsing_anno_color, w2, 0)
    return vis_im
    
    
def get_encodings(target_face_img):
    if not predictor:
        setup_face_detection()
    #target_face_img = face_recognition.load_image_file(filename)
    target_face_img = np.array(target_face_img)
    target_encodings = face_recognition.face_encodings(target_face_img)
    return target_encodings

    
def get_face(img, target_encodings=None):
    if not predictor:
        setup_face_detection()
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

    
def align_face_from_path(img_path, face_width=256, resize_width=800):
    img = Image.open(img_path).convert('RGB') 
    x, face_found = align_face(img, face_width, resize_width)
    return x, face_found


def parse_face_from_path(img_path):
    img = Image.open(img_path).convert('RGB') 
    parse = parse_face(img)
    return parse