import os
import subprocess
import numpy as np
import PIL
import cv2
import torch
from tqdm import tqdm

from .. import image
from ..utils import downloads
from . import submodules

with submodules.import_from('Wav2Lip'):
    import audio
    import face_detection
    from models import Wav2Lip


model = None


def setup():
    global model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = downloads.download_from_gdrive(
        gdrive_fileid='1_OvqStxNxLc7bXzlaVG5sz695p-FVfYY',
        output_path='Wav2Lip/wav2lip_gan.pth')
    model = load_model(checkpoint_path, device)


def __load__(checkpoint_path, device):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path, device):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = __load__(path, device)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    model = model.to(device)
    return model.eval()


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images, pads, nosmooth, batch_size, device):
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, 
        flip_input=False, 
        device=device)

    while True:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = pads
    for rect, image in zip(predictions, images):
        if rect is None:
            #cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results 


def datagen(frames, mels, box, static, img_size, wav2lip_batch_size, face_det_batch_size, pads, nosmooth, device):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if box[0] == -1:
        if not static:
            face_det_results = face_detect(frames, pads, nosmooth, face_det_batch_size, device) # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]], pads, nosmooth, face_det_batch_size, device)
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if static else i%len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (img_size, img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch


def modify_frame(frame, resize_factor, rotate, crop):
    frame = np.array(frame)
    if resize_factor > 1:
        new_size = (frame.shape[1]//resize_factor, frame.shape[0]//resize_factor)
        frame = cv2.resize(frame, new_size)
    if rotate:
        frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
    y1, y2, x1, x2 = crop
    if x2 == -1: x2 = frame.shape[1]
    if y2 == -1: y2 = frame.shape[0]
    frame = frame[y1:y2, x1:x2]
    return frame


def run(input_video, input_audio, output_video, pads=None, resize_factor=1, crop=None, box=None, fps=25, rotate=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_size = 96
    mel_step_size = 16    
    static = False
    nosmooth = False
    wav2lip_batch_size = 128
    face_det_batch_size = 16

    if pads is None:
        pads = [0, 10, 0, 0]
    if crop is None:
        crop = [0, -1, 0, -1]
    if box is None:
        box = [-1, -1, -1, -1]
    
    input_is_image = isinstance(input_video, (PIL.Image.Image, np.ndarray))
    input_is_image_list = isinstance(input_video, list) and isinstance(input_video[0], (PIL.Image.Image, np.ndarray))
    input_is_str = isinstance(input_video, str)
    input_is_movieplayer = isinstance(input_video, image.MoviePlayer)
    
    if input_is_str:        
        if not os.path.isfile(input_video):
            raise ValueError('No image or video found at {}'.format(input_video))

        elif input_video.split('.')[1] in ['jpg', 'png', 'jpeg']:
            full_frames = [cv2.imread(input_video)]

        else:
            video_stream = cv2.VideoCapture(input_video)
            fps = video_stream.get(cv2.CAP_PROP_FPS)
            full_frames = []
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                frame = modify_frame(frame, resize_factor, rotate, crop)
                full_frames.append(frame)
    
    elif input_is_image_list:
        full_frames = [np.array(img)[...,[2,1,0]] for img in input_video]
        
    elif input_is_image:
        full_frames = [np.array(input_video)[...,[2,1,0]]]
    
    elif input_is_movieplayer:
        full_frames = []
        for f in range(input_video.num_frames):
            frame = input_video.get_frame(f+1)
            frame = modify_frame(frame, resize_factor, rotate, crop)
            full_frames.append(frame)

    print ("Number of frames available for inference: "+str(len(full_frames)))

    scratch_folder = downloads.get_ml4a_folder('scratch/wav2lip')
    temp_video_file = os.path.join(scratch_folder, 'temp_video.avi')
    temp_audio_file = os.path.join(scratch_folder, 'temp_audio.wav')

    if not input_audio.endswith('.wav'):
        print('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(input_audio, temp_audio_file)
        subprocess.call(command, shell=True)
        input_audio = temp_audio_file

    wav = audio.load_wav(input_audio, 16000)
    mel = audio.melspectrogram(wav)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80./fps 
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[:len(mel_chunks)]
    gen = datagen(full_frames.copy(), mel_chunks,
                  box, static, img_size, 
                  wav2lip_batch_size, face_det_batch_size,
                  pads, nosmooth, device)

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                            total=int(np.ceil(float(len(mel_chunks))/wav2lip_batch_size)))):
        if model is None:
            setup()

        
        if i == 0:
            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter(temp_video_file,
                                  cv2.VideoWriter_fourcc(*'DIVX'), 
                                  fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()
    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(input_audio, temp_video_file, output_video)
    subprocess.call(command, shell=True)