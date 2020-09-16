import os
import cv2
import dlib
import time
import argparse
import numpy as np
from random import random
from imutils import video

DOWNSAMPLE_RATIO = 1





def reshape_for_polyline(array):
    return np.array(array, np.int32).reshape((-1, 1, 2))


def main():
    #os.makedirs('original')
    #os.makedirs('landmarks')
        
    cap = cv2.VideoCapture(args.filename)
    fps = video.FPS().start()

    count = 0
    idx_f = 0

    while cap.isOpened():

        start_time = time.time()


        rrr = float(idx_f+0.5) / 5000.0

        frame_no = int(rrr * float(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.set(1, frame_no);
        ret, frame = cap.read()
        idx_f +=1

        if frame is None:
            continue

        if frame.shape[0]==0 or frame.shape[1]==1 or frame.shape[2]==0:
            continue


#        if idx_f % 30 > 0:
#            continue
        
#        print("GO!!!",cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES),cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT),idx_f)
        print("done %0.2f"%(float(100.0*cap.get(cv2.CAP_PROP_POS_FRAMES))/float(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
        #        frame_resize = cv2.resize(frame, None, fx=1.0 / DOWNSAMPLE_RATIO, fy=1.0 / DOWNSAMPLE_RATIO)
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("test.jpg",gray) 

            faces = detector(gray, 1)
            black_image = np.zeros(frame.shape, np.uint8)

            t = time.time()
            #print("len faces ", len(faces))
            # Perform if there is a face detected

            if len(faces) != 3:
                print("No face detected")
                continue

            faces_sorted = sorted(faces, key=lambda f:f.left()+0.5*f.width())
            face = faces_sorted[1]
            facel, facer = faces_sorted[0], faces_sorted[2]

            detected_landmarks = predictor(gray, face).parts()
            landmarks = [[p.x * DOWNSAMPLE_RATIO, p.y * DOWNSAMPLE_RATIO] for p in detected_landmarks]


            black_image = np.zeros(frame.shape, np.uint8)

            whole_face = reshape_for_polyline(landmarks[0:17] + list(reversed(landmarks[22:27])) + list(reversed(landmarks[17:22])))
            jaw = reshape_for_polyline(landmarks[0:17])
            left_eyebrow = reshape_for_polyline(landmarks[22:27])
            right_eyebrow = reshape_for_polyline(landmarks[17:22])
            nose_bridge = reshape_for_polyline(landmarks[27:31])
            lower_nose = reshape_for_polyline(landmarks[30:35])
            left_eye = reshape_for_polyline(landmarks[42:48])
            right_eye = reshape_for_polyline(landmarks[36:42])
            outer_lip = reshape_for_polyline(landmarks[48:60])
            inner_lip = reshape_for_polyline(landmarks[60:68])

            # paint
            cv2.fillPoly(black_image, [whole_face], (255, 255, 255))
            cv2.fillPoly(black_image, [left_eye], (255, 0, 0))
            cv2.fillPoly(black_image, [right_eye], (255, 0, 0))
            cv2.fillPoly(black_image, [lower_nose], (255, 255, 0))
            cv2.fillPoly(black_image, [outer_lip], (0, 0, 255))
            cv2.fillPoly(black_image, [inner_lip], (0, 255, 0))
            cv2.polylines(black_image, [left_eyebrow], False, (255, 0, 255), 4)
            cv2.polylines(black_image, [right_eyebrow], False, (255, 0, 255), 4)
            cv2.polylines(black_image, [nose_bridge], False, (0, 255, 255), 4)


            #color = (255, 255, 255)
            #thickness = 3
            #cv2.polylines(black_image, [jaw], False, color, thickness)
            #cv2.polylines(black_image, [left_eyebrow], False, color, thickness)
            #cv2.polylines(black_image, [right_eyebrow], False, color, thickness)
            #cv2.polylines(black_image, [nose_bridge], False, color, thickness)
            #cv2.polylines(black_image, [lower_nose], True, color, thickness)
            #cv2.polylines(black_image, [left_eye], True, color, thickness)
            #cv2.polylines(black_image, [right_eye], True, color, thickness)
            #cv2.polylines(black_image, [outer_lip], True, color, thickness)
            #cv2.polylines(black_image, [inner_lip], True, color, thickness)


            minp = np.array(landmarks).min(axis=0)
            maxp = np.array(landmarks).max(axis=0)
            #cx, cy = ctr[0], ctr[1]
            cx, cy = 0.5*(minp[0]+maxp[0]), 0.5*(minp[1]+maxp[1])

            min_x, min_y = np.min(np.array(landmarks)[:,0]), np.min(np.array(landmarks)[:,1])
            max_x, max_y = np.max(np.array(landmarks)[:,0]), np.max(np.array(landmarks)[:,1])

            w, h = max_x-min_x, max_y-min_y

            size = max(w, h) * 3.5 * (0.9 + 0.2 * random())
            s2 = size*2

            if size < 100:
                continue

            
            x1, x2 = int(cx - s2/2.0), int(cx + s2/2.0)
            y1, y2 = int(cy - size/2.0), int(cy + size/2.0)




            x1, x2 = facel.left(), facer.left() + facer.width()
            s2 = x2-x1
            size = s2/2.0
            y1, y2 = int(cy - size/2.0), int(cy + size/2.0)

            #margin = 512-size
            #x1m, y1m = x1, y1
            #if margin > 0:
            #    x1m, y1m = int(x1-margin/2), int(y1-margin/2)

            frame = frame[y1:y2, x1:x2, :]
            black_image = black_image[y1:y2, x1:x2, :]


            frame2 = cv2.resize(frame, (1024, 512), interpolation = cv2.INTER_LANCZOS4)
            black_image2 = cv2.resize(black_image, (1024, 512), interpolation = cv2.INTER_LANCZOS4)



            # Display the resulting frame
            count += 1
            
            #cv2.imwrite("original/{}.png".format(count), frame)
            #cv2.imwrite("landmarks/{}.png".format(count), black_image)
            cv2.imwrite("combined/frame%05d.png"%(10000+count), np.concatenate([black_image2, frame2], axis=1))
            fps.update()

            #print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        except:
            print('oops')

        dt = 1.0 - time.time() + start_time
        #if dt > 0:
        #    time.sleep(dt) 

        if count == args.number:  # only take 400 photos
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    cap.release()
#    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', dest='filename', type=str, help='Name of the video file.')
    parser.add_argument('--num', dest='number', type=int, help='Number of train data to be created.')
    parser.add_argument('--landmark-model', dest='face_landmark_shape_file', type=str, help='Face landmark model file.')
    args = parser.parse_args()

    # Create the face predictor and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.face_landmark_shape_file)

    main()
