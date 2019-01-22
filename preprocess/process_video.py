import os
import glob
import argparse
import random

import cv2
import dlib
import numpy as np
from tqdm import tqdm

# Basic model parameters as external flags.
FLAGS = None
predictor_path = "/home/karim/Documents/Development/FacialCapture/Facial-Capture/models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def getFaceKeypoints(img, detector, predictor, maxImgSizeForDetection=320):
    imgScale = 1
    scaledImg = img
    if max(img.shape) > maxImgSizeForDetection:
        imgScale = maxImgSizeForDetection / float(max(img.shape))
        scaledImg = cv2.resize(img, (int(img.shape[1] * imgScale), int(img.shape[0] * imgScale)))

    dets = detector(scaledImg, 1)

    if len(dets) == 0:
        return None

    shapes2D = []
    for det in dets:
        faceRectangle = dlib.rectangle(int(det.left() / imgScale), int(det.top() / imgScale), int(det.right() / imgScale), int(det.bottom() / imgScale))
        dlibShape = predictor(img, faceRectangle)
        shape2D = np.array([[p.x, p.y] for p in dlibShape.parts()])
        shape2D = shape2D.T
        shapes2D.append(shape2D)

    return shapes2D


def cropFace(frame, landmarks, size=256, ratio=1.5):
    # add borders to avoid cropping problems
    bordersize=300
    frame = cv2.copyMakeBorder(frame, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT)
    landmarks = landmarks + bordersize

    # crop r.o.i around landmarks
    landmarks = np.asarray(landmarks, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(landmarks)

    mx = (w if w > h else h)
    x = int(x - float(ratio * w - w) / 2)
    y = int((y - float(ratio * h - h) / 2) - float(ratio * h * 0.08))
    mx = int(mx * ratio)

    roi = frame[y:y + mx , x:x + mx]
    h, w, channels = roi.shape


    # If the cropped img is small or face region mx is small
    if h != w or w < 150 or mx < 150:
        # print('Face too small, skipped')
        return None, None

    landmarks[:, 0] = landmarks[:, 0] - x
    landmarks[:, 1] = landmarks[:, 1] - y
    scale_ratio = float(w) / float(size)
    landmarks = landmarks / scale_ratio 
    roi = cv2.resize(roi, (size, size), interpolation = cv2.INTER_LINEAR)

    return roi, landmarks


def drawPoints(img, points, color=(0, 255, 0)):
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 2, color)


def processVideo(video_path, id):
    cap = cv2.VideoCapture(video_path)
    frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_size = FLAGS.num_frames

    # # Random samples
    # if sample_size < frames_num:
    #     samples = random.sample(range(0, frames_num), sample_size)
    # else:
    #     print('Samples are more than number of frames')
    #     return

    # Sequential samples
    if sample_size < frames_num:
        samples = range(0, sample_size)
    else:
        print('Samples are more than number of frames')
        return

    # validation_set = random.sample(range(0, sample_size), int(sample_size * 0.2))

    frame_cnt = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if frame_cnt in samples:
                shape2D = getFaceKeypoints(frame, detector, predictor)
                if shape2D is None:
                    continue

                shape2D = np.asarray(shape2D)[0].T
                frame, shape2D = cropFace(frame, shape2D)
                if frame is None:
                    continue
                fname = str(id) + '_' + str(frame_cnt)
                # if frame_cnt in validation_set:
                #     fname = os.path.join(FLAGS.validation_dir, fname)
                # else:
                #     fname = os.path.join(FLAGS.output_dir, fname)

                fname = os.path.join(FLAGS.output_dir, fname)
                cv2.imwrite(fname + ".jpg", frame)
                np.save(fname + ".npy", shape2D)
                # drawPoints(frame, shape2D)
                # cv2.imshow('frame', frame)
            frame_cnt = frame_cnt + 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break


def main():
    videos = sorted(glob.glob(os.path.join(FLAGS.videos, '*.mp4')))

    video_names = []
    for vid_path in tqdm(videos):
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        processVideo(vid_path, vid_name)
        video_names.append(vid_name)

    video_names = np.asarray(video_names)
    np.savetxt(os.path.join(FLAGS.output_dir, '0_ids.txt'), video_names, fmt='%s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Extract Keyframes for initalization')
    parser.add_argument('--videos', help = 'Path to input videos')
    parser.add_argument('--num_frames', help = 'Number of key frames to extract', type = int, default = 200)
    parser.add_argument('--output_dir', help = 'Output directory', default='.')
    parser.add_argument('--validation_dir', help = 'Output directory', default='.')

    FLAGS, unparsed = parser.parse_known_args()

    main()