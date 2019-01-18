import os
import glob
import argparse
import random

import cv2
import dlib
import numpy as np
import pandas as pd

from tqdm import tqdm

# Basic model parameters as external flags.
FLAGS = None


def getFaceKeypoints(frame_cnt, openFace_landmarks):
    shapes2D = []
    frame = openFace_landmarks[openFace_landmarks['frame'] == frame_cnt]

    # skip if low confidence
    if frame['confidence'].values[0] < 0.95:
        return None

    for i in range(0, 68):
        x = frame[' x_' + str(i)].values[0]
        y = frame[' y_' + str(i)].values[0]
        shapes2D.append([x, y])

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
    landmarks_path = os.path.join(FLAGS.openFace_landmarks, str(id) + '.csv')
    landmarks = pd.read_csv(landmarks_path)


    cap = cv2.VideoCapture(video_path)
    frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_size = FLAGS.num_frames

    # Sequential samples
    if sample_size > frames_num:
        print('Samples are more than number of frames')
        return

    frame_cnt = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if frame_cnt < samples:
                shape2D = getFaceKeypoints(frame_cnt, landmarks)

                if shape2D is None:
                    continue

                shape2D = np.asarray(shape2D)
                frame, shape2D = cropFace(frame, shape2D)
                if frame is None:
                    continue

                fname = str(id) + '_' + str(frame_cnt)
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

    # delete sequence if frame_cnt is too low
    if frame_cnt < 40:
        print('Deleting sequence ' + str(id))
        try:
            os.remove(os.path.join(FLAGS.output_dir, str(id) + '*'))
        except:
            print('Failed to delete sequence ' + str(id))


def main():
    videos = sorted(glob.glob(os.path.join(FLAGS.videos, '*.mp4')))
    # videos = videos[:3]

    for vid_path in tqdm(videos):
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        processVideo(vid_path, vid_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Extract Keyframes for initalization')
    parser.add_argument('--videos', help = 'Path to input videos')
    parser.add_argument('--openFace_landmarks', help = 'Path to openface landmarks')
    parser.add_argument('--num_frames', help = 'Number of key frames to extract', type = int, default = 200)
    parser.add_argument('--output_dir', help = 'Output directory', default='.')
    parser.add_argument('--validation_dir', help = 'Output directory', default='.')

    FLAGS, unparsed = parser.parse_known_args()

    main()