import os
import glob
import argparse
import random

import cv2
import dlib
import numpy as np
from tqdm import tqdm


FLAGS = None


def getFaceKeypoints(frame_cnt, landmarks_dir):
    frame_path = os.path.join(landmarks_dir, format(frame_cnt, '06d') + ".pts")
    
    file = open(frame_path, "r") 
    file.readline()
    file.readline()
    file.readline()

    shapes2D = []
    for i in range(0, 68):
        nums = file.readline().split(" ")
        x = float(nums[0])
        y = float(nums[1])
        shapes2D.append([x, y])

    return shapes2D


def drawPoints(img, points, color=(0, 255, 0)):
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 2, color)


def cropFace(frame, landmarks, size=256, ratio=2):
    landmarks = np.asarray(landmarks, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(landmarks)

    # scale roi
    mx = (w if w > h else h)
    x = int(x - mx / ratio)
    y = int(y - mx / ratio / 0.8)
    w = int(mx * ratio)
    h = int(mx * ratio)

    roi = frame[y:y + h, x:x + w]
    h, w, channels = roi.shape

    # If the cropped img is small or face region mx is small
    if h != w or w < 200 or mx < 120:
        return None, None

    landmarks[:, 0] = landmarks[:, 0] - x
    landmarks[:, 1] = landmarks[:, 1] - y
    scale_ratio = float(w) / float(size)
    landmarks = landmarks / scale_ratio 
    roi = cv2.resize(roi, (size, size), interpolation = cv2.INTER_CUBIC)

    return roi, landmarks


def processVideo(path, landmarks, sample_size, output_dir):
    cap = cv2.VideoCapture(path)
    frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if sample_size < frames_num:
        samples = random.sample(range(1, frames_num+1), sample_size)
    else:
        samples = range(1, frames_num+1)

    frame_cnt = 1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            if frame_cnt in samples:
                shape2D = getFaceKeypoints(frame_cnt, landmarks)
                shape2D = np.asarray(shape2D, dtype=int)

                frame, lnd = cropFace(frame, shape2D)
                if frame is not None:
                    #drawPoints(frame, lnd)
                    fname = 'img_' + str(FLAGS.current_frame)
                    cv2.imwrite(os.path.join(output_dir, fname + ".png"), frame)
                    np.save(os.path.join(output_dir, fname + ".npy"), lnd)
                    FLAGS.current_frame = FLAGS.current_frame + 1
            frame_cnt = frame_cnt + 1
        else:
            break


def main():

    videos = [dI for dI in os.listdir(FLAGS.dataset) if os.path.isdir(os.path.join(FLAGS.dataset,dI))]
    samples = np.asarray(random.sample(range(0, len(videos)), FLAGS.videos_num), dtype=int)
    videos = np.asarray(videos)[samples]

    validation_set = ['114', '124', '125', '126', '150', '158', '401', '402', '505', '506', '507', '508', '509', '510', '511', '514', '515']


    FLAGS.current_frame = 0
    print("Training set:")
    for i in tqdm(range(len(videos))):
        if videos[i] not in validation_set:
            cur_path = os.path.join(FLAGS.dataset, videos[i])
            vid_path = os.path.join(cur_path, 'vid.avi')
            lnd_path = os.path.join(cur_path, 'annot')
            processVideo(vid_path, lnd_path, FLAGS.imgs_per_video, FLAGS.output_dir)

    FLAGS.current_frame = 0
    print("Validation set:")
    for i in tqdm(range(len(validation_set))):
        cur_path = os.path.join(FLAGS.dataset, validation_set[i])
        vid_path = os.path.join(cur_path, 'vid.avi')
        lnd_path = os.path.join(cur_path, 'annot')
        processVideo(vid_path, lnd_path, FLAGS.imgs_per_video, FLAGS.validation_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Extract Keyframes for initalization')
    parser.add_argument('--dataset', help = 'Path to 300VW videos')
    parser.add_argument('--output_dir', help = 'Output directory')
    parser.add_argument('--validation_dir', help = 'Output directory')
    parser.add_argument('--imgs_per_video', help = 'Images per video', default=1000)
    parser.add_argument('--videos_num', help = 'Number of videos', default=101)

    FLAGS, unparsed = parser.parse_known_args()

    main()