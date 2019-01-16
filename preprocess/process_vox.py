from __future__ import unicode_literals

import os
import glob
import argparse
import random

import cv2
import dlib
import numpy as np

import youtube_dl
from tqdm import tqdm

detector = dlib.get_frontal_face_detector()
predictor_path = "/home/karim/Documents/Development/FacialCapture/Facial-Capture/models/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
FLAGS = None


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


def cropLandmarks(frame, landmarks, size=256, ratio=1.8):
    # add borders to avoid cropping problems
    bordersize=300
    frame = cv2.copyMakeBorder(frame, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT)
    landmarks = landmarks + bordersize

    # crop r.o.i around landmarks
    landmarks = np.asarray(landmarks, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(landmarks)

    mx = (w if w > h else h)
    x = int(x - float(ratio * w - w) / 2)
    y = int(y - float(ratio * h - h) / 2)
    mx = int(mx * ratio)

    roi = frame[y:y + mx , x:x + mx]
    h, w, channels = roi.shape

    # If the cropped img is small or face region mx is small
    if h != w or w < 120 or mx < 120:
        print("Skipped in Key Crop")
        return None, None

    landmarks[:, 0] = landmarks[:, 0] - x
    landmarks[:, 1] = landmarks[:, 1] - y
    scale_ratio = float(w) / float(size)
    landmarks = landmarks / scale_ratio 
    roi = cv2.resize(roi, (size, size), interpolation = cv2.INTER_LINEAR)

    return roi, landmarks


def cropFace(frame, bounding_box, size=256, ratio=2):
    # add borders to avoid cropping problems
    bordersize = 400
    frame = cv2.copyMakeBorder(frame, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT)

    # crop r.o.i
    x = bounding_box[0] + bordersize
    y = bounding_box[1] + bordersize
    w = bounding_box[2]
    h = bounding_box[3]

    mx = (w if w > h else h)
    x = int(x - float(ratio * w - w) / 2)
    y = int(y - float(ratio * h - h) / 2)
    mx = int(mx * ratio)

    roi = frame[y:y + mx , x:x + mx]
    h, w, channels = roi.shape

    # If the cropped img is small or face region mx is small
    if h != w or w < 120 or mx < 120:
        print("Skipped in HOG Crop")
        return None, None

    landmarks = getFaceKeypoints(roi, detector, predictor)
    if landmarks is None:
        print("Skipped in keypoints")
        return None, None

    landmarks = np.asarray(landmarks, dtype=np.int32)[0].T

    return cropLandmarks(roi, landmarks, size)


def drawPoints(img, points, color=(0, 255, 0)):
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 2, color)
            

def prcoessTxt(file_path):
    file = open(file_path, "r")
    for i in range(7):
        file.readline()

    selected_frames = []
    bounding_rects = []
    for line in file:
        nums = line.split(" ")
        frame_id = int(nums[0])
        x = int(nums[1])
        y = int(nums[2])
        w = int(nums[3])
        h = int(nums[4])
        selected_frames.append(frame_id)
        bounding_rects.append([x, y, w, h])

    return selected_frames, bounding_rects


def processSelectedFrames(path):
    selected_frames = []
    bounding_rects = []

    for file in sorted(glob.glob(os.path.join(path, '*.txt'))):
        frames, boxes = prcoessTxt(file)
        selected_frames.append(frames)
        bounding_rects.append(boxes)

    selected_frames = [item for sublist in selected_frames for item in sublist]
    bounding_rects  = [item for sublist in bounding_rects for item in sublist]
    
    # sort
    bounding_rects = [x for _,x in sorted(zip(selected_frames, bounding_rects))]
    selected_frames.sort()

    # remove duplicates
    prev = -1
    s_frames = []
    b_rects  = []
    for i in range(len(selected_frames)):
        if selected_frames[i] != prev:
            s_frames.append(selected_frames[i])
            b_rects.append(bounding_rects[i])
            prev = selected_frames[i]
    selected_frames = s_frames
    bounding_rects = b_rects

    selected_frames = np.asarray(selected_frames, dtype=np.int32)
    bounding_rects = np.asarray(bounding_rects, dtype=np.int32)

    return selected_frames, bounding_rects


def processVideo(vid_path, selected_frames_path, id):
    FLAGS.processed_videos = FLAGS.processed_videos + 1

    # create dir
    output_path = os.path.join(FLAGS.output_dir, str(id))
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    selected_frames, bounding_rects = processSelectedFrames(selected_frames_path)
    cap = cv2.VideoCapture(vid_path)

    if selected_frames.size > FLAGS.imgs_per_video:
        selected_frames = selected_frames[:FLAGS.imgs_per_video]
        bounding_rects = bounding_rects[:FLAGS.imgs_per_video]

    frame_cnt = 0
    id_tracker = 0
    FLAGS.current_frame = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            scale = int(frame.shape[0] / 360)

            if id_tracker < len(selected_frames) and frame_cnt == selected_frames[id_tracker]:
                bounding_rects[id_tracker] = bounding_rects[id_tracker] * scale
                frame, landmarks = cropFace(frame, bounding_rects[id_tracker])
                if frame is not None:
                    fname = 'img_' + str(FLAGS.current_frame)
                    cv2.imwrite(os.path.join(output_path, fname + ".png"), frame)
                    np.save(os.path.join(output_path, fname + ".npy"), landmarks)                    
                    FLAGS.current_frame = FLAGS.current_frame + 1
                    # drawPoints(frame, landmarks)
                    # cv2.imshow('frame', frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                id_tracker = id_tracker + 1
            frame_cnt = frame_cnt + 1
        else:
            break

def processURL(path, id):
    class MyLogger(object):
        def debug(self, msg):
            pass
        def warning(self, msg):
            pass
        def error(self, msg):
            pass

    ydl_opts = {
        'format': '(mp4)[height>=480][height<=?1080]',
        'logger': MyLogger(),
        'outtmpl': str(id) + '.mp4'
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download(['http://youtube.com/watch?v=' + str(id)])
    except:
        # print("Resolution too low, skipping")
        return

    processVideo(str(id) + '.mp4', path, id)
    os.remove(str(id) + '.mp4')


def processIdentity(path, id):
    urls = [str(dI) for dI in os.listdir(path) if os.path.isdir(os.path.join(path,dI))]
    urls.sort()

    for i in (range(len(urls))):
        processURL(os.path.join(path, urls[i]), urls[i])


def main():
    identites = [str(dI) for dI in os.listdir(FLAGS.dataset) if os.path.isdir(os.path.join(FLAGS.dataset,dI))]
    identites.sort()

    FLAGS.processed_videos = 0

    print("Process Identites set:")
    for i in tqdm(range(len(identites))):
        path = os.path.join(FLAGS.dataset, identites[i])
        processIdentity(path, identites[i])

    print('Processed Videos: ')
    print(FLAGS.processed_videos)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Extract Keyframes for initalization')
    parser.add_argument('--dataset', help = 'Path to 300VW videos', required=True)
    parser.add_argument('--output_dir', help = 'Output directory')
    parser.add_argument('--validation_dir', help = 'Output directory')
    parser.add_argument('--imgs_per_video', help = 'Images per video', default=300)
    parser.add_argument('--videos_num', help = 'Number of videos', default=101)

    FLAGS, unparsed = parser.parse_known_args()

    main()