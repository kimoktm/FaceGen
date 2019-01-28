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


def contrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final

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
        dlibShape = predictor(contrast(img), faceRectangle)
        shape2D = np.array([[p.x, p.y] for p in dlibShape.parts()])
        shape2D = shape2D.T
        shapes2D.append(shape2D)

    return shapes2D


def cropLandmarks(frame, size=256, ratio=1.5):
    landmarks = getFaceKeypoints(frame, detector, predictor)
    if landmarks is None:
        # print("Skipped in landmarks detection")
        return None, None
    landmarks = np.asarray(landmarks, dtype=np.int32)[0].T

    # add borders to avoid cropping problems
    bordersize=400
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
    if h != w or w < 160 or mx < 160:
        # print("Skipped in Key Crop")
        return None, None

    landmarks[:, 0] = landmarks[:, 0] - x
    landmarks[:, 1] = landmarks[:, 1] - y
    scale_ratio = float(w) / float(size)
    landmarks = landmarks / scale_ratio 
    roi = cv2.resize(roi, (size, size), interpolation = cv2.INTER_LINEAR)

    return roi, landmarks


def cropFace(frame, bounding_box, ratio=2.6):
    # add borders to avoid cropping problems
    bordersize = 900
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
    if h != w or w < 128 or mx < 128:
        print("Skipped in HOG Crop")
        return None

    return roi


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
        x = float(nums[1])
        y = float(nums[2])
        w = float(nums[3])
        h = float(nums[4])
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
    bounding_rects = np.asarray(bounding_rects, dtype=float)

    return selected_frames, bounding_rects


def processVideo(vid_path, selected_frames_path, id):
    FLAGS.processed_videos = FLAGS.processed_videos + 1

    # create dir
    output_path = os.path.join(FLAGS.frames_path, os.path.relpath(selected_frames_path, FLAGS.dataset))
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    selected_frames, bounding_rects = processSelectedFrames(selected_frames_path)
    cap = cv2.VideoCapture(vid_path)

    if selected_frames.size > FLAGS.imgs_per_video:
        selected_frames = selected_frames[:FLAGS.imgs_per_video]
        bounding_rects = bounding_rects[:FLAGS.imgs_per_video]
    else:
        return

    # decrease frame-rate to 24
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    fps_skip = round(fps / (fps - 24))

    frame_cnt = 0
    id_tracker = 0
    low_fps_cnt = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame_cnt = frame_cnt + 1
            if frame_cnt % fps_skip == 0:
                continue

            scale = float(frame.shape[0] / 360.0)
            if id_tracker < len(selected_frames) and low_fps_cnt == selected_frames[id_tracker]:
                frame = cropFace(frame, bounding_rects[id_tracker] * scale)
                frame, landmarks = cropLandmarks(frame)
                if frame is not None:
                    fname = str(id) + '_' + str(low_fps_cnt)
                    cv2.imwrite(os.path.join(output_path, fname + ".jpg"), frame)
                    np.save(os.path.join(output_path, fname + ".npy"), landmarks)                    
                    # drawPoints(frame, landmarks)
                    # cv2.imshow('frame', frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                id_tracker = id_tracker + 1
            low_fps_cnt = low_fps_cnt + 1
        else:
            break


def processURL(path, id):
    vid_path = os.path.join(FLAGS.videos_path, os.path.relpath(path, FLAGS.dataset))
    vid_path = os.path.join(vid_path, str(id) + '.mp4')

    class MyLogger(object):
        def debug(self, msg):
            pass
        def warning(self, msg):
            pass
        def error(self, msg):
            pass


    ydl_opts = {
        'format': '(mp4)[height>=480][height<=?1080]',
        'outtmpl': vid_path,
        'logger': MyLogger()
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download(['http://youtube.com/watch?v=' + str(id)])
            processVideo(vid_path, path, id)
    except:
        # print("Resolution too low, skipping")
        return


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

    # Processed Videos: 
    # 15438
    print('Processed Videos: ')
    print(FLAGS.processed_videos)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Extract Keyframes for initalization')
    parser.add_argument('--dataset', help = 'Path to 300VW videos', required=True)
    parser.add_argument('--output_dir', help = 'Output directory')
    parser.add_argument('--validation_dir', help = 'Output directory')
    parser.add_argument('--imgs_per_video', help = 'Images per video', default=200)
    parser.add_argument('--videos_num', help = 'Number of videos', default=101)

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.videos_path = os.path.join(FLAGS.output_dir, 'videos')
    FLAGS.frames_path = os.path.join(FLAGS.output_dir, 'frames')
    if not os.path.isdir(FLAGS.videos_path):
        os.makedirs(FLAGS.videos_path)
    if not os.path.isdir(FLAGS.frames_path):
        os.makedirs(FLAGS.frames_path)

    main()