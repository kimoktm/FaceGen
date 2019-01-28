from __future__ import unicode_literals

import os
import glob
import argparse
import random

import cv2
import dlib
import numpy as np

from tqdm import tqdm
import concurrent.futures

FLAGS = None


def checkVidQuality(vid_path):
    cap = cv2.VideoCapture(vid_path)

    avg_quality = 0.0
    frames_cnt = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = np.max(cv2.convertScaleAbs(cv2.Laplacian(gray,3)))
            avg_quality = avg_quality + score
            frames_cnt = frames_cnt + 1
        else:
            break

    FLAGS.avg_frames = FLAGS.avg_frames + frames_cnt

    return avg_quality / frames_cnt


def checkQuality(path, id):
    good_path = os.path.join(FLAGS.output_dir, 'good_quality')
    if not os.path.isdir(good_path):
        os.mkdir(good_path)

    bad_path = os.path.join(FLAGS.output_dir, 'bad_quality')
    if not os.path.isdir(bad_path):
        os.mkdir(bad_path)

    # pick the segment with best video quality
    best_score = 0.0
    best_video = None
    for vid_path in sorted(glob.glob(os.path.join(path, '*.mp4'))):
        score = checkVidQuality(vid_path)
        if score > best_score:
            best_score = score 
            best_video = vid_path

    vid_name = str(id) + '_' + os.path.basename(best_video)
    FLAGS.vids_cnt = FLAGS.vids_cnt + 1

    good_pose = checkVidPose(best_video)
    # if good_pose:
    #     shutil.copy2(vid_path, os.path.join(good_path, vid_name))
    # else:
    #     shutil.copy2(vid_path, os.path.join(bad_path, vid_name))
    if score < 200:
        shutil.copy2(vid_path, os.path.join(bad_path, vid_name))
    else:
        shutil.copy2(vid_path, os.path.join(good_path, vid_name))



def cropFace(img, landmarks):
    landmarks = np.asarray(landmarks, dtype=np.int32)
    hullIndex = cv2.convexHull(landmarks)

    # Get mask by filling triangle
    mask = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype = np.uint8)
    cv2.fillConvexPoly(mask, hullIndex, [1, 1, 1]);
    face_pixels = np.count_nonzero(mask)
    # Apply mask to cropped region
    res = img * mask
    face_mean = np.sum(res) / face_pixels
    # cv2.imshow('IMG', res)
    # k = cv2.waitKey(100)
    # if k == 27:
    #     exit()

    return res, face_mean


import face_recognition
def processFrames(path, id=''):
    #if id != 'oTbZXWMTGhs':
    #    return

    img_mean = -1.0
    face_encoding = []
    frames = sorted(glob.glob(os.path.join(path, '*.jpg')))
    FLAGS.total_frames = FLAGS.total_frames + len(frames)

    correct_frames = 0
    if len(frames) >= FLAGS.minimum_frames:
        for img_path in frames:
            img = cv2.imread(img_path)
            lnd_path = img_path.replace('.jpg', '.npy')
            #landmarks = np.load(lnd_path)
            # _, img_mean = cropFace(img, landmarks)

            if len(face_encoding) < 1:
                enc = face_recognition.face_encodings(img)
                if len(enc) > 0:
                    face_encoding.append(enc[0])
                    correct_frames = correct_frames + 1
                else:
                    os.remove(img_path)
                    os.remove(lnd_path)
            else:
                curr_frame = face_recognition.face_encodings(img)
                if len(curr_frame) == 0:
                    os.remove(img_path)
                    os.remove(lnd_path)
                    continue
                results = face_recognition.compare_faces(face_encoding, curr_frame[0])
                if results[0] == False:
                    os.remove(img_path)
                    os.remove(lnd_path)
                else:
                    correct_frames = correct_frames + 1

    if correct_frames < FLAGS.minimum_frames:
        for img_path in frames:
            lnd_path = img_path.replace('.jpg', '.npy')
            try:
                os.remove(img_path)
                os.remove(lnd_path)
            except OSError:
                pass
            continue

    FLAGS.processed_frames = FLAGS.processed_frames + correct_frames


def processIdentity(path, id):
    #if id != 'id10363':
    #    return

    urls = [str(dI) for dI in os.listdir(path) if os.path.isdir(os.path.join(path,dI))]
    urls.sort()

    vid_paths = []
    for i in (range(len(urls))):
        #processFrames(os.path.join(path, urls[i]), urls[i])
        vid_paths.append(os.path.join(path, urls[i])) 

    with concurrent.futures.ThreadPoolExecutor() as executor:
    #    for vid_path in vid_paths:
        executor.map(processFrames, vid_paths)



def main():
    identites = [str(dI) for dI in os.listdir(FLAGS.dataset) if os.path.isdir(os.path.join(FLAGS.dataset,dI))]
    identites.sort()

    identites = identites[8:]

    FLAGS.processed_frames = 0
    FLAGS.total_frames = 0
    print("Process Identites set:")
    for i in tqdm(range(len(identites))):
        path = os.path.join(FLAGS.dataset, identites[i])
        processIdentity(path, identites[i])


    print("Total frames:")
    print(FLAGS.total_frames)
    print("Processed frames:")
    print(FLAGS.processed_frames)
    # Processed Videos: 
    # 15438


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Extract Keyframes for initalization')
    parser.add_argument('--dataset', help = 'Path to 300VW videos', required=True)
    parser.add_argument('--minimum_frames', help = 'Images per video', default=60)

    FLAGS, unparsed = parser.parse_known_args()

    main()