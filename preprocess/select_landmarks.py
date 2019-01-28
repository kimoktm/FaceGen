import os
import glob
import argparse
import random

import cv2
import dlib
import numpy as np
import pandas as pd
import shutil

from tqdm import tqdm

FLAGS = None


def drawPoints(img, points, color=(0, 255, 0)):
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 2, color)


def getFaceKeypoints(frame_id, openFace_landmarks):
    shapes2D = []
    frame = openFace_landmarks[openFace_landmarks['frame'] == frame_id]

    # skip if low confidence
    if frame[' confidence'].values[0] <= 0.93:
        return None

    #if frame[' pose_Rz'].values[0] <= 0.785398 and frame[' pose_Rz'].values[0] >= -0.785398:
    #    return None

    for i in range(0, 68):
        x = frame[' x_' + str(i)].values[0]
        y = frame[' y_' + str(i)].values[0]
        shapes2D.append([x, y])

    shapes2D = np.asarray(shapes2D)

    return shapes2D


def compareLandmarks(dlib_landmarks, openFace_landmarks):
    # Dlib is good in the face area and bad in strong rotations & vice verse for openface
    # Check contour landmarks if they don't align take openFace else dlib
    dlib_left_contour  = dlib_landmarks[:7][:, 0]
    dlib_right_contour = dlib_landmarks[10:17][:, 0]
    
    oface_left_contour  = openFace_landmarks[:7][:, 0]
    oface_right_contour = openFace_landmarks[10:17][:, 0]

    left_contour_distance  = np.linalg.norm(dlib_left_contour - oface_left_contour)
    right_contour_distance = np.linalg.norm(dlib_right_contour - oface_right_contour)

    # print(left_contour_distance)
    # print(right_contour_distance)
    # print()

    MIN_DISTANCE = 20
    if left_contour_distance < MIN_DISTANCE and right_contour_distance < MIN_DISTANCE:
        return dlib_landmarks

    return openFace_landmarks


def processFrames(path, id):
    # if id != '1zcIwhmdeo4':
    #    return

    frames = sorted(glob.glob(os.path.join(path, '*.jpg')))
    FLAGS.total_frames = FLAGS.total_frames + len(frames)

    correct_frames = 0
    if len(frames) >= FLAGS.minimum_frames:
        # looad openFace landmarks
        openFace_landmarks_path = os.path.join(FLAGS.openFace_landmarks, str(id) + '.csv')
        all_openFace_landmarks = pd.read_csv(openFace_landmarks_path)


        for i, img_path in enumerate(frames):
            lnd_path           = img_path.replace('.jpg', '.npy')
            #dlib_landmarks     = np.load(lnd_path)
            openFace_landmarks = getFaceKeypoints(i + 1, all_openFace_landmarks)

            if openFace_landmarks is None:
                continue

            #final_landmarks = compareLandmarks(dlib_landmarks, openFace_landmarks)
            final_landmarks = openFace_landmarks

            # save to output_dir
            lnd_name = os.path.basename(lnd_path)
            np.save(os.path.join(FLAGS.output_dir, lnd_name), final_landmarks)
            shutil.copy2(img_path, os.path.join(FLAGS.output_dir, os.path.basename(img_path)))

            correct_frames = correct_frames + 1

            if correct_frames >= FLAGS.minimum_frames:
                break


    # delete sequence if samples_cnt is too low
    # if correct_frames < FLAGS.minimum_frames:
    #     print('Deleting sequence ' + str(id) + ', good frames: ' + str(correct_frames))
    #     try:
    #         map(os.remove, glob.glob(os.path.join(FLAGS.output_dir, str(id) + '*')))
    #         return False
    #     except:
    #         print('Failed to delete sequence ' + str(id))

    # else:
    #     FLAGS.processed_frames = FLAGS.processed_frames + correct_frames
    #     return True



def processIdentity(path, id):
    # if id != 'id10001':
    #    return

    urls = [str(dI) for dI in os.listdir(path) if os.path.isdir(os.path.join(path,dI))]
    urls.sort()

    vid_paths = []
    for i in (range(len(urls))):
        if (processFrames(os.path.join(path, urls[i]), urls[i])):
            FLAGS.video_names.append(urls[i])


def main():
    identites = [str(dI) for dI in os.listdir(FLAGS.dataset) if os.path.isdir(os.path.join(FLAGS.dataset,dI))]
    identites.sort()

    validation_size = 100 # identities
    validation_list = random.sample(range(0, len(identites)), validation_size) 
    

    identites = identites[:20]

    FLAGS.video_names = []
    FLAGS.processed_frames = 0
    FLAGS.total_frames = 0
    print("Process Identites set:")
    for i in tqdm(range(len(identites))):
        path = os.path.join(FLAGS.dataset, identites[i])
        processIdentity(path, identites[i])

    FLAGS.video_names = np.asarray(FLAGS.video_names)
    np.savetxt(os.path.join(FLAGS.output_dir, '0_ids.txt'), FLAGS.video_names, fmt='%s')


    print("Total frames:")
    print(FLAGS.total_frames)
    print("Processed frames:")
    print(FLAGS.processed_frames)
    # Processed Videos: 
    # 15438


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Extract Keyframes for initalization')
    parser.add_argument('--dataset', help = 'Path to 300VW videos', required=True)
    parser.add_argument('--minimum_frames', help = 'Images per video', default=30)
    parser.add_argument('--openFace_landmarks', help = 'Path to openface landmarks')
    parser.add_argument('--output_dir', help = 'Output directory', default='.')

    FLAGS, unparsed = parser.parse_known_args()

    main()