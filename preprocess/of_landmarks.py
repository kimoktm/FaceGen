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


def drawPoints(img, points, color=(255, 0, 0)):
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 2, color)


def getFaceKeypoints(openFace_landmarks_path):
    try:
        openFace_landmarks = pd.read_csv(openFace_landmarks_path)

        shapes2D = []
        frame = openFace_landmarks[openFace_landmarks['face'] == 0]

        # skip if low confidence
        if frame[' confidence'].values[0] <= 0.93:
            return None

        for i in range(0, 68):
            x = frame[' x_' + str(i)].values[0]
            y = frame[' y_' + str(i)].values[0]
            shapes2D.append([x, y])

        shapes2D = np.asarray(shapes2D)

        return shapes2D
    except:
        return None



def processFrames(path):
    # if id != '1zcIwhmdeo4':
    #    return

    frames = sorted(glob.glob(os.path.join(path, '*.jpg')))
    FLAGS.total_frames = len(frames)

    correct_frames = 0
    if len(frames) >= FLAGS.minimum_frames:
        for img_path in tqdm(frames):

            landmarks_path = os.path.basename(img_path).replace('.jpg', '.csv')
            landmarks_path = os.path.join(FLAGS.openFace_landmarks, landmarks_path)

            openFace_landmarks = getFaceKeypoints(landmarks_path)

            if openFace_landmarks is None:
                continue


            # save to output_dir
            lnd_name = os.path.basename(landmarks_path).replace('.csv', '.npy')
            np.save(os.path.join(FLAGS.output_dir, lnd_name), openFace_landmarks)
            
            #img = cv2.imread(img_path)
            #drawPoints(img, openFace_landmarks)
            #cv2.imwrite(os.path.join(FLAGS.output_dir, os.path.basename(img_path)), img)
            shutil.copy2(img_path, os.path.join(FLAGS.output_dir, os.path.basename(img_path)))

            correct_frames = correct_frames + 1

            #if correct_frames >= FLAGS.minimum_frames:
            #    break


    FLAGS.processed_frames = correct_frames
    # # delete sequence if samples_cnt is too low
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



def main():

    processFrames(FLAGS.dataset)

    print("Total frames:")
    print(FLAGS.total_frames)
    print("Processed frames:")
    print(FLAGS.processed_frames)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Extract Keyframes for initalization')
    parser.add_argument('--dataset', help = 'Path to 300VW videos', required=True)
    parser.add_argument('--minimum_frames', help = 'Images per video', default=50)
    parser.add_argument('--openFace_landmarks', help = 'Path to openface landmarks')
    parser.add_argument('--output_dir', help = 'Output directory', default='.')

    FLAGS, unparsed = parser.parse_known_args()

    main()