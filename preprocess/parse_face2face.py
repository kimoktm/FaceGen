import os
import glob
import argparse
import random
import math

import cv2
import dlib
import numpy as np

from tqdm import tqdm

FLAGS = None





# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])


def parseId(path):
    file = open(path, "r")
    for i in range(2):
        file.readline()

    # pose
    pose  = np.zeros([4, 4], dtype=np.float32)
    for i in range(4):
        line = file.readline()
        nums = [float(s) for s in line.split('\t')]
        pose[i] = nums

    # shape
    file.readline()
    line = file.readline()
    nums = [float(s) for s in line.split('\t')[:-1]]
    shape = np.asarray(nums, dtype=np.float32)

    # tex
    line = file.readline()
    nums = [float(s) for s in line.split('\t')[:-1]]
    tex  = np.asarray(nums, dtype=np.float32)

    # exp
    file.readline()
    line = file.readline()
    nums = [float(s) for s in line.split('\t')[:-1]]
    exp  = np.asarray(nums, dtype=np.float32)

    return pose, shape, tex, exp


def parseTracking(path):
    file = open(path, "r")
    
    frame_ids = []
    poses     = []
    exps      = []
    shs       = []


    while True:
        try:
            frame_num = int(file.readline())
            frame_ids.append(frame_num)

            # pose
            pose  = np.zeros([4, 4], dtype=np.float32)
            for i in range(4):
                line = file.readline()
                nums = [float(s) for s in line.split('\t')]
                pose[i] = nums
            poses.append(pose)

            # exp
            line = file.readline()
            nums = [float(s) for s in line.split('\t')[:-1]]
            exp  = np.asarray(nums, dtype=np.float32)
            exps.append(exp)

            # SH
            line = file.readline()
            nums = [float(s) for s in line.split('\t')[:-1]]
            sh  = np.asarray(nums, dtype=np.float32)
            shs.append(sh)
            file.readline()
        except:
            break

    frame_ids = np.asarray(frame_ids)
    poses     = np.asarray(poses)
    exps      = np.asarray(exps)
    shs       = np.asarray(shs)

    return frame_ids, poses, exps, shs



def processIdentity(path, id):
    id_path  = os.path.join(path, 'recording.id')
    trk_path = os.path.join(path, 'recording.trck')

    intrinsics, shape, tex, _ =  parseId(id_path)
    intrinsics = np.transpose(intrinsics)

    frame_ids, poses, exps, shs = parseTracking(trk_path)

    width  = 640
    height = 480

    for i in range(len(frame_ids)):
        frame_id    = frame_ids[i]
        frame_pose  = np.transpose(poses[i])
        frame_exps  = exps[i]
        frame_sh    = shs[i]
        frame_shape = shape
        frame_text  = tex

        # pose, shape, exp, tex, shs
        proj_mat        = np.matmul(intrinsics, frame_pose)
        # print(intrinsics)
        # print(frame_pose)
        # print(proj_mat)
        translation     = proj_mat[:2, 3]
        translation[1]  = translation[1] + height
        rotation_matrix = frame_pose[:3, :3]
        euler_angles    = rotationMatrixToEulerAngles(rotation_matrix)

        print(translation)
        print(euler_angles)
        exit()



def main():
    processIdentity('./', 'ids')

    # identites = [str(dI) for dI in os.listdir(FLAGS.dataset) if os.path.isdir(os.path.join(FLAGS.dataset,dI))]
    # identites.sort()

    # print("Process Identites set:")
    # for i in tqdm(range(len(identites))):
    #     path = os.path.join(FLAGS.dataset, identites[i])
    #     processIdentity(path, identites[i])



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Extract Keyframes for initalization')
    parser.add_argument('--dataset', help = 'Path to 300VW videos', required=False)
    parser.add_argument('--output_dir', help = 'Images per video', default=60)

    FLAGS, unparsed = parser.parse_known_args()

    main()