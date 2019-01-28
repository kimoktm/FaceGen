from __future__ import unicode_literals

import os
import glob
import argparse
import random
import subprocess

import numpy as np

from tqdm import tqdm

FLAGS = None



def processIdentity(path, id):
    urls = [str(dI) for dI in os.listdir(path) if os.path.isdir(os.path.join(path,dI))]
    urls.sort()

    for i in (range(len(urls))):
        vid_name = urls[i]
        vid_path = os.path.join(path, urls[i])
        if os.listdir(vid_path):
            FLAGS.command = FLAGS.command + " -fdir " + vid_path

def main():
    identites = [str(dI) for dI in os.listdir(FLAGS.dataset) if os.path.isdir(os.path.join(FLAGS.dataset,dI))]
    identites.sort()

    identites = identites[1050:]

    # print command
    FLAGS.command = "./FeatureExtraction -pose -2Dfp -out_dir '/home/karim/Documents/Data/VoxCeleb/train/openface_landmarks'"

    for i in tqdm(range(len(identites))):
        path = os.path.join(FLAGS.dataset, identites[i])
        processIdentity(path, identites[i])


    print(FLAGS.command)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Extract Keyframes for initalization')
    parser.add_argument('--dataset', help = 'Path to 300VW videos', required=True)

    FLAGS, unparsed = parser.parse_known_args()

    main()