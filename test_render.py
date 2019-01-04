from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import numpy as np
import tensorflow as tf

import cv2
import argparse
from tqdm import tqdm
import glob as glob

import tf_mesh_renderer.mesh_renderer as mesh_renderer
import tf_mesh_renderer.camera_utils as camera_utils

from bfm.morphable_model import MorphabelModel
from bfm.morphable_model_np import MorphabelModelNP



ARGS = None

def get_normal(vertices, triangles):
    ''' calculate normal direction in each vertex
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
    Returns:
        normal: [nver, 3]
    '''
    pt0 = vertices[triangles[:, 0], :] # [ntri, 3]
    pt1 = vertices[triangles[:, 1], :] # [ntri, 3]
    pt2 = vertices[triangles[:, 2], :] # [ntri, 3]
    tri_normal = np.cross(pt0 - pt1, pt0 - pt2) # [ntri, 3]. normal of each triangle

    normal = np.zeros_like(vertices) # [nver, 3]
    for i in range(triangles.shape[0]):
        normal[triangles[i, 0], :] = normal[triangles[i, 0], :] + tri_normal[i, :]
        normal[triangles[i, 1], :] = normal[triangles[i, 1], :] + tri_normal[i, :]
        normal[triangles[i, 2], :] = normal[triangles[i, 2], :] + tri_normal[i, :]
    
    # normalize to unit length
    mag = np.sum(normal**2, 1) # [nver]
    zero_ind = (mag == 0)
    mag[zero_ind] = 1;
    normal[zero_ind, 0] = np.ones((np.sum(zero_ind)))

    normal = normal/np.sqrt(mag[:,np.newaxis])

    return normal


def renderFace(params, bfm, face_normals, image_height = 480, image_width = 640):
  # camera position:
  eye = tf.constant([[0.0, 0.0, -6.0]], dtype=tf.float32)
  center = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)
  world_up = tf.constant([[0.0, -1.0, 0.0]], dtype=tf.float32)
  light_positions = tf.reshape(eye, [1, 1, 3])
  light_intensities = tf.ones([1, 1, 3], dtype=tf.float32)

  # variables
  ep   = params[: ARGS.exp_size]
  ep   = tf.concat([ep, tf.zeros([bfm.n_exp_para - ARGS.exp_size])], axis=0)
  ep   = tf.expand_dims(ep, 1)
  pose = params[ARGS.exp_size :]

  # constants
  sp = np.float32(bfm.get_shape_para('zero', 2))
  tx = np.float32(bfm.get_tex_para('zero', 3))

  ## IMPORTANT: Remove scale
  face_vertices  = bfm.generate_vertices(sp, ep) * -0.00001
  face_triangles = bfm.triangles
  face_colors    = bfm.generate_colors(tx)
  scaled_vertices = face_vertices * 1.6

  # FIX NORMALS
  # model_rotation = camera_utils.euler_matrices([[0.,0.,0.]])[0, :3, :3]
  # normals_world_space = tf.reshape(
  #     tf.matmul(face_normals, model_rotation, transpose_b=True), [1, face_vertices.shape[0], face_vertices.shape[1]])
  normals_world_space = tf.reshape(face_normals, [1, face_vertices.shape[0], face_vertices.shape[1]])
  face_colors    = tf.reshape(face_colors, [1, face_vertices.shape[0], face_vertices.shape[1]])

  if ARGS.poses_size == 6:
    # apply transformation
    initial_euler_angles = [pose[:3]]
    model_rotation = camera_utils.euler_matrices(initial_euler_angles)[0, :4, :3]
    trns = tf.concat([pose[3:], [1.]], 0)
    trns = tf.expand_dims(trns, 1)
    model_trans = tf.concat([model_rotation, trns], axis=-1)
    homo = tf.ones([53215, 1], dtype=tf.float32)
    scaled_vertices = tf.concat([scaled_vertices, homo], axis = 1)
    scaled_vertices = tf.matmul(scaled_vertices, model_trans, transpose_b=True)[:, :3]
  
  vertices_world_space = tf.reshape(scaled_vertices, [1, face_vertices.shape[0], face_vertices.shape[1]])

  render = mesh_renderer.mesh_renderer(
      vertices_world_space, face_triangles, normals_world_space,
      face_colors, eye, center, world_up, light_positions,
      light_intensities, image_width, image_height)
  
  return render


def getRandomBG(imgfiles):
    """
    Load a random image from given urls
    """

    b,g,r = cv2.split(cv2.imread(imgfiles[np.random.randint(len(imgfiles))]))
    BG = cv2.merge([r,g,b])
    BG = cv2.resize(BG, (ARGS.size, ARGS.size))
    BG = cv2.GaussianBlur(BG, (3, 3), 0)
    BG = BG / 255.
    BG = BG.astype(np.float32)

    return BG


def saveImage(img, path, mask = None):
  b,g,r,c = cv2.split(img[0])

  if mask is not None:
    mb,mg,mr = cv2.split(mask)
    c = 1.0 - c
    b = b + mb * c
    g = g + mg * c
    r = r + mr * c
  img = cv2.merge([r,g,b])
  cv2.imwrite(path, img * 255)


def randomParams(bfm, STD, type = 'uniform'):
  if type == 'uniform':
    params = tf.random_uniform([ARGS.exp_size], -STD, STD)
  else:
    params = tf.random_normal([ARGS.exp_size], 0, STD)

  return params


def main():
  tf.reset_default_graph()

  if not os.path.exists(ARGS.output):
    os.mkdir(ARGS.output)

  samples = ARGS.samples
  output = ARGS.output
  STD = 3.0
  image_height = ARGS.size
  image_width  = ARGS.size
  path = "../face3dMM/examples/Data/BFM/Out/BFM.mat"
  imgfiles = glob.glob(os.path.join(ARGS.background, '*.jpg'))
  bfm = MorphabelModel(path)
  
  # get normals
  sp = np.float32(bfm.get_shape_para('zero'))
  ep = np.float32(bfm.get_exp_para('zero'))
  bfmnp = MorphabelModelNP(path)
  vertices = bfmnp.generate_vertices(sp, ep)
  cube_normals = get_normal(vertices, bfmnp.triangles)
  np.save("/home/karim/Documents/Development/FacialCapture/face3dMM/examples/results/faces/normals.npy", cube_normals)

  # Pick the desired target Face:
  ARGS.poses_size = 6
  ARGS.exp_size = 29

  s = 0.4
  # trgt_params = randomParams(bfm, STD)
  # trgt_pose   = tf.concat([tf.random_uniform([3], -0.2 * s, 0.2 * s), tf.random_uniform([2], -0.18 * s, 0.18 * s), tf.random_uniform([1], -0.4 * s, 0.4 * s)], axis=0)
  trgt_params = randomParams(bfm, STD * s, 'normal')
  trgt_pose   = tf.concat([tf.random_normal([3], 0, 0.2 * s), tf.random_normal([2], 0, 0.18 * s), tf.random_normal([1], 0, 0.4 * s)], axis=0)
  trgt_params = tf.concat([trgt_params, trgt_pose[:ARGS.poses_size]], axis=0)
  desired_render = renderFace(trgt_params, bfm, cube_normals, image_height, image_width)

  # src faces
  src_params1 = tf.zeros([ARGS.exp_size + ARGS.poses_size]) 
  src_render1 = renderFace(src_params1, bfm, cube_normals, image_height, image_width)

  s = 0.2
  #src_pose2 = tf.concat([tf.random_normal([3], 0, 0.2 * s), tf.random_normal([2], 0, 0.18 * s), tf.random_normal([1], 0, 0.4 * s)], axis=0)
  src_pose2 = tf.concat([tf.random_uniform([3], -0.2 * s, 0.2 * s), tf.random_uniform([2], -0.18 * s, 0.18 * s), tf.random_uniform([1], -0.4 * s, 0.4 * s)], axis=0)
  s = 0.85
  src_params2 = trgt_params + tf.concat([randomParams(bfm, STD * s), src_pose2[:ARGS.poses_size]], 0)
  #src_params2 = tf.random_uniform([ARGS.exp_size], -STD , STD )
  src_render2 = renderFace(src_params2, bfm, cube_normals, image_height, image_width)

  #s = 0.15 * 0.55
  #src_pose3 = tf.concat([tf.random_normal([3], -0.2 * s, 0.2 * s), tf.random_normal([2], -0.3 * s, 0.3 * s), tf.random_normal([1], -0.5 * s, 0.5 * s)], axis=0)
  #s = 0.15
  #s=.8
  #src_params3 = trgt_params + tf.concat([tf.random_uniform([ARGS.exp_size], -STD * s, STD * s), src_pose3], axis = 0)
  #src_params3 =  tf.random_uniform([ARGS.exp_size], -STD * s, STD * s)
  #src_render3 = renderFace(src_params3, bfm, cube_normals, image_height, image_width)


  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in tqdm(range(samples)):

      BG = getRandomBG(imgfiles)

      params_s, image_s, params_t, image_t = sess.run([src_params2, src_render2, trgt_params, desired_render])
      delta_param = params_t - params_s
      params_s = params_s.flatten()
      delta_param = delta_param.flatten()
      # print(params_s)
      # print(params_t)
      # print(delta_param)
      # print()
      # exit()
      np.save('{}/generated_{}_delta.npy'.format(output, i), delta_param)
      np.save('{}/generated_{}_src.npy'.format(output, i), params_s)
      saveImage(image_s, '{}/generated_{}_src.png'.format(output, i))
      saveImage(image_t, '{}/generated_{}_trgt.png'.format(output, i), BG)

      # BIASED BY DESIGN
      # if i < samples / 4: 
      #   params_s, image_s, params_t, image_t = sess.run([src_params1, src_render1, trgt_params, desired_render])
      #   delta_param = params_t - params_s
      #   params_s = params_s.flatten()
      #   delta_param = delta_param.flatten()
      #   np.save('{}/generated_{}_delta.npy'.format(output, i), delta_param)
      #   np.save('{}/generated_{}_src.npy'.format(output, i), params_s)
      #   saveImage(image_s, '{}/generated_{}_src.png'.format(output, i))
      #   saveImage(image_t, '{}/generated_{}_trgt.png'.format(output, i), BG)
      # else:
      #   params_s, image_s, params_t, image_t = sess.run([src_params2, src_render2, trgt_params, desired_render])
      #   delta_param = params_t - params_s
      #   params_s = params_s.flatten()
      #   delta_param = delta_param.flatten()
      #   np.save('{}/generated_{}_delta.npy'.format(output, i), delta_param)
      #   np.save('{}/generated_{}_src.npy'.format(output, i), params_s)
      #   saveImage(image_s, '{}/generated_{}_src.png'.format(output, i))
      #   saveImage(image_t, '{}/generated_{}_trgt.png'.format(output, i), BG)
      # else:
      #   params_s, image_s, params_t, image_t = sess.run([src_params3, src_render3, trgt_params, desired_render])
      #   delta_param = params_t - params_s
      #   params_s = params_s.flatten()
      #   delta_param = delta_param.flatten()
      #   np.save('{}/generated_{}_delta.npy'.format(output, i), delta_param)
      #   np.save('{}/generated_{}_src.npy'.format(output, i), params_s)
      #   saveImage(image_s, '{}/generated_{}_src.png'.format(output, i))
      #   saveImage(image_t, '{}/generated_{}_trgt.png'.format(output, i), BG)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pasrse and normalize BFM17 to a standard dictionary format.')
    parser.add_argument('--output', help = 'Output path to save (.mat)', default = '/home/karim/Documents/Development/FacialCapture/face3dMM/examples/results/faces/del')
    parser.add_argument('--samples', help = 'Number of samples to generate', default = 12, type = int)
    parser.add_argument('--size', help = 'Image size', default = 256, type = int)
    parser.add_argument('--background', help = 'Path to background images (.jpgs)', default = '/home/karim/Documents/Data/Random')

    ARGS = parser.parse_args()

    main()