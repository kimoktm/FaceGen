from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import numpy as np
import tensorflow as tf

import tf_mesh_renderer.mesh_renderer as mesh_renderer
import tf_mesh_renderer.camera_utils as camera_utils
import cv2

from bfm.morphable_model import MorphabelModel
from bfm.morphable_model_np import MorphabelModelNP
import time


# Use CPU only
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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
  eye = tf.constant([[0.0, 0.0, 3.0]], dtype=tf.float32)
  center = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)
  world_up = tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32)
  light_positions = tf.reshape(eye, [1, 1, 3])
  light_intensities = tf.ones([1, 1, 3], dtype=tf.float32)

  # variables
  ep   = params[: bfm.n_exp_para]
  pose = params[bfm.n_exp_para :]

  # constants
  sp = np.float32(bfm.get_shape_para('zero'))
  tx = np.float32(bfm.get_tex_para('zero'))

  ## IMPORTANT: Remove scale
  face_vertices  = bfm.generate_vertices(sp, ep) * 8e-06
  face_triangles = bfm.triangles
  face_colors    = bfm.generate_colors(tx)


  initial_euler_angles = [[0.0, 0.0, 0.0]]
  # euler_angles = tf.Variable(initial_euler_angles)
  model_rotation = camera_utils.euler_matrices(initial_euler_angles)[0, :3, :3]
  scaled_vertices = face_vertices * 1.0

  # apply transformation
  vertices_world_space = tf.reshape(
      tf.matmul(scaled_vertices, model_rotation, transpose_b=True), [1, face_vertices.shape[0], face_vertices.shape[1]])
  
  vertices_world_space = vertices_world_space + pose[:3, 0]
  normals_world_space = tf.reshape(
      tf.matmul(face_normals, model_rotation, transpose_b=True), [1, face_vertices.shape[0], face_vertices.shape[1]])
  face_colors    = tf.reshape(face_colors, [1, face_vertices.shape[0], face_vertices.shape[1]])

  render = mesh_renderer.mesh_renderer(
      vertices_world_space, face_triangles, normals_world_space,
      face_colors, eye, center, world_up, light_positions,
      light_intensities, image_width, image_height)
  
  return render


def fwd_gradients(ys, xs, d_xs):
  """Forward-mode pushforward analogous to the pullback defined by tf.gradients.
  With tf.gradients, grad_ys is the vector being pulled back, and here d_xs is
  the vector being pushed forward."""
  v = tf.placeholder(ys.dtype, shape=ys.get_shape())  # dummy variable
  g = tf.gradients(ys, xs, grad_ys=v)
  return tf.gradients(g, v, grad_ys=d_xs)


if __name__ == '__main__':
  tf.reset_default_graph()

  image_height = 240
  image_width  = 240
  path = "../face3dMM/examples/Data/BFM/Out/BFM.mat"
  bfm = MorphabelModel(path)
  
  # get normals
  sp = np.float32(bfm.get_shape_para('zero'))
  ep = np.float32(bfm.get_exp_para('zero'))
  bfmnp = MorphabelModelNP(path)
  vertices = bfmnp.generate_vertices(sp, ep)
  cube_normals = get_normal(vertices, bfmnp.triangles)

  # Start face
  ps = tf.concat([tf.zeros([bfm.n_exp_para]), tf.random_uniform([3], -0.2, 0.2)], 0)
  ps = tf.expand_dims(ps, -1)
  params = tf.Variable(ps)
  render = renderFace(params, bfm, cube_normals, image_height, image_width)

  # Pick the desired target Face:
  # trgt_params = np.float32(bfm.get_exp_para('random', 3.2))
  trgt_params = tf.constant([-1.4303448 ,-3.1902063 ,-0.44127062,-2.8239923 , 2.5229132 ,-2.758617  ,-1.1830097 ,-2.0942547 ,-2.3744998 , 2.439571  ,-1.0012991 , 3.076722  ,-1.137647  , 0.63871115, 1.2823539 , 0.01328026, 0.3616442 , 2.8016732 , 2.414662  ,-1.0671458 ,-1.6383948 ,-1.4478279 , 0.6882379 ,-1.4079901 , 2.1107602 ,-1.4473487 ,-2.410577  , 2.7083504 , 0.52026314])
  trgt_params = tf.concat([trgt_params, [0, 0, 0]], 0)

  # trgt_params = tf.constant([-1.4303448, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
  trgt_params = tf.expand_dims(trgt_params, -1)
  desired_render = renderFace(trgt_params, bfm, cube_normals, image_height, image_width)


  isTraining = tf.Variable(True)
  x = (desired_render[:, :, :, :3] - render[:, :, :, :3])
  # x = tf.concat([desired_render[:, :, :, :3], render[:, :, :, :3]], 3)

  with tf.variable_scope("network"):
    F = tf.layers.conv2d(x, 32, 3)
    F = tf.nn.relu(F)
    F = tf.layers.conv2d(F, 64, 3)
    F = tf.nn.relu(F)
    F = tf.layers.max_pooling2d(inputs=F, pool_size=3, strides=2, padding='SAME')
    F = tf.layers.conv2d(F, 128, 3)
    F = tf.nn.relu(F)
    F = tf.reduce_mean(F, [1, 2], keepdims=True)
    F = tf.squeeze(F, [1, 2])


  # network loss
  GT_P = trgt_params - params
  J_F = tf.gradients(tf.square(F), params)[0] * -1
  print(J_F)

  nt_loss = tf.losses.mean_squared_error(labels=GT_P, predictions=J_F)
  nt_optimizer = tf.train.AdamOptimizer(0.00005)
  nt_loss = tf.tuple([nt_loss], control_inputs=tf.get_collection(tf.GraphKeys.UPDATE_OPS))[0]
  nt_grads = tf.gradients(nt_loss, tf.trainable_variables('network'))
  nt_grads_and_vars = list(zip(nt_grads, tf.trainable_variables('network')))
  nt_opt_func = nt_optimizer.apply_gradients(nt_grads_and_vars)
  nt_grads_F = tf.gradients(nt_loss, F)

  # Fitting loss
  loss = tf.reduce_mean(tf.abs(desired_render[:, :, :, :3] - render[:, :, :, :3]))
  # loss = tf.square(F)
  optimizer = tf.train.AdamOptimizer(0.1)
  # optimizer = tf.train.GradientDescentOptimizer(1.)
  grads_and_vars = optimizer.compute_gradients(loss, params)
  opt_func = optimizer.apply_gradients(grads_and_vars)
  st_BN = tf.assign(isTraining, False)


  start_time = time.time()

  with tf.Session() as sess:
    cv2.namedWindow('optimizer')
    sess.run(tf.global_variables_initializer())
    
    # train function network
    for i in range(50):
      lss, _ = sess.run([nt_loss, nt_opt_func])
      print(lss)
      # cv2.imshow('optimizer', learned_F[0])
      # k = cv2.waitKey(1)
      # if k == 27:
      #   exit()
      # print(grd)
      # print(".")
      # print(".")
      # print(fgrd)
      # print("----------------------------")
      # print(".")
      # print(learned_g)
    print("Finished training")
    print("----------------------------")

    # run fitting
    # sess.run(st_BN)
    for i in range(0):
      lss, _, x  = sess.run([loss, opt_func, grads_and_vars])
      print(lss)
      # print(x)
      final_image, desired_image = sess.run([render, desired_render])
      numpy_horizontal = np.hstack((final_image[0], desired_image[0] ))
      b,g,r,c = cv2.split(numpy_horizontal)
      numpy_horizontal = cv2.merge([r,g,b])
      cv2.imshow('optimizer', numpy_horizontal)
      k = cv2.waitKey(1)

      if k == 27:
        exit()

    elapsed_time = time.time() - start_time
    print(elapsed_time)
    exit()


  k = cv2.waitKey(0)
  if k == 27:
    exit()