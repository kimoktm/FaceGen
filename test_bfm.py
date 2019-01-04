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
  ep   = params[: bfm.n_exp_para]
  # pose = params[bfm.n_exp_para :]

  # constants
  sp = np.float32(bfm.get_shape_para('zero', 2))
  tx = np.float32(bfm.get_tex_para('zero', 3))
  
  ## IMPORTANT: Remove scale
  face_vertices  = bfm.generate_vertices(sp, ep) * -0.00001
  face_triangles = bfm.triangles
  face_colors    = bfm.generate_colors(tx)


  initial_euler_angles = [[0.0, 0.0, 0.0]]
  euler_angles = tf.Variable(initial_euler_angles)
  model_rotation = camera_utils.euler_matrices(euler_angles)[0, :3, :3]
  scale = tf.Variable(1.0)
  scaled_vertices = face_vertices * scale

  # apply transformation
  vertices_world_space = tf.reshape(
      tf.matmul(scaled_vertices, model_rotation, transpose_b=True), [1, face_vertices.shape[0], face_vertices.shape[1]])
  
  vertices_world_space = vertices_world_space + [0., 0., 2.]
  normals_world_space = tf.reshape(
      tf.matmul(face_normals, model_rotation, transpose_b=True), [1, face_vertices.shape[0], face_vertices.shape[1]])
  face_colors    = tf.reshape(face_colors, [1, face_vertices.shape[0], face_vertices.shape[1]])

  render = mesh_renderer.mesh_renderer(
      vertices_world_space, face_triangles, normals_world_space,
      face_colors, eye, center, world_up, light_positions,
      light_intensities, image_width, image_height)
  
  return render



if __name__ == '__main__':
  tf.reset_default_graph()

  image_height = 480
  image_width  = 640
  path = "../face3dMM/examples/Data/BFM/Out/BFM.mat"
  bfm = MorphabelModel(path)
  
  # get normals
  sp = np.float32(bfm.get_shape_para('zero'))
  ep = np.float32(bfm.get_exp_para('zero'))
  bfmnp = MorphabelModelNP(path)
  vertices = bfmnp.generate_vertices(sp, ep)
  cube_normals = get_normal(vertices, bfmnp.triangles)

  # Start face
  params = tf.Variable(ep)
  render = renderFace(params, bfm, cube_normals, image_height, image_width)

  # Pick the desired target Face:
  trgt_params = np.float32(bfm.get_exp_para('random', 2))
  desired_render = renderFace(trgt_params, bfm, cube_normals, image_height, image_width)



  # masked_trgt = tf.multiply(desired_render, tf.expand_dims(render[:, :, 3], -1))
  loss = tf.reduce_mean(tf.abs(render - desired_render))
  optimizer = tf.train.AdamOptimizer(0.009)
  grads_and_vars = optimizer.compute_gradients(loss, params)
  opt_func = optimizer.apply_gradients(grads_and_vars)


  with tf.Session() as sess:
    cv2.namedWindow('optimizer')
    sess.run(tf.global_variables_initializer())
    for i in range(200):
      lss, _  = sess.run([loss, opt_func])
      print(lss)
      final_image, desired_image = sess.run([render, desired_render])
      numpy_horizontal = np.hstack((final_image[0], desired_image[0]))
      b,g,r,c = cv2.split(numpy_horizontal)
      numpy_horizontal = cv2.merge([r,g,b])
      cv2.imshow('optimizer', numpy_horizontal)
      cv2.imwrite('/home/karim/Desktop/differentiable_renderer/optimizing_' + '{:03}'.format(i) + '.jpg', numpy_horizontal * 255)
      k = cv2.waitKey(1)

      if k == 27:
        exit()


  k = cv2.waitKey(0)
  if k == 27:
    exit()