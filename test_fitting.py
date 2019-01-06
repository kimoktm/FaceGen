from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import numpy as np
import tensorflow as tf

import cv2

import tf_mesh_renderer.mesh_renderer as mesh_renderer
import tf_mesh_renderer.camera_utils as camera_utils

from bfm.morphable_model import MorphabelModel



def get_normals(vertices, triangles, bfm):
    ''' calculate normal direction in each vertex
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
    Returns:
        normal: [nver, 3]
    '''

    pt0 = tf.gather(vertices, triangles[:, 0]) # [ntri, 3]
    pt1 = tf.gather(vertices, triangles[:, 1]) # [ntri, 3]
    pt2 = tf.gather(vertices, triangles[:, 2]) # [ntri, 3]
    tri_normal = tf.cross(pt0 - pt1, pt0 - pt2) # [ntri, 3]. normal of each triangle

    # add a zero normal to the end
    tri_normal = tf.concat([tri_normal, [[0, 0, 0]]], axis=0)
    zero_indx  = tri_normal.get_shape()[0] - 1  # (nver + 1) -1

    v2f = np.empty([bfm.nver, 8], dtype=int)
    mx = 0
    mn = 9999
    for x in range(bfm.vertex2face.shape[0]):
      sz = bfm.vertex2face[x].shape[0]
      mx = np.maximum(sz, mx)
      mn = np.minimum(sz, mn)
      v2f[x] = np.concatenate((bfm.vertex2face[x], np.ones(8-sz) * zero_indx))

    print(mx)
    print(mn)

    normals = tf.gather(tri_normal, v2f)
    normals = tf.reduce_sum(normals, axis=1)
    normals = tf.nn.l2_normalize(normals, axis=1)

    return normals


def sh9(normals):
  """
  First nine spherical harmonics as functions of Cartesian coordinates
  """

  # h = np.empty((9, x.size))
  # h[0, :] = 1/np.sqrt(4*np.pi) * np.ones(x.size)
  # h[1, :] = np.sqrt(3/(4*np.pi)) * z
  # h[2, :] = np.sqrt(3/(4*np.pi)) * x
  # h[3, :] = np.sqrt(3/(4*np.pi)) * y
  # h[4, :] = 1/2*np.sqrt(5/(4*np.pi)) * (3*np.square(z) - 1)
  # h[5, :] = 3*np.sqrt(5/(12*np.pi)) * x * z
  # h[6 ,:] = 3*np.sqrt(5/(12*np.pi)) * y * z
  # h[7, :] = 3/2*np.sqrt(5/(12*np.pi)) * (np.square(x) - np.square(y))
  # h[8, :] = 3*np.sqrt(5/(12*np.pi)) * x * y
  # return h * np.r_[np.pi, np.repeat(2 * np.pi/ 3, 3), np.repeat(np.pi/ 4, 5)][:, np.newaxis]
  # h = h.T

  x = normals[:, 0]
  y = normals[:, 1]
  z = normals[:, 2]
  h = []
  h.append(tf.ones_like(x))
  h.append(y)
  h.append(z)
  h.append(x)
  h.append(x * y)
  h.append(y * z)
  h.append(3 * tf.square(z) - 1.)
  h.append(x * z)
  h.append(tf.square(x) - tf.square(y))
  h = tf.stack(h, axis=1)
 
  return h


def renderFace(identity, albedo, expressions, pose, sh_coff,
 bfm, prespective=True, image_height = 256, image_width = 256):
  # camera position:
  eye = tf.constant([[0.0, 0.0, 3.0]], dtype=tf.float32)
  center = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)
  world_up = tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32)
  light_positions = tf.reshape(-eye, [1, 1, 3])
  light_intensities = tf.ones([1, 1, 3], dtype=tf.float32)


  # variables
  if not identity:
    identity = np.float32(bfm.get_shape_para('zero', 2))
  if not albedo:
    albedo = np.float32(bfm.get_tex_para('zero', 3))
  if not expressions:
    expressions = np.float32(bfm.get_exp_para('zero', 3))
  if not pose:
    pose = np.zeros(6, dtype=np.float32)
    if not prespective:
      pose[-1] = 1
  if not sh_coff:
    sh_coff = np.zeros([9, 3], dtype=np.float32)
    sh_coff[0, 0] = 1.0
    sh_coff[0, 1] = 1.0
    sh_coff[0, 2] = 1.0


  ## IMPORTANT: Remove scale
  #face_vertices  = bfm.generate_vertices(identity, expressions) * 8e-06
  face_vertices  = bfm.generate_vertices(identity, expressions) * 8e-03
  face_colors    = bfm.generate_colors(albedo)


  # apply transformation
  initial_euler_angles = [pose[:3]]
  model_rotation = camera_utils.euler_matrices(initial_euler_angles)[0, :4, :3]
  trns = tf.concat([pose[3:], [1.]], 0)
  if not prespective:
    scale = pose[-1]
    face_vertices = face_vertices * scale
    trns = tf.concat([pose[3:5], [0, 1.]], 0)
  trns = tf.expand_dims(trns, 1)
  model_trans = tf.concat([model_rotation, trns], axis=-1)
  homo = tf.ones([bfm.nver, 1], dtype=tf.float32)
  face_vertices = tf.concat([face_vertices, homo], axis = 1)
  face_vertices = tf.matmul(face_vertices, model_trans, transpose_b=True)[:, :3]
  

  # calculate normals & Spherical harmonics
  face_normals = get_normals(face_vertices, bfm.triangles, bfm)

  # Spherical harmonics
  albedo_colors = tf.identity(face_colors)
  face_colors = []
  for c in range(3):
    color_channel = tf.tensordot(sh9(face_normals), sh_coff[:, c], 1) * albedo_colors[:, c]
    face_colors.append(color_channel)
  face_colors = tf.stack(face_colors, axis=1)
  

  # cast as batches
  face_colors          = tf.reshape(face_colors, [1, face_vertices.shape[0], face_vertices.shape[1]])
  normals_world_space  = tf.reshape(face_normals, [1, face_vertices.shape[0], face_vertices.shape[1]])
  vertices_world_space = tf.reshape(face_vertices, [1, face_vertices.shape[0], face_vertices.shape[1]])

  render, pv = mesh_renderer.mesh_renderer(
      vertices_world_space, bfm.triangles, normals_world_space,
      face_colors, eye, center, world_up, light_positions,
      light_intensities, image_width, image_height, perspective=prespective)
  
  return render, pv


def randomParams(size, STD, type = 'uniform'):
  if type == 'uniform':
    params = tf.random_uniform([size], -STD, STD)
  else:
    params = tf.random_normal([size], 0, STD)

  return params * 0


def getFaceKeypoints(img, detector, predictor, maxImgSizeForDetection=620):
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
        # shape2D = np.array([[(det.left() / imgScale), (det.top() / imgScale)], \
        #                     [(det.right() / imgScale), (det.top() / imgScale)], \
        #                     [(det.right() / imgScale), (det.bottom() / imgScale)],
        #                     [(det.left() / imgScale), (det.bottom() / imgScale)]])

        shape2D = np.array([[p.x, p.y] for p in dlibShape.parts()])
        shape2D = shape2D.T
        shapes2D.append(shape2D)

    return shapes2D


def drawPoints(img, points, color=(0, 255, 0)):
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 2, color)


def loadImg(path, image_width, image_height):
  img = cv2.imread(path)

  # Create a black image
  height, width, channels = img.shape
  x = height if height > width else width
  y = height if height > width else width
  square= np.zeros((x,y,3), np.uint8)
  square[int((y-height)/2):int(y-(y-height)/2), int((x-width)/2):int(x-(x-width)/2)] = img
  img = square

  b,g,r = cv2.split(img)
  img = cv2.merge([r,g,b])
  img = cv2.resize(img, (image_width, image_height))
  img = cv2.GaussianBlur(img, (3, 3), 0)

  return img



if __name__ == '__main__':
  tf.reset_default_graph()
  tf.set_random_seed(125)

  Prespective = False
  STD = 2.0
  STD_pose = STD * 0.02
  image_height = 480
  image_width  = 480
  ARGS_landmarks = [22143, 22813, 22840, 23250, 44124, 45884, 47085, 47668, 48188, 48708, 49299, 50498, 52457, 32022, 32386, 32359, 32979, 38886, 39636, 40030, 40238, 40433, 41172, 41368, 41578, 42011, 42646, 8291, 8305, 8314, 8320, 6783, 7687, 8331, 8977, 9879, 1832, 3760, 5050, 6087, 4546, 3516, 10731, 11758, 12919, 14859, 13191, 12157, 5523, 6155, 7442, 8345, 9506, 10799, 11199, 10179, 9277, 8374, 7471, 6566, 5909, 7322, 8354, 9386, 10941, 9141, 8367, 7194]


  TRGT_landmarks = [0, 1, 2, 3, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
  ARGS_landmarks = [16203, 16235, 16260, 16290, 26869, 27061, 27253, 22481, 22451, 22426, 22394, 22586, 22991, 23303, 23519, 23736, 24312, 24527, 24743, 25055, 25466, 8134, 8143, 8151, 8157, 6986, 7695, 8167, 8639, 9346, 2602, 4146, 4920, 5830, 4674, 3900, 10390, 11287, 12061, 13481, 12331, 11557, 5522, 6026, 7355, 8181, 9007, 10329, 10857, 9730, 8670, 8199, 7726, 6898, 6291, 7364, 8190, 9016, 10088, 8663, 8191, 7719]
  path = "../face3dMM/examples/Data/BFM/Out/BFM17Face.mat"
  bfm = MorphabelModel(path)


  # Start face
  STD = 0
  identity    = tf.Variable(randomParams(bfm.n_shape_para, STD, 'normal'))
  albedo      = tf.Variable(randomParams(bfm.n_tex_para, STD, 'normal'))
  expressions = tf.Variable(randomParams(bfm.n_exp_para, STD, 'normal'))
  #pose        = tf.Variable(randomParams(6, STD_pose))
  #sh_coff     = tf.reshape(randomParams(27, STD_pose) *0 + 0.5, [9, 3])
  pose = np.zeros(6, dtype=np.float32)
  pose[-1] = 1.0
  pose = tf.Variable(pose)
  sh_coff = np.zeros([9, 3], dtype=np.float32)
  sh_coff[0, 0] = 1.0
  sh_coff[0, 1] = 1.0
  sh_coff[0, 2] = 1.0
  sh_coff     = tf.Variable(sh_coff)
  render, pvs = renderFace(identity, albedo, expressions, pose, sh_coff, bfm, Prespective, image_height, image_width)
  pvs         = tf.gather(pvs[0], ARGS_landmarks)[:, :2]
  pvs         = pvs * image_height / 2. + image_height / 2.


  # Pick the desired target Face:
  trgt_identity    = tf.Variable(randomParams(bfm.n_shape_para, STD, 'normal'))
  trgt_albedo      = tf.Variable(randomParams(bfm.n_tex_para, STD, 'normal'))
  trgt_expressions = tf.Variable(randomParams(bfm.n_exp_para, STD, 'normal'))
  trgt_pose        = tf.Variable(randomParams(6, STD_pose))
  trgt_sh_coff     = tf.Variable(randomParams(27, STD_pose) *0 + 1)
  trgt_sh_coff     = None
  trgt_render, pvt = renderFace(trgt_identity, trgt_albedo, trgt_expressions, trgt_pose, trgt_sh_coff, bfm, Prespective, image_height, image_width)
  trgt_render      = trgt_render[:, :, :, :3]
  pvt              = tf.gather(pvt[0], ARGS_landmarks)[:, :2]
  pvt              = pvt * image_height / 2. + image_height / 2.


  # Load real-image
  import dlib
  trgt_render = loadImg('/home/karim/Desktop/face_6.png', image_width, image_height)
  predictor_path = "/home/karim/Documents/Development/FacialCapture/Facial-Capture/models/shape_predictor_68_face_landmarks.dat"
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(predictor_path)
  pvt = getFaceKeypoints(trgt_render, detector, predictor)
  pvt = np.asarray(pvt)[0].T
  pvt = pvt[TRGT_landmarks, :2]
  pvt[:, 1] = image_height - pvt[:, 1]
  pvt = tf.convert_to_tensor(pvt, dtype=tf.float32)
  trgt_render = np.float32(trgt_render / 255.)
  trgt_render = tf.expand_dims(trgt_render, 0)



  # mask target
  alpha       = render[:, :, :, -1]
  render      = render[:, :, :, :3]
  mask_r      = tf.boolean_mask(render, alpha)
  mask_t      = tf.boolean_mask(trgt_render, alpha)

  # loss function
  pixel_loss = tf.reduce_mean(tf.square(mask_t - mask_r))
  landmarks_loss = tf.reduce_mean(tf.square(pvt - pvs))
  reg_loss = tf.reduce_sum(tf.square(identity)) + tf.reduce_sum(tf.square(albedo)) + tf.reduce_sum(tf.square(expressions))
  loss = 1.1 * pixel_loss + 0.1* 2.5e-5 * landmarks_loss + 5e-8 * reg_loss

  global_step = tf.train.get_or_create_global_step()
  decay_learning_rate = tf.train.exponential_decay(0.02, global_step,
                                               400, 0.8, staircase=True)
  optimizer = tf.train.AdamOptimizer(decay_learning_rate)
  grads_and_vars = optimizer.compute_gradients(loss, [identity, albedo, expressions, pose, sh_coff])
  opt_func = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

  # pose loss
  pos_optimizer = tf.train.AdamOptimizer(0.1)
  pos_grads_and_vars = pos_optimizer.compute_gradients(landmarks_loss, [pose])
  pos_opt_func = pos_optimizer.apply_gradients(pos_grads_and_vars)



  with tf.Session() as sess:
    cv2.namedWindow('optimizer')
    sess.run(tf.global_variables_initializer())

    # Fit pose first
    print("Pose fitting")
    for i in range(70):
      lss, _ = sess.run([landmarks_loss, pos_opt_func])
      print(lss)
      final_image, final_lnd, trgt_image, trgt_lnd = sess.run([render, pvs, trgt_render, pvt])
      final_lnd[:, 1] = image_height - final_lnd[:, 1]
      trgt_lnd[:, 1]  = image_height - trgt_lnd[:, 1]
      drawPoints(final_image[0], final_lnd)
      drawPoints(trgt_image[0], trgt_lnd)

      numpy_horizontal = np.hstack((final_image[0], trgt_image[0]))
      b,g,r = cv2.split(numpy_horizontal)
      numpy_horizontal = cv2.merge([r,g,b])
      cv2.imshow('optimizer', numpy_horizontal)
      k = cv2.waitKey(1)

      if k == 27:
        exit()


    # Global fitting
    print("Global fitting")
    for i in range(1000):
      lss, _, pl, ll, rl = sess.run([loss, opt_func, pixel_loss, landmarks_loss, reg_loss])
      print(lss)
      print(pl)
      print(ll)
      print(rl)
      print("")
      final_image, final_lnd, trgt_image, trgt_lnd = sess.run([render, pvs, trgt_render, pvt])
      final_lnd[:, 1] = image_height - final_lnd[:, 1]
      trgt_lnd[:, 1]  = image_height - trgt_lnd[:, 1]
      #drawPoints(final_image[0], final_lnd)
      #drawPoints(final_image[0], trgt_lnd, (255, 0, 0))
      drawPoints(trgt_image[0], trgt_lnd)

      numpy_horizontal = np.hstack((final_image[0], trgt_image[0]))
      b,g,r = cv2.split(numpy_horizontal)
      numpy_horizontal = cv2.merge([r,g,b])
      cv2.imshow('optimizer', numpy_horizontal)
      #cv2.imwrite('/home/karim/Desktop/differentiable_renderer/optimizing_' + '{:03}'.format(i) + '.jpg', numpy_horizontal * 255)
      k = cv2.waitKey(1)

      if k == 27:
        exit()

  k = cv2.waitKey(0)
  if k == 27:
    exit()