from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import os
import cv2
import dlib

from bfm.morphable_model import MorphabelModel
from bfm.morphable_model_np import MorphabelModelNP
import render.face_renderer as fr

from tqdm import tqdm



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


def loadImg(path, masked_landmarks, image_width, image_height):
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
    #img = cv2.GaussianBlur(img, (3, 3), 0)

    # extract landmarks
    predictor_path = "/home/karim/Documents/Development/FacialCapture/Facial-Capture/models/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    pvt = getFaceKeypoints(img, detector, predictor)
    pvt = np.asarray(pvt)[0].T
    pvt = pvt[masked_landmarks, :2]
    pvt[:, 1] = image_height - pvt[:, 1]
    pvt = tf.convert_to_tensor(pvt, dtype=tf.float32)
    img = tf.convert_to_tensor(img / 255., dtype=tf.float32)
    #pvt = tf.expand_dims(pvt, 0)
    #img = tf.expand_dims(img, 0)

    return img, pvt


def loadImgs(paths, masked_landmarks, image_width, image_height):
    imgs = []
    lnds = []
    for p in paths:
        img, lnd = loadImg(p, masked_landmarks, image_width, image_height)
        imgs.append(img)
        lnds.append(lnd)
    
    imgs = tf.stack(imgs, axis=0)
    lnds = tf.stack(lnds, axis=0)

    return imgs, lnds



CURR_FRAME = 0
def loadNextFrame(vid_path, masked_landmarks, image_width, image_height):
    global CURR_FRAME
    frame_path = os.path.join(vid_path, str(CURR_FRAME) + '.png')
    CURR_FRAME = CURR_FRAME + 1

    img, pvt = loadImg(frame_path, masked_landmarks, image_width, image_height)
    img = tf.expand_dims(img, 0)
    pvt = tf.expand_dims(pvt, 0)

    return img, pvt


def writeObj(obj_name, vertices, triangles, colors):
    ''' Save 3D face model with texture represented by colors.
    Args:
            obj_name: str
            vertices: shape = (nver, 3)
            triangles: shape = (ntri, 3)
            colors: shape = (nver, 3)
    '''

    triangles = triangles.copy()
    triangles += 1 # meshlab start with 1
    
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
            
    # write obj
    with open(obj_name, 'w') as f:
        # write vertices & colors
        for i in range(vertices.shape[0]):
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
            f.write(s)

        # write f: ver ind/ uv ind
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            f.write(s)


def showImages(left_images, right_images, left_landmarks, right_landmarks, image_height=256, show_markers=True):
    batch_size = left_images.shape[0]
    stacked_imgs = []

    for i in range(batch_size):
        left_landmarks[i, :, 1]  = image_height - left_landmarks[i, :, 1]
        right_landmarks[i, :, 1] = image_height - right_landmarks[i, :, 1]
        if show_markers:
            drawPoints(left_images[i], left_landmarks[i])
            drawPoints(left_images[i], right_landmarks[i], (255, 0, 0))
        drawPoints(right_images[i], right_landmarks[i])
        progress_img = np.hstack((left_images[i], right_images[i]))
        stacked_imgs.append(progress_img)

    if batch_size > 3:
        if batch_size % 2 != 0:
            stacked_imgs.append(np.zeros_like(stacked_imgs[0]))
            batch_size = batch_size + 1
        stacked_imgs_l = np.vstack(stacked_imgs[: int(batch_size/2)])
        stacked_imgs_r = np.vstack(stacked_imgs[int(batch_size/2):])
        stacked_imgs = np.hstack((stacked_imgs_l, stacked_imgs_r))
    else:
        stacked_imgs = np.vstack(stacked_imgs)
    
    b,g,r        = cv2.split(stacked_imgs)
    stacked_imgs = cv2.merge([r,g,b])

    cv2.imshow('Optimizer', stacked_imgs)
    k = cv2.waitKey(1)

    if k == 27:
        exit()


if __name__ == '__main__':
    tf.reset_default_graph()

    BATCH_SIZE   = 1
    perspective  = False
    image_height = 512
    image_width  = 512  
    path         = "../face3dMM/examples/Data/BFM/Out/BFM17Face.mat"
    vid_path     = '/home/karim/Documents/Development/FacialCapture/Facial-Capture/data/craig/orig'


    bfm = MorphabelModel(path)
    ARGS_landmarks = bfm.landmarks
    TRGT_landmarks = bfm.landmarks_ids


    # Start face
    identity          = tf.Variable(tf.zeros([BATCH_SIZE, bfm.n_shape_para]))
    albedo            = tf.Variable(tf.zeros([BATCH_SIZE, bfm.n_tex_para]))
    expressions       = tf.Variable(tf.zeros([BATCH_SIZE, bfm.n_exp_para]))
    pose              = tf.Variable(tf.zeros([BATCH_SIZE, 6]))
    sh_coff           = np.zeros([BATCH_SIZE, 9, 3], dtype=np.float32)
    sh_coff[:, 0, 0]  = 1.0
    sh_coff[:, 0, 1]  = 1.0
    sh_coff[:, 0, 2]  = 1.0
    sh_coff           = tf.Variable(sh_coff)
    flow_field        = tf.Variable(tf.zeros([BATCH_SIZE, bfm.nver, 3]))
    render, pvs, colr = fr.renderFaces(identity, expressions, pose, albedo, sh_coff, flow_field, bfm, perspective, image_width, image_height)


    # Load real-image
    trgt_render, pvt = loadNextFrame(vid_path, TRGT_landmarks, image_width, image_height)
    trgt_render = tf.Variable(trgt_render)
    pvt = tf.Variable(pvt)

    # mask target
    alpha       = render[:, :, :, -1]
    render      = render[:, :, :, :3]
    mask_r      = tf.boolean_mask(render, alpha)
    mask_t      = tf.boolean_mask(trgt_render, alpha)


    # loss function
    pixel_loss     = tf.reduce_mean(tf.square(mask_t - mask_r))
    landmarks_loss = tf.reduce_mean(tf.square(pvt - pvs))
    reg_loss       = tf.reduce_sum(tf.square(albedo)) * 0.1 + tf.reduce_sum(tf.square(identity)) + tf.reduce_sum(tf.square(expressions))

    # Pose optimizer
    pose_loss = landmarks_loss + reg_loss
    pos_optimizer = tf.train.AdamOptimizer(0.2)
    pos_grads_and_vars = pos_optimizer.compute_gradients(pose_loss, [pose, identity, expressions])
    pos_opt_func = pos_optimizer.apply_gradients(pos_grads_and_vars)


    # Global fitting optimizer
    loss = 1.1 * pixel_loss + 2.5e-5 * landmarks_loss + 5e-8 * reg_loss
    loss = 1.1 * pixel_loss + 5e-5 * landmarks_loss + 5e-8 * reg_loss

    global_step = tf.train.get_or_create_global_step()
    decay_learning_rate = tf.train.exponential_decay(0.02, global_step, 400, 0.8, staircase=True)
    optimizer = tf.train.AdamOptimizer(0.01)
    grads_and_vars = optimizer.compute_gradients(loss, [identity, albedo, expressions, pose, sh_coff])
    opt_func = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


    # Flow field optimizer
    flow_loss =  1.1 * pixel_loss + 1 * tf.reduce_mean(flow_field) + 5e-6 * reg_loss
    flow_optimizer = tf.train.AdamOptimizer(0.000005)
    flow_grads_and_vars = flow_optimizer.compute_gradients(flow_loss, [identity, albedo, expressions, pose, sh_coff, flow_field])
    flow_opt_func = flow_optimizer.apply_gradients(flow_grads_and_vars, global_step=global_step)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Fit pose first
        print("Show video")
        for i in tqdm(range(100)):
            xf, xl = loadNextFrame(vid_path, TRGT_landmarks, image_width, image_height)
            up_frame = tf.assign(trgt_render, xf)
            up_lnd = tf.assign(pvt, xl)

            sess.run([up_frame, up_lnd])

            # Fit pose first
            print("Pose fitting")
            for i in tqdm(range(0)):
                lss, _ = sess.run([pose_loss, pos_opt_func])
                prog_image, prog_lnd, trgt_image, trgt_lnd = sess.run([render, pvs, trgt_render, pvt])
                showImages(prog_image, trgt_image, prog_lnd, trgt_lnd, image_height)


            # Global fitting
            print("Global fitting")
            for i in tqdm(range(300)):
                lss, _, pl, ll, rl = sess.run([loss, opt_func, pixel_loss, landmarks_loss, reg_loss])
                prog_image, prog_lnd, trgt_image, trgt_lnd = sess.run([render, pvs, trgt_render, pvt])
                showImages(prog_image, trgt_image, prog_lnd, trgt_lnd, image_height, False)


            # Flow field fitting
            print("Flow field fitting")
            for i in tqdm(range(0)):
                lss, _ = sess.run([flow_loss, flow_opt_func])
                id_params, ep_params, alb_params, flow_params = sess.run([identity, expressions, colr, flow_field])
                prog_image, prog_lnd, trgt_image, trgt_lnd = sess.run([render, pvs, trgt_render, pvt])
                showImages(prog_image, trgt_image, prog_lnd, trgt_lnd, image_height, False)


    # # Save Obj file
    # bfmNp = MorphabelModelNP(path)
    # final_ver = bfmNp.generate_vertices(id_params, ep_params) * SCALE_FACTOR + flow_params
    # final_ver = final_ver / np.amax(final_ver) * 1000.
    # final_alb = alb_params
    # #final_alb = bfmNp.generate_colors(alb_params)
    # write_obj_with_colors('/home/karim/Desktop/optimized_face.obj', final_ver, bfmNp.triangles, final_alb)
    print("Done :)")

    k = cv2.waitKey(0)
    if k == 27:
        exit()