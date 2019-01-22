from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
import tensorflow as tf

import cv2
import dlib
import glob
import os

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
    img = cv2.GaussianBlur(img, (3, 3), 0)

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

    # clip colors
    colors = np.clip(colors, 0.0, 1.0)
    
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



xx = 0
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

    global xx
    if xx % 10:
        cv2.imwrite('/home/karim/Desktop/differentiable_renderer/' + str(xx) + '.jpg', stacked_imgs * 255)
    xx = xx + 1
    cv2.imshow('Optimizer', stacked_imgs)

    k = cv2.waitKey(1)

    if k == 27:
        exit()



def load_uv_coords(obj_path):
    """Returns the shape vertices and the list of vertex indices for each mesh face.
    
    Args:
        fName (str): Filename of .obj file
        dataToImport (list): A list containing strings that indicate what part of the .obj file to read (``v`` = geometric vertices, ``f`` = face indices). Note that all of the .obj files for a given 3DMM have the same face indices.
    
    Returns:
        ndarray or tuple: the vertex coordinates, the vertex indices for each face, or both
    """
        
    with open(obj_path) as fd:
        uv_coords = []

        for line in fd:
            if line.startswith('vt'):
                uv_coords.append([float(num) for num in line[3:].split(' ')])
            else:
                continue

    uv_coords = np.array(uv_coords)

    return uv_coords



def bilinear_sampler(data, v, img_shape=100):
    """
        Args:
            x - Input tensor [N, H, W, C]
            v - Vector flow tensor [N, H, W, 2], tf.float32
            (optional)
            out  - Handling out of boundary value.
                         Zero value is used if out="CONSTANT".
                         Boundary values are used if out="EDGE".
    """

    # if out == "CONSTANT":
    #     x = tf.pad(x,
    #         ((0,0), (1,1), (1,1), (0,0)), mode='CONSTANT')
    # elif out == "EDGE":
    #     x = tf.pad(x,
    #         ((0,0), (1,1), (1,1), (0,0)), mode='REFLECT')

    # vy, vx = tf.split(v, 2, axis=2)

    # vx0 = tf.floor(vx)
    # vy0 = tf.floor(vy)
    # vx1 = vx0 + 1
    # vy1 = vy0 + 1 # [N, H, W, 1]

    # H_1 = tf.cast(H_+1, tf.float32)
    # W_1 = tf.cast(W_+1, tf.float32)
    # iy0 = tf.clip_by_value(vy0, 0., H_1)
    # iy1 = tf.clip_by_value(vy1, 0., H_1)
    # ix0 = tf.clip_by_value(vx0, 0., W_1)
    # ix1 = tf.clip_by_value(vx1, 0., W_1)

    # i00 = tf.concat([iy0, ix0], 2)
    # i01 = tf.concat([iy1, ix0], 2)
    # i10 = tf.concat([iy0, ix1], 2)
    # i11 = tf.concat([iy1, ix1], 2) # [N, H, W, 3]
    # i00 = tf.cast(i00[0], tf.int32)
    # i01 = tf.cast(i01[0], tf.int32)
    # i10 = tf.cast(i10[0], tf.int32)
    # i11 = tf.cast(i11[0], tf.int32)


    x = v[:,0]
    y = v[:,1]
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, img_shape-1);
    x1 = np.clip(x1, 0, img_shape-1);
    y0 = np.clip(y0, 0, img_shape-1);
    y1 = np.clip(y1, 0, img_shape-1);

    wa = tf.convert_to_tensor((x1-x) * (y1-y), dtype=tf.float32)
    wb = tf.convert_to_tensor((x1-x) * (y-y0), dtype=tf.float32)
    wc = tf.convert_to_tensor((x-x0) * (y1-y), dtype=tf.float32)
    wd = tf.convert_to_tensor((x-x0) * (y-y0), dtype=tf.float32)
    wa = tf.expand_dims(wa, 1)
    wb = tf.expand_dims(wb, 1)
    wc = tf.expand_dims(wc, 1)
    wd = tf.expand_dims(wd, 1)

    data = data[0]

    iy0 = tf.expand_dims(y0, 1)
    ix0 = tf.expand_dims(x0, 1)
    iy1 = tf.expand_dims(y1, 1)
    ix1 = tf.expand_dims(x1, 1)

    i00 = tf.concat([iy0, ix0], 1)
    i01 = tf.concat([iy1, ix0], 1)
    i10 = tf.concat([iy0, ix1], 1)
    i11 = tf.concat([iy1, ix1], 1)

    x00 = tf.gather_nd(data, i00)
    x01 = tf.gather_nd(data, i01)
    x10 = tf.gather_nd(data, i10)
    x11 = tf.gather_nd(data, i11)

    print(x00)
    print(wa.shape)
    output = wa*x00 + wb*x01 + wc*x10 + wd*x11

    print(output)

    return output


import scipy.spatial as spatial
from sklearn.neighbors import KDTree


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def spatialNeighbours(uv_coords, samples=6000, radius=60, img_size=1000):
    #control_ids = np.random.random_integers(0, uv_coords.shape[0], samples)
    
    control_ids = []
    for i in range(uv_coords.shape[0]):
        if i % 101 == 0:
            control_ids.append(i)
    control_ids = np.asarray(control_ids, dtype=int)


    # visualize cluster centers
    blank_image = np.zeros((img_size,img_size,3), np.uint8)
    for v_id in control_ids:
        x, y = uv_coords[v_id]
        blank_image[img_size-int(y), int(x)] = (255, 255, 255)

    cv2.imwrite('/home/karim/Desktop/clusters.png', blank_image)
    #exit()


    point_tree = spatial.cKDTree(uv_coords)

    nearest_ind = []
    for v_id in control_ids:
        vn = point_tree.query_ball_point(uv_coords[v_id], radius)
        nearest_ind.append(vn)

    # build weight masks
    cluster_influence = np.zeros([control_ids.shape[0], uv_coords.shape[0], 3], dtype=np.float32)

    for i in range(control_ids.shape[0]):
        for j in range(len(nearest_ind[i])):
            dist = np.sum((uv_coords[nearest_ind[i][j]] - uv_coords[control_ids[i]])**2)**(0.5)
            cluster_influence[i, nearest_ind[i][j]] = 1 - dist / radius

    return control_ids, cluster_influence



def nearestNeighbours(uv_coords, samples=3000, k=4):
    control_ids = np.random.random_integers(0, uv_coords.shape[0], samples)
    #control_ids = np.asarray(range(0, uv_coords.shape[0]))

    tree = KDTree(uv_coords)
    nearest_dist, nearest_ind = tree.query(uv_coords[control_ids], k=int(uv_coords.shape[0]/samples))
    #nearest_dist, nearest_ind = tree.query(uv_coords[control_ids], k=k)

    # build weight masks
    cluster_influence = np.zeros([control_ids.shape[0], uv_coords.shape[0], 3], dtype=np.float32)

    for i in range(control_ids.shape[0]):
        mx_dist = np.amax(nearest_dist[i]) + 5
        for j in range(nearest_ind[i].shape[0]):
            cluster_influence[i, nearest_ind[i][j]] = 1 - nearest_dist[i][j] / mx_dist

    return control_ids, cluster_influence


def nearestNeighboursIds(uv_coords, k=8):
    tree = KDTree(uv_coords)
    nearest_dist, nearest_ind = tree.query(uv_coords, k=k)

    return nearest_ind

def nearestNeighboursIds3D(vertices, k=8):
    tree = KDTree(vertices)
    nearest_dist, nearest_ind = tree.query(vertices, k=k)

    return nearest_ind


def getMirrorIds(uv_coords, img_size = 1024):
    blank_image = np.zeros((img_size,img_size,3), np.uint8)

    left_ids = []
    left_face = []
    for uv_id in range(len(uv_coords)):
        x, y = uv_coords[uv_id] * img_size
        if x < int(img_size / 2) + 10:
            mirror_x = img_size - x
            left_face.append([mirror_x, y])
            left_ids.append(uv_id)
            # blank_image[img_size-int(y), int(mirror_x)] = (255, 255, 255)
    # cv2.imwrite('/home/karim/Desktop/clusters.png', blank_image)

    tree = KDTree(uv_coords * img_size)
    _, right_ids = tree.query(left_face, k=1)

    return left_ids, right_ids[:, 0]


def loadRig(uv_coords, path):
    clusters = []
    for file in sorted(glob.glob(os.path.join(path, '*.jpg'))):
        weights = np.asarray(cv2.imread(file), dtype=np.float32) / 255.
        size = weights.shape[0]
        cluster_influence = []
        for uv in uv_coords:
            w = weights[size - int(uv[1] * size), int(uv[0] * size)]
            cluster_influence.append(w)
        clusters.append(cluster_influence)

    # build weight masks
    cluster_influence = np.asarray(clusters, dtype=np.float32)

    return cluster_influence



if __name__ == '__main__':
    tf.reset_default_graph()
    tf.set_random_seed(125)


    BATCH_SIZE   = 3
    perspective  = False
    image_height = 256
    image_width  = 256
    flowimg_size = 1000
    path         = "../face3dMM/examples/Data/BFM/Out/BFM17Face_raw.mat"
    pth          = '/home/karim/Desktop/'
    imgs         = [pth + 'x_1.png', pth + 'x_2.png', pth + 'x_3.png', pth + 'face_6.png', pth + 'face_2.png', pth + 'face_3.png', pth + 'face_4.png', pth + 'face_5.png', pth + 'face_6.png', pth + 'face_7.png',  pth + 'face_8.png']
    
    #pth          = '/home/karim/Desktop/data/'
    #imgs         = [pth + 'img_1.png', pth + 'img_2.png', pth + 'img_3.png', pth + 'img_4.png', pth + 'img_5.png', pth + 'img_6.png', pth + 'img_7.png',  pth + 'img_8.png']
    imgs         = imgs[:BATCH_SIZE]


    bfm = MorphabelModel(path)
    uv_coords = load_uv_coords('/home/karim/Downloads/FaceUved.obj')
    
    # spatial neighbours
    control_ids, cluster_influence = spatialNeighbours(uv_coords * flowimg_size, img_size=flowimg_size)

    #control_ids, cluster_influence = nearestNeighbours(uv_coords * flowimg_size)
    #print(cluster_influence.shape)

    # load rig
    #cluster_influence = loadRig(uv_coords, '/home/karim/Desktop/data/rig')
    #print(cluster_influence.shape)

    ARGS_landmarks = bfm.landmarks
    TRGT_landmarks = bfm.landmarks_ids

    landmarks_weights = np.ones([68, 1], dtype=np.float32)
    landmarks_weights[:6]    = 0.3 # left cheeck
    landmarks_weights[10:16] = 0.3 # right cheeck
    landmarks_weights[7:9]   = 0.5 # bottom chin
    landmarks_weights = landmarks_weights[TRGT_landmarks]


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
    normal_field      = tf.Variable(tf.zeros([BATCH_SIZE, bfm.nver, 3]))


    # # control field
    # flow_field        = tf.Variable(tf.zeros([BATCH_SIZE, cluster_influence.shape[0], 3]))
    # flow_control = []
    # for b in range(BATCH_SIZE):
    #     fx = []
    #     for c in range(3):
    #         fx.append(tf.reduce_sum(tf.expand_dims(flow_field[b, :, c], 1) * cluster_influence[:, :, c], 0))
    #     fx = tf.stack(fx, axis=1)
    #     flow_control.append(fx)
    # flow_control = tf.stack(flow_control, axis=0)

    render, pvs, colr = fr.renderFaces(identity, expressions, pose, albedo, sh_coff, flow_field + normal_field, bfm, perspective, image_width, image_height)


    # Load real-image
    trgt_render, pvt = loadImgs(imgs, TRGT_landmarks, image_width, image_height)


    # mask target
    alpha       = render[:, :, :, -1]
    render      = render[:, :, :, :3]
    mask_r      = tf.boolean_mask(render, alpha)
    mask_t      = tf.boolean_mask(trgt_render, alpha)


    # loss function
    pixel_loss     = tf.reduce_mean(tf.square(mask_t - mask_r))
    landmarks_loss = tf.reduce_mean(tf.square(pvt - pvs) * landmarks_weights)
    reg_loss       = tf.reduce_sum(tf.square(albedo)) + tf.reduce_sum(tf.square(identity)) + tf.reduce_sum(tf.square(expressions))

    # Pose optimizer
    pose_loss = landmarks_loss + reg_loss
    pos_optimizer = tf.train.AdamOptimizer(0.05)
    pos_grads_and_vars = pos_optimizer.compute_gradients(pose_loss, [pose, identity, expressions])
    pos_opt_func = pos_optimizer.apply_gradients(pos_grads_and_vars)


    # Global fitting optimizer
    # loss = 1.1 * pixel_loss + 2.5e-5 * landmarks_loss + 5e-8 * reg_loss # normalized BFM
    loss = 1.1 * pixel_loss + 4e-4 * landmarks_loss + 1e-5 * reg_loss
    #loss = 1.1 * pixel_loss + 1.5e-4 * landmarks_loss + 1e-1 * reg_loss


    global_step = tf.train.get_or_create_global_step()
    decay_learning_rate = tf.train.exponential_decay(0.005, global_step, 400, 0.8, staircase=True)
    optimizer = tf.train.AdamOptimizer(decay_learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss, [pose, identity, albedo, expressions, sh_coff])
    opt_func = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


    # Flow field optimizer

    # laplacian regularizer
    Neighbours_size = 8
    bfmNp = MorphabelModelNP(path)
    base_ver = bfmNp.generate_vertices(bfmNp.get_shape_para('zero'), bfmNp.get_exp_para('zero'))
    neighbours_ids = nearestNeighboursIds3D(base_ver, Neighbours_size)
    # neighbours_ids = nearestNeighboursIds(uv_coords * flowimg_size, Neighbours_size)
    neighbours_ids = tf.reshape(neighbours_ids, [neighbours_ids.shape[0] * neighbours_ids.shape[1]])
    neighbours_flows = tf.gather(flow_field, neighbours_ids, axis=1)

    fids = range(0, bfm.nver)
    fids = np.repeat(fids, Neighbours_size)
    repeated_flow = tf.gather(flow_field, fids, axis=1)
    neighbours_diff = repeated_flow - neighbours_flows
    neighbours_diff = tf.reshape(neighbours_diff, [-1, bfm.nver, Neighbours_size, 3])
    neighbours_diff = tf.square(tf.reduce_sum(neighbours_diff, axis=2))
    smoothness_term = tf.reduce_mean(neighbours_diff)


    # mirror term
    left_face_ids, right_face_ids = getMirrorIds(uv_coords, flowimg_size)
    left_face   = tf.gather(flow_field, left_face_ids, axis=1)
    right_face  = tf.gather(flow_field, right_face_ids, axis=1)
    mirror_term = tf.reduce_mean(tf.square(left_face - right_face))

    flow_loss = 1.1 * pixel_loss + 1e-5 * landmarks_loss + 1e3 * smoothness_term + 7e4 * mirror_term
    # flow_loss =1.1 * pixel_loss + 1e-4 * landmarks_loss + 4e1 * tf.reduce_mean(tf.square(flow_field))

    flow_optimizer = tf.train.AdamOptimizer(0.001)
    flow_grads_and_vars = flow_optimizer.compute_gradients(flow_loss, [identity, albedo, expressions, pose, sh_coff, flow_field])
    flow_opt_func = flow_optimizer.apply_gradients(flow_grads_and_vars, global_step=global_step)



    neighbours_normals = tf.gather(normal_field, neighbours_ids, axis=1)

    fids = range(0, bfm.nver)
    fids = np.repeat(fids, Neighbours_size)
    repeated_normal = tf.gather(normal_field, fids, axis=1)
    neighbours_diff = repeated_normal - neighbours_normals
    neighbours_diff = tf.reshape(neighbours_diff, [-1, bfm.nver, Neighbours_size, 3])
    neighbours_diff = tf.square(tf.reduce_sum(neighbours_diff, axis=2))
    normal_smoothness_term = tf.reduce_mean(neighbours_diff)

    normal_loss = 1.1 * pixel_loss + 9e3 * tf.reduce_mean(tf.square(normal_field)) + 4 * normal_smoothness_term
    normal_optimizer = tf.train.AdamOptimizer(0.001)
    normal_grads_and_vars = normal_optimizer.compute_gradients(normal_loss, [normal_field])
    normal_opt_func = normal_optimizer.apply_gradients(normal_grads_and_vars, global_step=global_step)



    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

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
            grds = sess.run([grads_and_vars])
            #print(grds[0])
            prog_image, prog_lnd, trgt_image, trgt_lnd = sess.run([render, pvs, trgt_render, pvt])
            showImages(prog_image, trgt_image, prog_lnd, trgt_lnd, image_height, False)


        # Flow field fitting
        print("Flow field fitting")
        for i in tqdm(range(400)):
            lss, _ = sess.run([flow_loss, flow_opt_func])
            id_params, ep_params, alb_params, flow_params = sess.run([identity, expressions, colr, flow_field])
            prog_image, prog_lnd, trgt_image, trgt_lnd = sess.run([render, pvs, trgt_render, pvt])
            showImages(prog_image, trgt_image, prog_lnd, trgt_lnd, image_height, False)


        # # Shape from shading fitting
        # print("Normal field fitting")
        # for i in tqdm(range(200)):
        #     lss, _ = sess.run([normal_loss, normal_opt_func])
        #     normals = sess.run(normal_field)
        #     prog_image, prog_lnd, trgt_image, trgt_lnd = sess.run([render, pvs, trgt_render, pvt])
        #     showImages(prog_image, trgt_image, prog_lnd, trgt_lnd, image_height, False)



    # Save Obj file
    bfmNp = MorphabelModelNP(path)
    final_ver = bfmNp.generate_vertices(id_params[0], ep_params[0]) + flow_params[0]
    final_ver = final_ver / np.amax(final_ver) * 1000.
    final_alb = alb_params[0]
    #final_alb = bfmNp.generate_colors(alb_params[0])
    writeObj('/home/karim/Desktop/optimized_face.obj', final_ver, bfmNp.triangles, final_alb)
    print("Done :)")

    k = cv2.waitKey(0)
    if k == 27:
        exit()