from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf

import tf_mesh_renderer.mesh_renderer as mesh_renderer
import tf_mesh_renderer.camera_utils as camera_utils



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

    # A workaround to allow differentiable calculation
    # expand vertex2face to a matrix by referencing a tmp face with 0 normal
    tri_normal = tf.concat([tri_normal, [[0, 0, 0]]], axis=0) # add a zero normal to the end
    zero_indx  = tri_normal.get_shape()[0] - 1  # (nver + 1) -1

    v2f = np.empty([bfm.nver, 8], dtype=int)
    for x in range(bfm.vertex2face.shape[0]):
        sz = bfm.vertex2face[x].shape[0]
        v2f[x] = np.concatenate((bfm.vertex2face[x], np.ones(8-sz) * zero_indx))

    normals = tf.gather(tri_normal, v2f)
    normals = tf.reduce_sum(normals, axis=1)
    normals = tf.nn.l2_normalize(normals, axis=1)

    return normals


def sh9(normals):
    """
    First nine spherical harmonics as functions of Cartesian coordinates
    """

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


def renderFace(identity, albedo, expressions, pose, sh_coff, flow_field,
 bfm, perspective=True, image_width = 256, image_height = 256):
    # variables
    if not identity:
        identity = np.float32(bfm.get_shape_para('zero', 2))
    if not albedo:
        albedo = np.float32(bfm.get_tex_para('zero', 3))
    if not expressions:
        expressions = np.float32(bfm.get_exp_para('zero', 3))
    if not pose:
        pose = np.zeros(6, dtype=np.float32)
        if not perspective:
            pose[-1] = 1
    if not sh_coff:
        sh_coff = np.zeros([9, 3], dtype=np.float32)
        sh_coff[0, 0] = 1.0
        sh_coff[0, 1] = 1.0
        sh_coff[0, 2] = 1.0


    # expand dims
    identity    = tf.expand_dims(identity, 0)
    albedo      = tf.expand_dims(albedo, 0)
    expressions = tf.expand_dims(expressions, 0)
    pose        = tf.expand_dims(pose, 0)
    sh_coff     = tf.expand_dims(sh_coff, 0)

    render, projected_landmarks, faces_colors = renderFaces(identity, expressions, pose, albedo, sh_coff, flow_field,
                                                                                            bfm, perspective, image_width, image_height, batch_size=1)

    return render[0], projected_landmarks[0], faces_colors[0]



def renderFaces(identity, expressions, pose, albedo, sh_coff, flow_field,
                bfm, perspective=False, image_width = 256, image_height = 256,
                batch_size=None): 

    """ Generate and render faces given the model parameters.
    Identity, expressions, pose and albedo are normalized (multiplied by Std)
    Pose == [euler rotation, translation, scale]
    - rotations are multiplied by PI
    - scale is shifted by 1 for stable training 

    """

    if batch_size is None:
        batch_size = identity.shape[0].value       

    # calculate shape from identity & expressions
    faces_vertices = []
    for i in xrange(batch_size):
        faces_vertices.append(bfm.generate_vertices(identity[i], expressions[i]))
    faces_vertices = tf.reshape(faces_vertices, [batch_size, bfm.nver, 3])

    # apply 3D flow field
    if flow_field is not None:
        faces_vertices = faces_vertices + flow_field

    # calculate normals
    faces_normals = []
    for i in xrange(batch_size):
        faces_normals.append(get_normals(faces_vertices[i], bfm.triangles, bfm))
    faces_normals = tf.reshape(faces_normals, [batch_size, bfm.nver, 3])

    # calculate colors
    faces_colors = []
    for i in xrange(batch_size):
        faces_colors.append(bfm.generate_colors(albedo[i]))
    faces_colors = tf.reshape(faces_colors, [batch_size, bfm.nver, 3])


    # Spherical harmonics - loop approach
    albedo_colors = tf.identity(faces_colors)
    faces_colors = []
    for i in xrange(batch_size):
        face_colors = []
        for c in range(3):
            color_channel = tf.tensordot(sh9(faces_normals[i]), sh_coff[i, :, c], 1) * albedo_colors[i, :, c]
            face_colors.append(color_channel)
        face_colors = tf.stack(face_colors, axis=1)
        faces_colors.append(face_colors)
    faces_colors = tf.stack(faces_colors, axis=0)


    # apply pose
    if pose.get_shape()[1] == 6:
        model_rotation  = camera_utils.euler_matrices(pose[:, :3] * math.pi)[:, :4, :3]
        model_translate = pose[:, 3:]
        if not perspective:
            scale = tf.reshape(pose[:, -1], [batch_size, 1, 1]) + 1.
            faces_vertices = tf.multiply(faces_vertices, scale)
            model_translate = tf.concat([pose[:, 3:5], tf.zeros([batch_size, 1])], axis = 1)

        model_translate = tf.expand_dims(tf.concat([model_translate, tf.ones([batch_size, 1])], axis = 1), -1)
        model_transform = tf.concat([model_rotation, model_translate], axis=2)
        # change to homogenous c.o.s
        faces_vertices = tf.concat([faces_vertices, tf.ones([batch_size, bfm.nver, 1])], axis = -1)
        faces_vertices = tf.matmul(faces_vertices, model_transform, transpose_b=True)[:, :, :3]


    eye = tf.constant([0.0, 0.0, 3.0], dtype=tf.float32)
    center = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
    world_up = tf.constant([0.0, 1.0, 0.0], dtype=tf.float32)

    render, projected_vertices = mesh_renderer.render(faces_vertices, bfm.triangles, faces_normals,
                                                      faces_colors, eye, center, world_up, image_width, image_height,
                                                      perspective=perspective)

    # get projected landmarks and flip them along Y-axis
    projected_landmarks = tf.gather(projected_vertices, bfm.landmarks, axis=1)[:, :, :2]
    projected_landmarks = projected_landmarks * image_height / 2. + image_height / 2.

    return render, projected_landmarks, faces_colors