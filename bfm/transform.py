import tensorflow as tf


def tf_deg2rad(deg):
    pi_on_180 = 0.017453292519943295

    return deg * pi_on_180


def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch. positive for looking down.
        y: yaw. positive for looking left. 
        z: roll. positive for tilting head right. 
    Returns:
        R: [3, 3]. rotation matrix.
    '''

    # angles = tf.cast(angles, dtype=tf.float32)
    x, y, z = tf_deg2rad(angles[0]), tf_deg2rad(angles[1]), tf_deg2rad(angles[2])

    # x
    Rx = tf.stack([1,      0,       0,
                   0, tf.cos(x),  -tf.sin(x),
                   0, tf.sin(x),   tf.cos(x)])
    Rx = tf.reshape(Rx, (3,3))

    # y
    Ry = tf.stack([tf.cos(y), 0, tf.sin(y),
                   0,         1,      0,
                  -tf.sin(y), 0, tf.cos(y)])
    Ry = tf.reshape(Ry, (3,3))

    # z
    Rz = tf.stack([tf.cos(z), -tf.sin(z), 0,
                  tf.sin(z),  tf.cos(z), 0,
                   0,       0, 1])
    Rz = tf.reshape(Rz, (3,3))


    R = tf.tensordot(Rz, tf.tensordot(Ry, Rx, 1), 1)

    return R


def rotate(vertices, angles):
    ''' rotate vertices. 
    X_new = R.dot(X). X: 3 x 1   
    Args:
        vertices: [nver, 3]. 
        rx, ry, rz: degree angles
        rx: pitch. positive for looking down 
        ry: yaw. positive for looking left
        rz: roll. positive for tilting head right
    Returns:
        rotated vertices: [nver, 3]
    '''

    # vertices = tf.cast(vertices, dtype=tf.float32)

    R = angle2matrix(angles)
    rotated_vertices = tf.tensordot(vertices, tf.transpose(R), 1)

    return rotated_vertices


def similarity_transform(vertices, s, angles, t3d):
    ''' similarity transform. dof = 7.
    3D: s*R.dot(X) + t
    Homo: M = [[sR, t],[0^T, 1]].  M.dot(X)
    Args:(float32)
        vertices: [nver, 3]. 
        s: [1,]. scale factor.
        R: [3,3]. rotation matrix.
        t3d: [3,]. 3d translation vector.
    Returns:
        transformed vertices: [nver, 3]
    '''

    t3d = tf.squeeze(t3d)
    transformed_vertices = s * rotate(vertices, angles) + t3d

    return transformed_vertices