from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as sio


def load_BFM(model_path):
    ''' load BFM 3DMM model
    Args:
        model_path: path to BFM model. 
    Returns:
        model: (nver = 53215, ntri = 105840). nver: number of vertices. ntri: number of triangles.
            'shapeMU': [3*nver, 1]
            'shapePC': [3*nver, 199]
            'shapeEV': [199, 1]
            'expMU': [3*nver, 1]
            'expPC': [3*nver, 29]
            'expEV': [29, 1]
            'texPC': [3*nver, 199]
            'texEV': [199, 1]
            'tri': [ntri, 3] (start from 1, should sub 1 in python and c++)
    PS:
        You can change codes according to your own saved data.
        Just make sure the model has corresponding attributes.
    '''

    C = sio.loadmat(model_path)
    model = C['model']
    model = model[0,0]

    # change dtype from double(np.float64) to np.float32, 
    # since big matrix process(espetially matrix dot) is too slow in python.
    model['shapeMU']       = model['shapeMU'].astype(np.float32)
    model['shapePC']       = model['shapePC'].astype(np.float32)
    model['shapeEV']       = model['shapeEV'].astype(np.float32)
    model['expEV']         = model['expEV'].astype(np.float32)
    model['expPC']         = model['expPC'].astype(np.float32)
    model['texMU']         = model['texMU'].astype(np.float32)
    model['texPC']         = model['texPC'].astype(np.float32)
    model['texEV']         = model['texEV'].astype(np.float32)
    model['tri']           = model['tri'].astype(np.int32)
    model['landmarks']     = model['landmarks'].astype(np.int32)
    model['landmarks_ids'] = model['landmarks_ids'].astype(np.int32)

    return model


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


def load_pncc_code(path = 'pncc_code.mat'):
    ''' load pncc code of BFM
    PNCC code: Defined in 'Face Alignment Across Large Poses: A 3D Solution Xiangyu'
    download at http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm.
    Args:
        path: path to data.
    Returns:  
        pncc_code: [nver, 3]
    '''

    C = sio.loadmat(path)
    pncc_code = C['vertex_code'].T

    return pncc_code