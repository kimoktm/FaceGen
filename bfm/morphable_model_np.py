from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.neighbors import KDTree

from . import load


class  MorphabelModelNP(object):
    """docstring for  MorphabelModel
    model: nver: number of vertices. ntri: number of triangles. *: must have. ~: can generate ones array for place holder.
            'shapeMU': [3*nver, 1]. *
            'shapePC': [3*nver, n_shape_para]. *
            'shapeEV': [n_shape_para, 1]. ~
            'expMU': [3*nver, 1]. ~ 
            'expPC': [3*nver, n_exp_para]. ~
            'expEV': [n_exp_para, 1]. ~
            'texMU': [3*nver, 1]. ~
            'texPC': [3*nver, n_tex_para]. ~
            'texEV': [n_tex_para, 1]. ~
            'tri': [ntri, 3] (start from 1, should sub 1 in python and c++). *
            'tri_mouth': [114, 3] (start from 1, as a supplement to mouth triangles). ~
            'kpt_ind': [68,] (start from 1). ~
    """

    def __init__(self, model_path, model_type = 'BFM'):
        super( MorphabelModelNP, self).__init__()
        if model_type=='BFM':
            self.model = load.load_BFM(model_path)
        else:
            print('sorry, not support other 3DMM model now')
            exit()
            
        # fixed attributes
        self.nver          = int(self.model['shapePC'].shape[0]/3)
        self.ntri          = self.model['tri'].shape[0]
        self.n_shape_para  = self.model['shapePC'].shape[1]
        self.n_exp_para    = self.model['expPC'].shape[1]
        self.n_tex_para    = self.model['texPC'].shape[1]
        self.triangles     = self.model['tri']
        self.landmarks     = self.model['landmarks']
        self.landmarks_ids = self.model['landmarks_ids']

        # Find the face indices associated with each vertex (for fast & differentiable normals calculation)
        self.vertex2face = np.array([np.where(np.isin(self.triangles.T, vertexInd).any(axis = 0))[0] for vertexInd in range(self.nver)])


    def get_shape_para(self, type = 'random', std = 1.2):
        if type == 'zero':
            sp = np.zeros((self.n_shape_para, 1))
        elif type == 'random':
            sp = np.random.uniform(-std, std, [self.n_shape_para, 1])

        return sp


    def get_exp_para(self, type = 'random', std = 1.2):
        if type == 'zero':
            ep = np.zeros((self.n_exp_para, 1))
        elif type == 'random':
            ep = np.random.uniform(-std, std, [self.n_exp_para, 1])

        return ep 


    def generate_vertices(self, shape_para, exp_para):
        '''
        Args:
            shape_para: (n_shape_para, 1)
            exp_para: (n_exp_para, 1) 
        Returns:
            vertices: (nver, 3)
        '''

        if len(shape_para.shape) == 1:
            shape_para = np.expand_dims(shape_para, 1)
        if len(exp_para.shape) == 1:
            exp_para = np.expand_dims(exp_para, 1)

        vertices = self.model['shapeMU'] + self.model['shapePC'][:, :self.n_shape_para].dot(shape_para * self.model['shapeEV'][:self.n_shape_para])
        vertices = vertices + self.model['expPC'].dot(exp_para * self.model['expEV'][:self.n_exp_para])
        vertices = np.reshape(vertices, [int(3), int(len(vertices)/3)], 'F').T

        return vertices


    def generate_colors(self, tex_para):
        '''
        Args:
            tex_para: (n_tex_para, 1)
        Returns:
            colors: (nver, 3)
        '''

        if len(tex_para.shape) == 1:
            tex_para = np.expand_dims(tex_para, 1)

        colors = self.model['texMU'] + self.model['texPC'][:, :self.n_tex_para].dot(tex_para * self.model['texEV'][:self.n_tex_para])
        colors = np.reshape(colors, [int(3), int(len(colors)/3)], 'F').T/255.  

        return colors


    def get_nearest_neighbours(self, neighbours_num):
        '''
        Args:
            neighbours_num: int
        Returns:
            neighbours_ids: (nver, neighbours_num)
        '''

        # slightly open mouth & closed eyes
        exp_para     = self.get_exp_para('zero')
        exp_para[0]  = -0.24
        exp_para[1]  = -0.24
        exp_para[12] = 0.5

        vertices = self.model['shapeMU'] + self.model['expPC'].dot(exp_para * self.model['expEV'][:self.n_exp_para])
        vertices = np.reshape(vertices, [int(3), int(len(vertices)/3)], 'F').T
        _, neighbours_ids = KDTree(vertices).query(vertices, k=neighbours_num)

        return neighbours_ids