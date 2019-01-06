from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from . import load
# from .. import mesh

class  MorphabelModel(object):
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
        super( MorphabelModel, self).__init__()
        if model_type=='BFM':
            self.model = load.load_BFM(model_path)
        else:
            print('sorry, not support other 3DMM model now')
            exit()
            
        # fixed attributes
        self.nver = int(self.model['shapePC'].shape[0]/3)
        self.ntri = self.model['tri'].shape[0]
        self.n_shape_para = self.model['shapePC'].shape[1]
        self.n_exp_para = self.model['expPC'].shape[1]
        self.n_tex_para = self.model['texPC'].shape[1]
        self.triangles = self.model['tri']


        # # Find the face indices associated with each vertex (for norm calculation)
        self.vertex2face = np.array([np.where(np.isin(self.triangles.T, vertexInd).any(axis = 0))[0] for vertexInd in range(self.nver)])
        #np.save('/home/karim/Documents/Development/FacialCapture/FaceGen/vetrex2face.npy', self.vertex2face)
        #print("Done with V2F")
        #self.vertex2face = np.load('/home/karim/Documents/Development/FacialCapture/FaceGen/vetrex2face.npy')

        # limit PCA params
        #self.n_shape_para = 80
        #self.n_tex_para = 80


    # ------------------------------------- shape: represented with mesh(vertices & triangles(fixed))
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


    def get_tex_para(self, type = 'random', std = 1.2):
        if type == 'zero':
            tx = np.zeros((self.n_tex_para, 1))
        elif type == 'random':
            tx = np.random.uniform(-std, std, [self.n_tex_para, 1])

        return tx 

    def generate_vertices(self, shape_para, exp_para):
        '''
        Args:
            shape_para: (n_shape_para, 1)
            exp_para: (n_exp_para, 1) 
        Returns:
            vertices: (nver, 3)
        '''

        if len(shape_para.get_shape()) == 1:
            shape_para = tf.expand_dims(shape_para, 1)
        if len(exp_para.get_shape()) == 1:
            exp_para = tf.expand_dims(exp_para, 1)

        # multiply by Std to have normalized params
        shape_para = shape_para * self.model['shapeEV'][:self.n_shape_para]
        exp_para = exp_para * self.model['expEV'][:self.n_exp_para]
        shape_influence = tf.tensordot(self.model['shapePC'][:, :self.n_shape_para], shape_para, 1)
        exp_influence = tf.tensordot(self.model['expPC'][:, :self.n_exp_para], exp_para, 1)
        vertices = self.model['shapeMU'] + shape_influence + exp_influence
        vertices = tf.reshape(vertices, [int(self.nver), 3])

        return vertices


    def generate_colors(self, tex_para):
        '''
        Args:
            tex_para: (n_tex_para, 1)
        Returns:
            colors: (nver, 3)
        '''

        if len(tex_para.get_shape()) == 1:
            tex_para = tf.expand_dims(tex_para, 1)

        colors = self.model['texMU'] + tf.tensordot(self.model['texPC'][:, :self.n_tex_para], tex_para*self.model['texEV'][:self.n_tex_para], 1)
        colors = tf.reshape(colors, [int(self.nver), 3]) / 255.
        
        return colors