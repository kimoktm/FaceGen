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


        # Find the face indices associated with each vertex (for norm calculation)
        self.vertex2face = np.array([np.where(np.isin(self.triangles.T, vertexInd).any(axis = 0))[0] for vertexInd in range(self.nver)])


        close_mouth = [-1.5090884,-0.15453044,-3.8402557,0.9445557,-1.0569872,1.6198115,-2.239671,-3.6464038,0.6600208,-2.6687174,0.8665768,-3.064968,-1.8411735,-3.880725,2.5869136,-3.0662332,1.6370867,5.5037694,-6.1639357,-5.0047746,-0.3380434,2.9114015,-3.2750301,4.805709,4.3729453,-4.95749,-3.4698102,-2.7745442,3.8737736,4.2599225,-4.411754,1.9909493,5.938608,6.055374,1.3890951,-6.7608314,-2.5041585,5.082385,-5.2096877,4.6854486,3.7670238,3.5578146,-1.1065546,2.6221838,4.11583,-3.3964825,-0.8175518,-5.8215733,4.736252,-2.8394568,-0.63069046,-2.6047778,5.224814,3.3610501,5.473414,5.044716,-4.5016246,3.7234306,-3.8046336,-8.1779585,-2.0457428,-5.306759,-4.610994,-0.6649688,3.8743174,-3.1714785,6.8948298,4.324879,1.8900679,-6.17165,-5.660559,4.2499413,1.7972592,7.2815857,-2.2530048,5.178456,-6.111877,1.4841856,4.9566426,-3.685445,5.26457,-6.939066,-4.6458645,6.5899262,-5.391815,-5.9794455,6.170348,-8.342265,5.389521,3.9339674,-5.3724246,-5.079346,-4.1542244,0.20411202,3.0264375,-0.7128217,-4.4631314,-4.9623137,5.0232368,-4.9445524]
        close_mouth = np.asarray(close_mouth, dtype=np.float32)
        close_mouth = np.expand_dims(close_mouth, 1)
        #self.model['shapeMU'] = self.model['shapeMU'] + self.model['expPC'].dot(close_mouth * self.model['expEV'])

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