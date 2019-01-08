import numpy as np
import scipy.io as sio
import h5py
import argparse


def singlePrecision(model):
    """
    cast to float32 to save memory
    """

    model['shapeMU'] = model['shapeMU'].astype(np.float32)
    model['shapePC'] = model['shapePC'].astype(np.float32)
    model['shapeEV'] = model['shapeEV'].astype(np.float32)
    model['texMU']   = model['texMU'].astype(np.float32)
    model['texPC']   = model['texPC'].astype(np.float32)
    model['texEV']   = model['texEV'].astype(np.float32)
    model['expPC']   = model['expPC'].astype(np.float32)
    model['expEV']   = model['expEV'].astype(np.float32)
    model['tri']     = model['tri'].T.astype(np.int32)

    return model


def normalizeModel(model):
    """
    scale factors to normalize stds - emperically estimated
    """

    SHAPE_SCALE = 0.35
    TEX_SCALE   = 0.35
    EXP_SCALE   = 1e-2 / 0.75
    MESH_SCALE  = 8e-06

    model['shapeEV'] = model['shapeEV'] * SHAPE_SCALE
    model['texEV']   = model['texEV'] * TEX_SCALE
    model['expEV']   = model['expEV'] * EXP_SCALE
    model['shapeMU'] = model['shapeMU'] * MESH_SCALE
    model['shapePC'] = model['shapePC'] * MESH_SCALE
    model['expPC']   = model['expPC'] * MESH_SCALE

    return model


def parseBFM2009(fmodel, fexpressions, finfo):
    """
    Read the face models from the Basel Face Model 2017 dataset
    """

    # load model
    model = sio.loadmat(fmodel)
    model['tri'] = model['tl'].T - 1
    del model['tl']
    del model['segbin']
    del model['segMB']
    del model['segMM']

    # load expressions
    if fexpressions and finfo:
        info = sio.loadmat(finfo)
        expressions = sio.loadmat(fexpressions)

        trimIndex = info['trimIndex'].astype(np.int32) - 1
        trimIndex_f = np.array([3 * trimIndex, 3 * trimIndex + 1, 3 * trimIndex + 2]).flatten('F')

        # remove vertices to match expressions
        model['shapeMU'] = model['shapeMU'][trimIndex_f,:]
        model['shapePC'] = model['shapePC'][trimIndex_f, :]
        model['texMU']   = model['texMU'][trimIndex_f, :]
        model['texPC']   = model['texPC'][trimIndex_f, :]
        model['tri']     = info['tri'] - 1
        # add expressions
        model['shapeMU'] = model['shapeMU'] + expressions['mu_exp']
        model['expPC']   = expressions['w_exp']
        model['expEV']   = expressions['sigma_exp']

        # ibug landmarks
        model['landmarks']     = info['keypoints']
        model['landmarks_ids'] = range(0, 68)

    return model



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pasrse and normalize BFM17 to a standard dictionary format.')
    parser.add_argument('--input', help = 'BFM09 path (.mat)', required = True)
    parser.add_argument('--output', help = 'Output path to save (.mat)', required = True)
    parser.add_argument('--expressions', help = 'model_expression from 3DDFA')
    parser.add_argument('--info', help = 'model_info from 3DDFA')
    args = parser.parse_args()

    model = parseBFM2009(args.input, args.expressions, args.info)
    model = normalizeModel(model)
    model = singlePrecision(model)
    sio.savemat(args.output, {'model' : model})