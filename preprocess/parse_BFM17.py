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

    # SHAPE_SCALE = 1e-2 / 5.25
    # EXP_SCALE   = 1e-2 / 3.0
    # TEX_SCALE   = 1e-1 / 1.45 * 1.4
    # MESH_SCALE  = 8e-03

    print(np.mean(model['shapeEV']))
    print(np.mean(model['expEV']))
    print(np.mean(model['texEV']))

    # FACTOR = 100
    FACTOR = 20
    SHAPE_SCALE = 1e-2 / 5.25 * FACTOR
    EXP_SCALE   = 1e-2 / 3.0 * FACTOR
    TEX_SCALE   = 1e-1 / 1.45 * 1.4 * FACTOR
    MESH_SCALE  = 8e-03

    model['shapeEV'] = model['shapeEV'] * SHAPE_SCALE
    model['expEV']   = model['expEV'] * EXP_SCALE
    model['texMU']   = model['texMU'] * 255.0
    model['texEV']   = model['texEV'] * 255.0 * TEX_SCALE
    model['shapeMU'] = model['shapeMU'] * MESH_SCALE
    model['shapePC'] = model['shapePC'] * MESH_SCALE
    model['expPC']   = model['expPC'] * MESH_SCALE

    print("")
    print(np.mean(model['shapeEV']))
    print(np.mean(model['expEV']))
    print(np.mean(model['texEV']))


    return model


def parseBFM2017(fmodel):
    """
    Read the face models from the Basel Face Model 2017 dataset
    """

    data = h5py.File(fmodel, 'r')
    
    # Identity
    idMean = np.empty(data.get('/shape/model/mean').shape)
    data.get('/shape/model/mean').read_direct(idMean)
    idVar = np.empty(data.get('/shape/model/noiseVariance').shape)
    data.get('/shape/model/noiseVariance').read_direct(idVar)
    idEvec = np.empty(data.get('/shape/model/pcaBasis').shape)
    data.get('/shape/model/pcaBasis').read_direct(idEvec)
    idEval = np.empty(data.get('/shape/model/pcaVariance').shape)
    data.get('/shape/model/pcaVariance').read_direct(idEval)

    # Expression
    expMean = np.empty(data.get('/expression/model/mean').shape)
    data.get('/expression/model/mean').read_direct(expMean)
    expVar = np.empty(data.get('/expression/model/noiseVariance').shape)
    data.get('/expression/model/noiseVariance').read_direct(expVar)
    expEvec = np.empty(data.get('/expression/model/pcaBasis').shape)
    data.get('/expression/model/pcaBasis').read_direct(expEvec)
    expEval = np.empty(data.get('/expression/model/pcaVariance').shape)
    data.get('/expression/model/pcaVariance').read_direct(expEval)

    # Texture
    texMean = np.empty(data.get('/color/model/mean').shape)
    data.get('/color/model/mean').read_direct(texMean)
    texVar = np.empty(data.get('/color/model/noiseVariance').shape)
    data.get('/color/model/noiseVariance').read_direct(texVar)
    texEvec = np.empty(data.get('/color/model/pcaBasis').shape)
    data.get('/color/model/pcaBasis').read_direct(texEvec)
    texEval = np.empty(data.get('/color/model/pcaVariance').shape)
    data.get('/color/model/pcaVariance').read_direct(texEval)

    # Triangle face indices
    face = np.empty(data.get('/shape/representer/cells').shape, dtype = 'int')
    data.get('/shape/representer/cells').read_direct(face)

    # create model
    model = {}
    # model['shapeMU'] = idMean[:, np.newaxis] + expMean[:, np.newaxis]
    # not adding expMean helps in having closed mouth
    model['shapeMU'] = idMean[:, np.newaxis]
    model['shapePC'] = idEvec
    model['shapeEV'] = idEval[:, np.newaxis]
    model['texMU']   = texMean[:, np.newaxis]
    model['texPC']   = texEvec
    model['texEV']   = texEval[:, np.newaxis]
    model['expPC']   = expEvec
    model['expEV']   = expEval[:, np.newaxis]
    model['tri']     = face

    # ibug landmarks for Face only version
    model['landmarks']     = [16203,16235,16260,16290,26869,27061,27253,22481,22451,22426,22394,22586,22991,23303,23519,23736,24312,24527,24743,25055,25466,8134,8143,8151,8157,6986,7695,8167,8639,9346,2602,4146,4920,5830,4674,3900,10390,11287,12061,13481,12331,11557,5522,6026,7355,8181,9007,10329,10857,9730,8670,8199,7726,6898,6291,7364,8190,9016,10088,8663,8191,7719]
    model['landmarks_ids'] = [0,1,2,3,7,8,9,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67]

    return model



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pasrse and normalize BFM17 to a standard dictionary format.')
    parser.add_argument('--input', help = 'BFM17 path (.h5)', required = True)
    parser.add_argument('--output', help = 'Output path to save (.mat)', required = True)
    args = parser.parse_args()

    model = parseBFM2017(args.input)
    model = normalizeModel(model)
    model = singlePrecision(model)
    sio.savemat(args.output, {'model' : model})