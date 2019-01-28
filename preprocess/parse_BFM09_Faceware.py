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

    FACTOR = 3e1
    SHAPE_SCALE = 0.35 * FACTOR
    EXP_SCALE   = 1e4 * 7 * FACTOR
    TEX_SCALE   = 0.35 * FACTOR / 3
    MESH_SCALE  = 8e-06

    FACTOR = 4
    SHAPE_SCALE = 1 * FACTOR * 3
    EXP_SCALE   = 1e4 * 6 * FACTOR * 4
    TEX_SCALE   = 1 * FACTOR
    MESH_SCALE  = 8e-06

    print(np.mean(model['shapeEV']))
    print(np.mean(model['expEV']))
    print(np.mean(model['texEV']))


    model['shapeEV'] = model['shapeEV'] * SHAPE_SCALE
    model['texEV']   = model['texEV'] * TEX_SCALE
    model['expEV']   = model['expEV'] * EXP_SCALE
    model['shapeMU'] = model['shapeMU'] * MESH_SCALE
    model['shapePC'] = model['shapePC'] * MESH_SCALE
    model['expPC']   = model['expPC'] * MESH_SCALE

    print("")
    print(np.mean(model['shapeEV']))
    print(np.mean(model['expEV']))
    print(np.mean(model['texEV']))
    print(np.mean(model['shapePC']))
    print(np.mean(model['expPC']))
    print(np.mean(model['texPC']))

    return model


def parseBFM2009(fmodel, fexpressions, fblink, mask):
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

    # ibug landmarks
    model['landmarks']     = [22143,22813,22840,23250,44124,45884,47085,47668,48188,48708,49299,50498,52457,32022,32386,32359,32979,38886,39636,40030,40238,40433,41172,41368,41578,42011,42646,8291,8305,8314,8320,6783,7687,8331,8977,9879,1832,3760,5050,6087,4546,3516,10731,11758,12919,14859,13191,12157,5523,6155,7442,8345,9506,10799,11199,10179,9277,8374,7471,6566,5909,7322,8354,9386,10941,9141,8367,7194]
    model['landmarks_ids'] = range(0, 68)

    # load expressions
    if fexpressions:
        expressions = sio.loadmat(fexpressions)
        # remove first column as PC basis are zeros
        model['expPC']   = expressions['expPC'][:, 1:]
        model['expEV']   = expressions['expEV'][1:]

    if fblink:
        blinking = sio.loadmat(fblink)
        # append blinking expressions
        model['expPC']   = np.c_[blinking['right_eye'], model['expPC']]
        model['expPC']   = np.c_[blinking['left_eye'], model['expPC']]
        model['expEV']   = np.r_[[1.2, 1.2], model['expEV'][:, 0]][:, np.newaxis]


    if mask:
        mask = sio.loadmat(mask)

        # Remove mouth
        trimIndex = mask['trimIndex'].astype(np.int32) - 1
        trimIndex_f = np.array([3 * trimIndex, 3 * trimIndex + 1, 3 * trimIndex + 2]).flatten('F')

        # remove vertices to match expressions
        model['shapeMU'] = model['shapeMU'][trimIndex_f,:]
        model['shapePC'] = model['shapePC'][trimIndex_f, :]
        model['texMU']   = model['texMU'][trimIndex_f, :]
        model['texPC']   = model['texPC'][trimIndex_f, :]
        model['tri']     = mask['tri'] - 1
        model['expPC']   = model['expPC'][trimIndex_f, :]


        # Mask face
        vert_trimIndex = mask['ver_indices'].astype(np.int32).T
        vert_trimIndex_f = np.array([3 * vert_trimIndex, 3 * vert_trimIndex + 1, 3 * vert_trimIndex + 2]).flatten('F')

        # mask facemodel
        model['shapeMU'] = model['shapeMU'][vert_trimIndex_f,:]
        model['shapePC'] = model['shapePC'][vert_trimIndex_f, :]
        model['texMU']   = model['texMU'][vert_trimIndex_f, :]
        model['texPC']   = model['texPC'][vert_trimIndex_f, :]
        model['expPC']   = model['expPC'][vert_trimIndex_f, :]
        model['tri']     = mask['new_tris'] - 1

        # ibug landmarks
        model['landmarks']     = [16361,16469,16727,16885,24090,24591,25126,25379,25630,26178,26667,20755,20855,21081,21152,21442,21829,22093,22247,22380,22996,23127,23281,23544,23927,8153,8169,8181,8189,6758,7722,8203,8682,9641,2216,3887,4921,5829,4802,3641,10456,11354,12384,14067,12654,11493,5523,6026,7496,8216,8936,10396,10796,9556,8837,8237,7637,6916,5910,7386,8226,9066,10538,8828,8228,7628]
        model['landmarks_ids'] = [0,1,2,3,4,5,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67]


    return model



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pasrse and normalize BFM17 to a standard dictionary format.')
    parser.add_argument('--input', help = 'BFM09 path (.mat)', required = True)
    parser.add_argument('--output', help = 'Output path to save (.mat)', required = True)
    parser.add_argument('--expressions', help = 'model_expression from faceware')
    parser.add_argument('--blinking_expressions', help = 'model_expression from faceware')
    parser.add_argument('--mask', help = 'face_maask from 3DDFA')


    args = parser.parse_args()

    model = parseBFM2009(args.input, args.expressions, args.blinking_expressions, args.mask)
    model = normalizeModel(model)
    model = singlePrecision(model)
    sio.savemat(args.output, {'model' : model})