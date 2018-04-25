import numpy as np
import os
from os.path import dirname
import scipy.io as scio
import tensorflow as tf

import netvlad_tf.layers as layers

#%% Spyder section for debugging.

def structedMatPath():
    return os.path.join(dirname(dirname(dirname(__file__))), 'matlab',
                        'structed.mat')

def exampleImgPath():
    return os.path.join(dirname(dirname(dirname(__file__))), 'example.jpg')

def exampleStatPath():
    return os.path.join(dirname(dirname(dirname(__file__))), 'matlab',
                        'example_stats.mat')

def netFromMat():
    ''' Method for parsing vd16_pitts30k_conv5_3_vlad_preL2_intra_white.mat ,
    probably also others, but not tested. Requires structed.mat in the matlab
    folder, which can be generated with matlab/net_class2struct.m, otherwise
    python can't read the parameteres of the custom layers. '''
    #%% Load mat from netvlad.
    mat = scio.loadmat(structedMatPath(),
                       struct_as_record=False, squeeze_me=True)
    matnet = mat['net']
    mat_layers = matnet.layers

    #%% Spyder section for debugging.
    tf_layers = [tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])]
    
    with tf.variable_scope('vgg16_netvlad_pca'):
        # Additional layer for average image normalization.
        average_rgb = tf.get_variable(
                'average_rgb', 3, initializer=tf.constant_initializer(
                        matnet.meta.normalization.averageImage[0,0,:]))
        tf_layers.append(tf_layers[-1] - average_rgb)


        for i in range(len(mat_layers)):
            layer = mat_layers[i]
            # make name TF-friendly:
            layer.name = layer.name.replace(':', '_')

            # Print layer info
            assert hasattr(layer, 'name')
            print('%02d: %s: %s' % (i + 1, layer.name, layer.type))

            if layer.type == 'conv':
                w = layer.weights[0]
                b = layer.weights[1]
                if len(tf_layers[-1].shape) == 4:
                    assert np.all(layer.pad == 1)
                    tf_layers.append(tf.layers.conv2d(
                                tf_layers[-1], b.size, w.shape[:2],
                                strides=layer.stride,
                                padding='same',
                                activation=None,
                                kernel_initializer=tf.constant_initializer(w),
                                bias_initializer=tf.constant_initializer(b),
                                name=layer.name))
                else:
                    # PCA convolution
                    assert len(tf_layers[-1].shape) == 2
                    assert layer.name == 'WPCA'
                    assert layer.pad == 0
                    w = np.expand_dims(np.expand_dims(w, 0), 0)
                    tf_layers.append(tf.layers.conv2d(
                                tf.expand_dims(tf.expand_dims(
                                        tf_layers[-1], 1), 1),
                                b.size, w.shape[:2],
                                strides=layer.stride,
                                padding='valid',
                                activation=None,
                                kernel_initializer=tf.constant_initializer(w),
                                bias_initializer=tf.constant_initializer(b),
                                name=layer.name))

            elif layer.type == 'relu':
                assert layer.leak == 0
                tf_layers.append(tf.nn.relu(tf_layers[-1]))

            elif layer.type == 'pool':
                assert layer.method == 'max'
                assert np.all(layer.pad == 0)
                tf_layers.append(tf.layers.max_pooling2d(
                        tf_layers[-1], layer.pool, layer.stride,
                        name=layer.name))

            elif layer.type == 'normalize':
                p = layer.param
                # Asserting desired normalization is l2 accross all layers.
                # See http://www.vlfeat.org/matconvnet/mfiles/vl_nnnormalize/
                assert np.all(p[[0, 2, 3]] == np.array([1024, 1, 0.5]))
                tf_layers.append(layers.matconvnetNormalize(
                        tf_layers[-1], p[1]))

            elif layer.type == 'custom':
                if layer.name == 'vlad_core':
                    a = layer.weights[0]
                    c = layer.weights[1]
                    tf_layers.append(layers.netVLAD(
                        tf_layers[-1], layer.K,
                        assign_weight_initializer=tf.constant_initializer(a),
                        cluster_initializer=tf.constant_initializer(c),
                        skip_postnorm=True))
                elif layer.name == 'postL2':
                    reshaped = tf.transpose(tf_layers[-1], perm=[0, 2, 1])
                    tf_layers.append(layers.matconvnetNormalize(
                        tf.layers.flatten(reshaped), 1e-12))
                elif layer.name == 'finalL2':
                    tf_layers.append(layers.matconvnetNormalize(
                        tf.layers.flatten(tf_layers[-1]), 1e-12))
                else:
                    raise Exception('Unknown custom layer %s' % layer.name)

            else:
                raise Exception('Unknown layer type %s' % layer.type)

            print(tf_layers[-1].shape)
    #%% Spyder section for debugging.

    return tf_layers
