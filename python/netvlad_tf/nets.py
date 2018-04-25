import numpy as np
import os
from os.path import dirname
import tensorflow as tf

import netvlad_tf.layers as layers

def defaultCheckpoint():
    return os.path.join(dirname(dirname(dirname(__file__))), 
                              'checkpoints', 
                              'vd16_pitts30k_conv5_3_vlad_preL2_intra_white')

def vgg16NetvladPca(image_batch):
    ''' Assumes rank 4 input, first 3 dims fixed or dynamic, last dim 1 or 3. 
    '''
    assert len(image_batch.shape) == 4
    
    with tf.variable_scope('vgg16_netvlad_pca'):
        # Gray to color if necessary.
        if image_batch.shape[3] == 1:
            x = tf.nn.conv2d(image_batch, np.ones((1, 1, 1, 3)), 
                                  np.ones(4).tolist(), 'VALID')
        else :
            assert image_batch.shape[3] == 3
            x = image_batch
        
        # Subtract trained average image.
        average_rgb = tf.get_variable(
                'average_rgb', 3, dtype=image_batch.dtype)
        x = x - average_rgb
        
        # VGG16
        def vggConv(inputs, numbers, out_dim, with_relu):
            if with_relu:
                activation = tf.nn.relu
            else:
                activation = None
            return tf.layers.conv2d(inputs, out_dim, [3, 3], 1, padding='same',
                                    activation=activation, 
                                    name='conv%s' % numbers)
        def vggPool(inputs):
            return tf.layers.max_pooling2d(inputs, 2, 2)
        
        x = vggConv(x, '1_1', 64, True)
        x = vggConv(x, '1_2', 64, False)
        x = vggPool(x)
        x = tf.nn.relu(x)
        
        x = vggConv(x, '2_1', 128, True)
        x = vggConv(x, '2_2', 128, False)
        x = vggPool(x)
        x = tf.nn.relu(x)
        
        x = vggConv(x, '3_1', 256, True)
        x = vggConv(x, '3_2', 256, True)
        x = vggConv(x, '3_3', 256, False)
        x = vggPool(x)
        x = tf.nn.relu(x)
        
        x = vggConv(x, '4_1', 512, True)
        x = vggConv(x, '4_2', 512, True)
        x = vggConv(x, '4_3', 512, False)
        x = vggPool(x)
        x = tf.nn.relu(x)
        
        x = vggConv(x, '5_1', 512, True)
        x = vggConv(x, '5_2', 512, True)
        x = vggConv(x, '5_3', 512, False)
        
        # NetVLAD
        x = tf.nn.l2_normalize(x, dim=-1)
        x = layers.netVLAD(x, 64)
        
        # PCA
        x = tf.layers.conv2d(tf.expand_dims(tf.expand_dims(x, 1), 1), 
                             4096, 1, 1, name='WPCA')
        x = tf.nn.l2_normalize(tf.layers.flatten(x), dim=-1)
        
    return x