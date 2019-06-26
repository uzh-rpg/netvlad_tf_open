import cv2
import glob
import numpy as np
import os
import tensorflow as tf

import netvlad_tf.nets as nets


class ImageDescriptor(object):

    def __init__(self, is_grayscale=False):
        self.is_grayscale = is_grayscale
        if is_grayscale:
            self.tf_batch = tf.placeholder(
                    dtype=tf.float32, shape=[None, None, None, 1])
        else:
            self.tf_batch = tf.placeholder(
                    dtype=tf.float32, shape=[None, None, None, 3])
        self.net_out = nets.vgg16NetvladPca(self.tf_batch)
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, nets.defaultCheckpoint())

    def describeAllJpegsInPath(self, path, batch_size, verbose=False):
        ''' returns a list of descriptors '''
        jpeg_paths = sorted(glob.glob(os.path.join(path, '*.jpg')))
        descs = []
        for batch_offset in range(0, len(jpeg_paths), batch_size):
            images = []
            for i in range(batch_offset, batch_offset + batch_size):
                if i == len(jpeg_paths):
                    break
                if verbose:
                    print('%d/%d' % (i, len(jpeg_paths)))
                if self.is_grayscale:
                    image = cv2.imread(jpeg_paths[i], cv2.IMREAD_GRAYSCALE)
                    images.append(np.expand_dims(
                            np.expand_dims(image, axis=0), axis=-1))
                else:
                    image = cv2.imread(jpeg_paths[i])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(np.expand_dims(image, axis=0))
            batch = np.concatenate(images, 0)
            descs = descs + list(self.sess.run(
                    self.net_out, feed_dict={self.tf_batch: batch}))
        return descs

    def describe(self, image):
        if self.is_grayscale:
            batch = np.expand_dims(np.expand_dims(image, axis=0), axis=-1)
        else:
            batch = np.expand_dims(image, axis=0)
        return self.sess.run(
            self.net_out, feed_dict={self.tf_batch: batch}).squeeze()
