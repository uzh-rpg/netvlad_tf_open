import netvlad_tf.nets as nets

import cv2
import numpy as np
import scipy.io as scio
import tensorflow as tf
import time
import unittest

import netvlad_tf.net_from_mat as nfm

class TestNets(unittest.TestCase):
    def testVgg16NetvladPca(self):
        ''' Need example_stats.mat in matlab folder, which can be generated
        with get_example_stats.m. Also need translated checkpoint, can be
        generated with mat_to_checkpoint.py. '''
        tf.reset_default_graph()

        image_batch = tf.placeholder(
                dtype=tf.float32, shape=[None, None, None, 3])

        net_out = nets.vgg16NetvladPca(image_batch)
        saver = tf.train.Saver()

        sess = tf.Session()
        saver.restore(sess, nets.defaultCheckpoint())

        inim = cv2.imread(nfm.exampleImgPath())
        inim = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)

        batch = np.expand_dims(inim, axis=0)

        #%% Generate TF results
        for _ in range(2):
            sess.run(net_out, feed_dict={image_batch: batch})
        t = time.time()
        result = sess.run(net_out, feed_dict={image_batch: batch})
        print('Took %f seconds' % (time.time() - t))

        #%% Load Matlab results
        mat = scio.loadmat(nfm.exampleStatPath(),
                           struct_as_record=False, squeeze_me=True)
        mat_outs = mat['outs']

        #%% Compare final output
        out_diff = np.abs(mat_outs[-1] - result)
        self.assertLess(np.linalg.norm(out_diff), 0.0053)
        print('Error of final vector is %f' % np.linalg.norm(out_diff))

if __name__ == '__main__':
    unittest.main()
