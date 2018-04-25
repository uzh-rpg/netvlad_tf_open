import netvlad_tf.net_from_mat as nfm

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import tensorflow as tf
import time
import unittest

class TestNetFromMat(unittest.TestCase):
    def testNetFromMat(self):
        ''' Need example_stats.mat in matlab folder, which can be generated
        with get_example_stats.m. '''

        tf.reset_default_graph()
        tf_layers = nfm.netFromMat()

        #%% Test if it's the same
        sess = tf.Session()
        tf.global_variables_initializer().run(session=sess)

        inim = cv2.imread(nfm.exampleImgPath())
        inim = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)

        batch = np.expand_dims(inim, axis=0)

        #%% Generate TF results
        for _ in range(2):
            sess.run(tf_layers, feed_dict={tf_layers[0]: batch})
        t = time.time()
        layer_outs = sess.run(tf_layers, feed_dict={tf_layers[0]: batch})
        print('Took %f seconds' % (time.time() - t))

        #%% Load Matlab results
        mat = scio.loadmat(nfm.exampleStatPath(),
                           struct_as_record=False, squeeze_me=True)
        mat_outs = mat['outs']

        #%% Compare layer by layer

        for i in range(1, len(layer_outs)):
            # We have an additional layer due to the initial normalization.
            out_diff = np.abs(mat_outs[i - 1] - layer_outs[i])
            maxod = np.max(out_diff)
            print('Layer %s:\t\tMax error is %f' %
                  (tf_layers[i].name[:18].ljust(18), maxod))
            self.assertLess(maxod, 0.018)

        self.assertLess(maxod, 0.00031)
        self.assertLess(np.linalg.norm(out_diff), 0.0053)
        print('Error of final vector is %f' % np.linalg.norm(out_diff))

if __name__ == '__main__':
    unittest.main()
