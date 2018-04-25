# This takes a lot of time to run, so not written as unit test.

import netvlad_tf.nets as nets

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import scipy.spatial.distance as scid
import scipy.signal as scisig
import tensorflow as tf
import time
import unittest

from netvlad_tf.image_descriptor import ImageDescriptor
import netvlad_tf.net_from_mat as nfm
import netvlad_tf.precision_recall as pr

#%% Init
use_dim = 128

#%% Get KITTI feats
tf.reset_default_graph()
imd = ImageDescriptor(is_grayscale=True)
feats = imd.describeAllJpegsInPath('kitti/00/image_0', 4, verbose=True)
#%%
use_feats = np.array(feats)[:, :use_dim]

#%% Suppressing a certain radius around the diagonal to prevent self-matches
# (not real loop closures).
suppression_diameter = 501
diag_suppression = scisig.convolve2d(
        np.eye(use_feats.shape[0]), np.ones((1, suppression_diameter)), 'same')

#%% NetVLAD matching, reject along diagonal (query ~ match):
sq_dists = scid.squareform(scid.pdist(use_feats, 'sqeuclidean'))
sq_dists[diag_suppression > 0] = np.inf
plt.imshow(sq_dists)
plt.title('Confusion matrix NetVLAD')
plt.colorbar()
plt.show()

nv_dists2 = sq_dists.min(axis=0)
nv_indices = sq_dists.argmin(axis=0)

#%% Ground truth matching:
kitti_poses = np.loadtxt('kitti/poses/00.txt')
kitti_positions = kitti_poses[:, [3, 7, 11]]

sq_dists = scid.squareform(scid.pdist(kitti_positions, 'sqeuclidean'))
sq_dists[diag_suppression > 0] = np.inf
plt.imshow(np.sqrt(sq_dists))
plt.title('Confusion matrix GPS')
plt.colorbar()
plt.show()

gt_dists2 = sq_dists.min(axis=0)
gt_indices = sq_dists.argmin(axis=0)

#%% Evaluate and compare to matlab results
mat = scio.loadmat('matlab/kitti_pr.mat',
                   struct_as_record=False, squeeze_me=True)

gt_radius = 5
precision, recall, auc = pr.evaluate(
        kitti_positions, nv_indices, nv_dists2, gt_dists2, gt_radius)
plt.plot(mat['recall'], mat['precision'])
plt.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend(['Matlab', 'TensorFlow'])
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title('PR curve, AUC = %f' % auc)
plt.show()
