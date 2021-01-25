import tensorflow as tf

from netvlad_tf.net_from_mat import netFromMat
from netvlad_tf.nets import defaultCheckpoint

tf.compat.v1.reset_default_graph()
layers = netFromMat()
saver = tf.compat.v1.train.Saver()

sess = tf.compat.v1.Session()
tf.compat.v1.global_variables_initializer().run(session=sess)
saver.save(sess, defaultCheckpoint())