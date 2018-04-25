import tensorflow as tf

from netvlad_tf.net_from_mat import netFromMat
from netvlad_tf.nets import defaultCheckpoint

tf.reset_default_graph()
layers = netFromMat()
saver = tf.train.Saver()

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
saver.save(sess, defaultCheckpoint())