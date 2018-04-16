import tensorflow as tf
import keras as K
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))