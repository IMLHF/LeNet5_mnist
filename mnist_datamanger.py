import pickle
import os
import numpy as np
from skimage import io as image_io
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import FLAGS


def read_trainset_to_ram():
  mnist = input_data.read_data_sets(FLAGS.PARAM.MNIST_DIR, one_hot=True)
  return np.reshape(mnist.train.images,[-1,28,28,1]), np.array(mnist.train.labels,np.float32)

def read_testset_to_ram():
  mnist = input_data.read_data_sets(FLAGS.PARAM.MNIST_DIR, one_hot=True)
  return np.reshape(mnist.test.images,[-1,28,28,1]), np.array(mnist.test.labels,np.float32)

def get_batch_use_tfdata(features, labels):
  features_placeholder = tf.placeholder(features.dtype, features.shape)
  labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

  dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
  dataset = dataset.batch(FLAGS.PARAM.BATCH_SIZE)
  iterator = dataset.make_initializable_iterator()
  inputs_batch, labels_batch = iterator.get_next()
  return features_placeholder, labels_placeholder, inputs_batch, labels_batch, iterator

if __name__ == '__main__':
  inputs_np, labels_np = read_trainset_to_ram()
  # print(np.shape(labels_np))
  # features, labels = read_testset_to_ram()
  x_p, y_p, inputs, labels, iterator = get_batch_use_tfdata(inputs_np, labels_np)
  sess = tf.Session()
  _, inputs_, labels_ = sess.run([iterator.initializer, inputs, labels],
                                 feed_dict={x_p: inputs_np,
                                            y_p: labels_np})
  print(labels_[40])
  image_io.imshow(inputs_[40])
  plt.show()
