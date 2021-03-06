# author: lihongfeng 2019-5-20
# github:
import tensorflow as tf
import numpy as np
import FLAGS


class LeNet5_CLASSIFY(object):
  infer = 'infer'
  train = 'train'
  validation = 'validation'

  def __init__(self, input_batch, onehot_label_batch, behavior):
    '''
    input_batch: [batch_size,28,28]
    label_batch: [batch_size,10]
    '''
    self._input_batch = input_batch
    self._onehot_label_batch = onehot_label_batch
    # with tf.variable_scope('model'):
    weights = {
        'w_conv1': tf.get_variable('w_conv1', [5, 5, 1, FLAGS.PARAM.CONV1_FILTERS],
                                   initializer=tf.random_normal_initializer(stddev=0.01)),
        'w_conv2': tf.get_variable('w_conv2', [5, 5, FLAGS.PARAM.CONV1_FILTERS, FLAGS.PARAM.CONV2_FILTERS],
                                   initializer=tf.random_normal_initializer(stddev=0.01)),
        'w_fc1': tf.get_variable('w_fc1', [4 * 4 * FLAGS.PARAM.CONV2_FILTERS, FLAGS.PARAM.FC1_units],
                                 initializer=tf.random_normal_initializer(stddev=0.01)),
        'w_fc2': tf.get_variable('w_fc2', [FLAGS.PARAM.FC1_units, FLAGS.PARAM.FC2_units],
                                 initializer=tf.random_normal_initializer(stddev=0.01)),
        'w_fc3': tf.get_variable('w_fc3', [FLAGS.PARAM.FC2_units, FLAGS.PARAM.FC3_units],
                                 initializer=tf.random_normal_initializer(stddev=0.01)),
    }
    biases = {
        'b_conv1': tf.get_variable('b_conv1', [FLAGS.PARAM.CONV1_FILTERS], initializer=tf.random_normal_initializer(stddev=0.01)),
        'b_conv2': tf.get_variable('b_conv2', [FLAGS.PARAM.CONV2_FILTERS], initializer=tf.random_normal_initializer(stddev=0.01)),
        'b_fc1': tf.get_variable('b_fc1', [FLAGS.PARAM.FC1_units], initializer=tf.random_normal_initializer(stddev=0.01)),
        'b_fc2': tf.get_variable('b_fc2', [FLAGS.PARAM.FC2_units], initializer=tf.random_normal_initializer(stddev=0.01)),
        'b_fc3': tf.get_variable('b_fc3', [FLAGS.PARAM.FC3_units], initializer=tf.random_normal_initializer(stddev=0.01)),
    }

    out_conv1 = FLAGS.PARAM.ACTIVATION(
        tf.nn.conv2d(self._input_batch,
                     weights['w_conv1'],
                     [1, 1, 1, 1],
                     padding=FLAGS.PARAM.CONV_PADDING
                     ) + biases['b_conv1'])
    out_mp1 = tf.nn.max_pool(out_conv1, ksize=[1, 2, 2, 1], strides=[
                             1, 2, 2, 1], padding=FLAGS.PARAM.POOL_PADDING)
    out_conv2 = FLAGS.PARAM.ACTIVATION(
        tf.nn.conv2d(out_mp1,
                     weights['w_conv2'],
                     [1, 1, 1, 1],
                     padding=FLAGS.PARAM.CONV_PADDING
                     ) + biases['b_conv2'])
    out_mp2 = tf.nn.max_pool(out_conv2, ksize=[1, 2, 2, 1], strides=[
                             1, 2, 2, 1], padding=FLAGS.PARAM.POOL_PADDING)
    out_conv_flatten = tf.reshape(out_mp2, [-1, 4*4*FLAGS.PARAM.CONV2_FILTERS])
    out_fc1 = FLAGS.PARAM.ACTIVATION(tf.matmul(out_conv_flatten, weights['w_fc1'])+biases['b_fc1'])
    if behavior == self.train:
      out_drop_fc1 = tf.nn.dropout(out_fc1, keep_prob=1.0-FLAGS.PARAM.DROP_RATE)
    else:
      out_drop_fc1 = out_fc1
    out_fc2 = FLAGS.PARAM.ACTIVATION(tf.matmul(out_drop_fc1, weights['w_fc2'] + biases['b_fc2']))
    out_fc3 = tf.matmul(out_fc2, weights['w_fc3'] + biases['b_fc3'])
    self._logits = out_fc3
    self._out_softmax = tf.nn.softmax(self._logits)
    self._accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(self._out_softmax, 1), tf.argmax(self._onehot_label_batch, 1)), tf.float32))
    self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=30)
    if behavior == self.infer:
      return
    self._cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=self._logits, labels=self._onehot_label_batch))
    self._train_op = FLAGS.PARAM.OPTIMIZER(
        learning_rate=FLAGS.PARAM.learning_rate).minimize(self._cross_entropy_loss)

  @property
  def cross_entropy_loss(self):
    return self._cross_entropy_loss

  @property
  def train_op(self):
    return self._train_op

  @property
  def out_softmax(self):
    return self._out_softmax

  @property
  def accuracy(self):
    return self._accuracy
