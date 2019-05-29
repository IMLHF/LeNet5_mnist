import tensorflow as tf


class base_config(object):
  learning_rate = 0.0005
  DROP_RATE = 0.3
  MNIST_DIR = 'MNIST_data'
  BATCH_SIZE = 512
  GPU_RAM_ALLOW_GROWTH = True
  EPOCHS = 35
  SAVE_DIR = 'exp'
  CHECK_POINT = 'nnet'
  CONV_PADDING = 'VALID'
  POOL_PADDING = 'VALID'


class C001(base_config):
  CHECK_POINT = 'nnet_LeNet5_C001'
  CONV1_FILTERS = 6
  CONV2_FILTERS = 16
  FC1_units = 120
  FC2_units = 84
  FC3_units = 10
  ACTIVATION = tf.nn.tanh
  OPTIMIZER = tf.train.AdamOptimizer


class C002(base_config):
  CHECK_POINT = 'nnet_LeNet5_C002'
  CONV1_FILTERS = 32
  CONV2_FILTERS = 64
  FC1_units = 1024
  FC2_units = 512
  FC3_units = 10
  ACTIVATION = tf.nn.tanh
  OPTIMIZER = tf.train.AdamOptimizer


class C003(base_config):
  CHECK_POINT = 'nnet_LeNet5_C003'
  CONV1_FILTERS = 6
  CONV2_FILTERS = 16
  FC1_units = 120
  FC2_units = 84
  FC3_units = 10
  DROP_RATE = 0.1
  ACTIVATION = tf.nn.tanh
  OPTIMIZER = tf.train.AdamOptimizer


class C004(base_config):
  CHECK_POINT = 'nnet_LeNet5_C004'
  CONV1_FILTERS = 6
  CONV2_FILTERS = 16
  FC1_units = 120
  FC2_units = 84
  FC3_units = 10
  DROP_RATE = 0.5
  ACTIVATION = tf.nn.tanh
  OPTIMIZER = tf.train.AdamOptimizer

PARAM = C001
