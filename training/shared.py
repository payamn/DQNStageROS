from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import os
import math
import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

NUM_EPOCHS = 100000
SHARD_SIZE = 10000
BATCH_SIZE = 32
_FILE_PATTERN = 'wanderer_%s_*.tfrecords'
_ITEMS_TO_DESCRIPTIONS = {
    'cmd_vel/linear': '',
    'cmd_vel/angular': '',
    'laser_data/ranges': '',
}
log_dir = '/tmp/tfslim_model_log/'

def get_split(split_name, dataset_dir):
    file_pattern = os.path.join(dataset_dir, _FILE_PATTERN % split_name)
    reader = tf.TFRecordReader

    keys_to_features = {
      'cmd_vel/linear': tf.FixedLenFeature([], tf.float32),
      'cmd_vel/angular': tf.FixedLenFeature([], tf.float32),
      'laser_data/ranges': tf.FixedLenFeature([360], tf.float32)
    }

    items_to_handlers = {
        'cmd_vel/linear': slim.tfexample_decoder.Tensor('cmd_vel/linear'),
        'cmd_vel/angular': slim.tfexample_decoder.Tensor('cmd_vel/angular'),
        'laser_data/ranges': slim.tfexample_decoder.Tensor('laser_data/ranges'),
    }
    
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=SHARD_SIZE,
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS
    )

def load_batch(dataset, is_training=False):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32, common_queue_min=8)
    linear, angular, ranges = data_provider.get(['cmd_vel/linear', 'cmd_vel/angular', 'laser_data/ranges'])

    # Batch it up.
    linears, angulars, rangeses = tf.train.batch(
          [linear, angular, ranges],
          batch_size=BATCH_SIZE,
          num_threads=1,
          capacity=2 * BATCH_SIZE)
    
    return linears, angulars, rangeses

def conv1d(input_, output_channels, 
    filter_width = 1, stride = 1, stddev=0.02,
    name = 'conv1d'):
    with tf.variable_scope(name):
        input_shape = input_.get_shape()
        input_channels = input_shape[-1]
        filter_ = tf.get_variable('w', [filter_width, input_channels, output_channels],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv1d(input_, filter_, stride = stride, padding = 'SAME')
        return conv

def my_cnn(inputs, is_training):  # is_training is not used...
    with tf.variable_scope('deep_regression', 'deep_regression', [inputs]):
        end_points = {}
        # Set the default weight _regularizer and acvitation for each fully_connected layer.
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(0.01)):

            # input 360
            # net = tf.nn.conv1d(inputs, [], stride=2, padding="VALID")

            # Creates a fully connected layer from the inputs with 32 hidden units. 

            # net = conv1d(inputs, 32) # 360x32
            # end_points['conv1d_1'] = net

            # net = tf.nn.max_pool(net, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME') # 180x16
            # end_points['maxpool_1'] = net

            # net = tf.nn.conv1d(net, [3, 16, 32], stride=1, padding="SAME") # 180x32
            # end_points['conv1d_2'] = net

            # net = tf.nn.max_pool(net, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME') # 90x32
            # end_points['maxpool_2'] = net

            # net = slim.flatten(net) # 90*32 = 2880

            # net = tf.nn.conv1d(inputs, [], stride=2, padding="VALID")

            # Creates a fully connected layer from the inputs with 32 hidden units.
            net = slim.fully_connected(inputs, 32, scope='fc1')
            end_points['fc1'] = net

            # Adds a dropout layer to prevent over-fitting.
            net = slim.dropout(net, 0.8, is_training=is_training)

            # Adds another fully connected layer with 16 hidden units.
            net = slim.fully_connected(net, 16, scope='fc2')
            end_points['fc2'] = net

            # Creates a fully-connected layer with a single hidden unit. Note that the
            # layer is made linear by setting activation_fn=None.
            predictions = slim.fully_connected(net, 2, activation_fn=None, scope='prediction')
            end_points['out'] = predictions

            return predictions, end_points