from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

if len(sys.argv) != 2:
    print ('Usage: python train.py <dataset_dir>')

dataset_dir = sys.argv[1]

import os
import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

SHARD_SIZE = 10000
_FILE_PATTERN = 'wanderer_%s_*.tfrecords'
_ITEMS_TO_DESCRIPTIONS = {
    'cmd_vel/linear': '',
    'cmd_vel/angular': '',
    'laser_data/ranges': '',
}
train_dir = '/tmp/tfslim_model/'

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


def load_batch(dataset, batch_size=32, is_training=False):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32, common_queue_min=8)
    linear, angular, ranges = data_provider.get(['cmd_vel/linear', 'cmd_vel/angular', 'laser_data/ranges'])

    # Batch it up.
    linears, angulars, rangeses = tf.train.batch(
          [linear, angular, ranges],
          batch_size=batch_size,
          num_threads=1,
          capacity=2 * batch_size)
    
    return linears, angulars, rangeses

def my_cnn(inputs, is_training):  # is_training is not used...
    with tf.variable_scope('deep_regression', 'deep_regression', [inputs]):
        end_points = {}
        # Set the default weight _regularizer and acvitation for each fully_connected layer.
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(0.01)):

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

def train(dataset_dir):
    global train_dir
    # This might take a few minutes.
    print('Will save model to %s' % train_dir)

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        dataset = get_split('train', dataset_dir)
        linears, angulars, rangeses = load_batch(dataset)

        linears = tf.reshape(linears, [32, 1])
        angulars = tf.reshape(angulars, [32, 1])
        targets = tf.concat([linears, angulars], 1)
        
        # Create the model:
        predictions, end_points = my_cnn(rangeses, is_training=True)

        # Add the loss function to the graph.
        loss = slim.losses.mean_squared_error(predictions, targets)
        
        # The total loss is the uers's loss plus any regularization losses.
        total_loss = slim.losses.get_total_loss()

        # Create some summaries to visualize the training process:
        tf.summary.scalar('losses/Total Loss', total_loss)

        # Specify the optimizer and create the train op:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
        train_op = slim.learning.create_train_op(total_loss, optimizer) 

        # Run the training inside a session.
        final_loss = slim.learning.train(
            train_op,
            logdir=train_dir,
            number_of_steps=5000,
            save_summaries_secs=5,
            log_every_n_steps=500
        )

        print('Finished training. Final batch loss %d' % final_loss)

def evaluate(dataset_dir):
    global train_dir
    # This might take a few minutes.
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.DEBUG)
        
        dataset = get_split('validation', dataset_dir)
        linears, angulars, rangeses = load_batch(dataset)

        linears = tf.reshape(linears, [32, 1])
        angulars = tf.reshape(angulars, [32, 1])
        targets = tf.concat([linears, angulars], 1)
        
        # Create the model:
        predictions, end_points = my_cnn(rangeses, is_training=True)
        
        # Specify metrics to evaluate:
        names_to_value_nodes, names_to_update_nodes = slim.metrics.aggregate_metric_map({
          'Mean Squared Error': slim.metrics.streaming_mean_squared_error(predictions, targets),
          'Mean Absolute Error': slim.metrics.streaming_mean_absolute_error(predictions, targets)
        })

        # Make a session which restores the old graph parameters, and then run eval.
        sv = tf.train.Supervisor(logdir=train_dir)
        with sv.managed_session() as sess:
            metric_values = slim.evaluation.evaluation(
                sess,
                num_evals=100,
                eval_op=names_to_update_nodes.values(),
                final_op=names_to_value_nodes.values())

        names_to_values = dict(zip(names_to_value_nodes.keys(), metric_values))
        for key, value in names_to_values.iteritems():
            print('%s: %f' % (key, value))

def main(_):
    with tf.Graph().as_default(): 
        dataset = get_split('train', '../dataset/')
        data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32, common_queue_min=1)
        linear, angular, ranges = data_provider.get(['cmd_vel/linear', 'cmd_vel/angular', 'laser_data/ranges'])
        
        with tf.Session() as sess:  
            train(dataset_dir)
            evaluate(dataset_dir)



if __name__ == "__main__":
    tf.app.run()