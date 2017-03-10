from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

if len(sys.argv) != 2:
    print ('Usage: python train.py <dataset_dir>')
    exit(0)

dataset_dir = sys.argv[1]

import os
import math
import tensorflow as tf
import numpy as np
from shared import *

slim = tf.contrib.slim

def evaluate(dataset_dir):
    global log_dir
    # This might take a few minutes.
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.DEBUG)
        
        dataset = get_split('validation', dataset_dir)
        linears, angulars, rangeses = load_batch(dataset)

        linears = tf.reshape(linears, [32, 1])
        angulars = tf.reshape(angulars, [32, 1])
        targets = tf.concat([linears, angulars], 1)
        
        # Create the model:
        predictions, end_points = my_cnn(rangeses, is_training=False)
        # Specify metrics to evaluate:
        names_to_value_nodes, names_to_update_nodes = slim.metrics.aggregate_metric_map({
            'Mean Squared Error': slim.metrics.streaming_mean_squared_error(predictions, targets),
            'Mean Absolute Error': slim.metrics.streaming_mean_absolute_error(predictions, targets)
        })

        # Create the summary ops such that they also print out to std output:
        summary_ops = []
        for metric_name, metric_value in names_to_value_nodes.iteritems():
            op = tf.summary.scalar(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)

        num_examples = 10000
        batch_size = 32
        num_batches = math.ceil(SHARD_SIZE / float(BATCH_SIZE))

        # Setup the global step.
        slim.get_or_create_global_step()

        eval_interval_secs = 10 # How often to run the evaluation.
        slim.evaluation.evaluation_loop(
            '',
            checkpoint_dir=log_dir,
            logdir=log_dir,
            num_evals=num_batches,
            eval_op=names_to_update_nodes.values(),
            summary_op=tf.summary.merge(summary_ops),
            eval_interval_secs=eval_interval_secs)

def main(_):
    with tf.Graph().as_default(): 
        with tf.Session() as sess:  
            evaluate(dataset_dir)

if __name__ == "__main__":
    tf.app.run()