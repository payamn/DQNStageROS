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

def train(dataset_dir):
    global log_dir
    # This might take a few minutes.
    print('Will save model to %s' % log_dir)

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        dataset = get_split('train', dataset_dir)
        linears, angulars, rangeses = load_batch(dataset)

        linears = tf.reshape(linears, [256, 1])
        angulars = tf.reshape(angulars, [256, 1])
        targets = tf.concat([linears, angulars], 1)
        
        # Create the model:
        predictions, end_points = my_cnn(rangeses, is_training=True)

        # Add the loss function to the graph.
        loss = slim.losses.mean_squared_error(predictions, targets)
        
        # The total loss is the uers's loss plus any regularization losses.
        total_loss = slim.losses.get_total_loss()

        # Create some summaries to visualize the training process:
        tf.summary.scalar('losses/total_loss', total_loss)

        # Specify the optimizer and create the train op:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
        train_op = slim.learning.create_train_op(total_loss, optimizer) 

        # Run the training inside a session.
        final_loss = slim.learning.train(
            train_op,
            logdir=log_dir,
            number_of_steps=NUM_EPOCHS,
            save_summaries_secs=10,
            log_every_n_steps=500,
            save_interval_secs=60
        )

        print('Finished training. Final batch loss %d' % final_loss)

def main(_):
    with tf.Graph().as_default(): 
        with tf.Session() as sess:
            train(dataset_dir)

if __name__ == "__main__":
    tf.app.run()
