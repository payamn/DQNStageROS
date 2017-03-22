#!/usr/bin/env python
import rospy
from  geometry_msgs.msg import Twist
from dqn_stage_ros.msg import stage_message

import numpy as np
import tensorflow as tf
from shared import *

import sys

if len(sys.argv) != 2:
    print ('Usage: python train.py <dataset_dir>')
    exit(0)

dataset_dir = sys.argv[1]

class CmdVelPublisher:
    def __init__(self):
        self.sess = tf.Session()
        self.input_tensor = tf.placeholder(tf.float32, shape=(1, 360))
        self.predictions, self.end_points = my_cnn(self.input_tensor, is_training=False)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, dataset_dir)

        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=100, latch=True)

    def laserCallback(self, laser_scan):
        ranges = [np.asarray(laser_scan.laser.ranges, dtype=np.float32) /  laser_scan.laser.range_max]
        rangeses = np.asarray(ranges, dtype=np.float32)

        predictions, end_points = self.sess.run([self.predictions, self.end_points], feed_dict={self.input_tensor: rangeses})
        linear = predictions[0, 0]
        #linear = 0.005
        angular = predictions[0, 1]
#        linear = 1
        self.publish(linear, angular)

    def publish(self, linear, angular):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        
        self.cmd_vel_pub.publish(msg)

rospy.init_node('wanderer_test')

cmd_vel_publisher = CmdVelPublisher()
rospy.Subscriber("/input_data", stage_message, cmd_vel_publisher.laserCallback)

cmd_vel_publisher.publish(0, 0)
rospy.spin()










# class CmdVelPublisher:
#     def __init__(self):
#         self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1, latch=True)
#         self.sess = tf.Session()
#         self.saver = tf.train.import_meta_graph('my-model.meta')
#         self.new_saver.restore(self.sess, tf.train.latest_checkpoint('./'))
#         all_vars = tf.get_collection('vars')
# for v in all_vars:
#     v_ = sess.run(v)
#     print(v_)

#     def laserCallback(self, laser_scan):
#             ranges = [np.asarray(laser_scan.laser.ranges, dtype=np.float32) /  laser_scan.laser.range_max]
#             rangeses = np.asarray(ranges, dtype=np.float32)

#             inputs = tf.constant(rangeses)
#             inputs.set_shape([1, 360])

#             predictions, end_points = my_cnn(inputs, is_training=False)
  
#             with sv.managed_session() as sess:
#                 inputs, predictions = sess.run([inputs, predictions])
#                 linear = predictions[0, 0]
#                 angular = predictions[0, 1]
#                 self.publish(linear, angular)

#     def publish(self, linear, angular):
#         msg = Twist()
#         msg.linear.x = linear
#         msg.angular.z = angular
        
#         self.cmd_vel_pub.publish(msg)

# rospy.init_node('wanderer_test')

# cmd_vel_publisher = CmdVelPublisher()
# rospy.Subscriber("/input_data", stage_message, cmd_vel_publisher.laserCallback)

# cmd_vel_publisher.publish(0, 0)
# rospy.spin()
