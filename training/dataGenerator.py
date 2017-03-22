#!/usr/bin/env python
import rospy
from  geometry_msgs.msg import TwistStamped
from dqn_stage_ros.msg import stage_message
import message_filters

import numpy as np
import math
import sys
import tensorflow as tf
from threading import Thread


if len(sys.argv) != 2:
    print ('Usage: python train.py <dataset_dir>')
    exit(0)

dataset_dir = sys.argv[1]

SHARD_SIZE = 1000

current_array = []
shard_id = 0

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_array_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def writeToTFRecord():
    global current_array, shard_id
    tfrecords_filename = dataset_dir + '/wanderer_train_%05d.tfrecords' % (shard_id)
    shard_id = shard_id + 1

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    original_images = []
    for current_data in current_array:        
        example = tf.train.Example(features=tf.train.Features(feature=current_data))
        writer.write(example.SerializeToString())

    writer.close()
    print ("stored shard %05d" % shard_id)

CONSECUTIVE_CRUISE_RATIO = 10
angularVelocityCounter = 0
counter = 0
def callback(laser_scan, cmd_vel):
    global angularVelocityCounter 
#    if math.fabs(cmd_vel.twist.linear.x) > .3: # cruse
#        angularVelocityCounter += 1
#        if angularVelocityCounter % CONSECUTIVE_CRUISE_RATIO != 0:
#            return
#    else:
#        angularVelocityCounter = 0
#
#        if math.fabs(angularVelocityCounter-1) <=100:
#            print "before if",cmd_vel.twist.linear.x," " ,cmd_vel.twist.angular.z
#            angularVelocityCounter = angularVelocityCounter - 1
#        else:
#            return
#    else: # turning
#        if math.fabs(angularVelocityCounter+1) <=100:
#            angularVelocityCounter = angularVelocityCounter + 1
#        else:
#            return
#    if math.fabs(cmd_vel.twist.angular.z) < 0.01:
#        return;
    global counter, current_array, shard_id
    counter = counter + 1
    print (str(laser_scan.header.stamp.secs) + " " + str(laser_scan.header.stamp.nsecs) + " " + str(counter))

    data = {}
    data['cmd_vel/linear'] = _float_feature(cmd_vel.twist.linear.x)
    data['cmd_vel/angular'] = _float_feature(cmd_vel.twist.angular.z)
    data['laser_data/ranges'] = _float_array_feature(np.asarray(laser_scan.laser.ranges, dtype=np.float32) /  laser_scan.laser.range_max)
    current_array.append(data)
    if len(current_array) > SHARD_SIZE:
        thread = Thread(target = writeToTFRecord, args = ())
        thread.start()
        #writeToTFRecord()
        current_array = []

rospy.init_node('wanderer_train')

laser_sub = message_filters.Subscriber('/input_data', stage_message)
cmd_vel_sub = message_filters.Subscriber('/cmd_vel_stamped', TwistStamped)

ts = message_filters.TimeSynchronizer([laser_sub, cmd_vel_sub], 1000)
ts.registerCallback(callback)
rospy.spin()
