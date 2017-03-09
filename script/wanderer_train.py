#!/usr/bin/env python
import rospy
from  geometry_msgs.msg import TwistStamped
from dqn_stage_ros.msg import stage_message
import message_filters

import numpy as np
import tensorflow as tf

SHARD_SIZE = 10000

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
    tfrecords_filename = 'wanderer_train_%05d.tfrecords' % (shard_id)
    shard_id = shard_id + 1

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    original_images = []
    for current_data in current_array:        
        example = tf.train.Example(features=tf.train.Features(feature=current_data))
        writer.write(example.SerializeToString())

    writer.close()
    print ("stored shard %05d" % shard_id)

counter = 0
def callback(laser_scan, cmd_vel):
    global counter
    counter = counter + 1
    print (str(laser_scan.header.stamp.secs) + " " + str(laser_scan.header.stamp.nsecs) + " " + str(counter))

    global current_array, shard_id
    data = {}
    data['cmd_vel/linear'] = _float_feature(cmd_vel.twist.linear.x)
    data['cmd_vel/angular'] = _float_feature(cmd_vel.twist.angular.z)
    data['laser_data/ranges'] = _float_array_feature(np.asarray(laser_scan.laser.ranges, dtype=np.float32) /  laser_scan.laser.range_max)
    current_array.append(data)
    if len(current_array) > SHARD_SIZE:
        writeToTFRecord()
        current_array = []

rospy.init_node('wanderer_train')

laser_sub = message_filters.Subscriber('/input_data', stage_message)
cmd_vel_sub = message_filters.Subscriber('/cmd_vel_stamped', TwistStamped)

ts = message_filters.TimeSynchronizer([laser_sub, cmd_vel_sub], 1000)
ts.registerCallback(callback)
rospy.spin()