# -*- coding: utf-8 -*-
"""
@author: Mohammad Asim
"""

import tensorflow as tf

def build_dataset(path, batch=True, batch_size=5, cache=True, ordered=False, shuffle=False, test=False):
    dataset = tf.data.TFRecordDataset(path)
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
        
    dataset = dataset.with_options(ignore_order)
    
    image_feature_description = {
        'flow': tf.io.FixedLenFeature([], tf.string),
        'num': tf.io.FixedLenFeature([], tf.int64),
        'scene': tf.io.FixedLenFeature([], tf.string),
        'current_frame': tf.io.FixedLenFeature([], tf.string),
        'previous_frame': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_image_function(example_proto):
        content = tf.io.parse_single_example(example_proto, image_feature_description)
        current_frame = tf.cast(tf.io.decode_jpeg(content['current_frame']), tf.float32)/255.0
        previous_frame = tf.cast(tf.io.decode_jpeg(content['previous_frame']), tf.float32)/255.0
        if test:
            flow = tf.io.parse_tensor(content['flow'], tf.float64)
            
        else:
            flow = tf.io.parse_tensor(content['flow'], tf.float32)
            
        img_stacked = tf.concat([previous_frame, current_frame], axis=-1)
        return (img_stacked, flow)
    
    dataset = dataset.map(_parse_image_function, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(1024)
        
    if batch:
        dataset = dataset.batch(batch_size, drop_remainder=False)
        
    if cache:
        dataset = dataset.cache()
        
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset