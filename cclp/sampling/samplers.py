#!/usr/bin/env python

# Copyright (c) 2018, Konstantinos Kamnitsas
#
# This program is free software; you can redistribute and/or modify
# it under the terms of the Apache License, Version 2.0. See the 
# accompanying LICENSE file or read the terms at:
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import division, print_function

import logging
LOG = logging.getLogger('main')

import numpy as np
import tensorflow as tf

def get_tf_batch_with_n_samples_per_class(all_imgs_per_class, n_per_class_per_batch, seed=None):
   
    n_classes = len(all_imgs_per_class)
    class_labels = np.arange(n_classes)
    batch_imgs = None
    batch_lbls = None
    list_batch_imgs_per_c = []
    list_batch_lbls_per_c = []
    for class_i in range(n_classes):
        all_imgs_c = all_imgs_per_class[class_i]
        all_lbls_c = [class_i] * len(all_imgs_c) # They all have the same label.
        # Sample a few random from this class:
        batch_imgs_c, batch_lbls_c = get_tf_batch(all_imgs_c, all_lbls_c, n_per_class_per_batch, seed) #!!!
        list_batch_imgs_per_c.append(batch_imgs_c)
        list_batch_lbls_per_c.append(batch_lbls_c)        
        
    batch_imgs = tf.concat(list_batch_imgs_per_c, 0)
    batch_lbls = tf.concat(list_batch_lbls_per_c, 0)
    return batch_imgs, batch_lbls

def get_tf_batch_from_samples_and_labels(samples=None,labels=None,n_classes=2,n_per_class_per_batch=64,seed=None):
    batch_imgs = None
    batch_lbls = None
    list_batch_imgs_per_c = []
    list_batch_lbls_per_c = []
    for class_i in range(n_classes):
        all_imgs_c = samples[labels==class_i]
        all_lbls_c = [class_i] * len(all_imgs_c) # They all have the same label.
        # Sample a few random from this class:
        batch_imgs_c, batch_lbls_c = get_tf_batch(all_imgs_c, all_lbls_c, n_per_class_per_batch, seed) #!!!
        list_batch_imgs_per_c.append(batch_imgs_c)
        list_batch_lbls_per_c.append(batch_lbls_c)        
        
    batch_imgs = tf.concat(list_batch_imgs_per_c, 0)
    batch_lbls = tf.concat(list_batch_lbls_per_c, 0)
    return batch_imgs, batch_lbls

def get_tf_batch(all_imgs, all_lbls, batch_size, seed=None):
    # all_imgs: np array [num_samples, x, y, chans], all images to make batches from.
    # all_lbls: np array [num_samples],  integer saying the class. Can be None, eg for unlabelled.
    # returns: two tf tensors: a batch of imgs and corresponding labels.
    batch_imgs = None; batch_lbls = None
    if all_lbls is not None :
        # !!! Deprecated function, should be replaced: https://www.tensorflow.org/api_docs/python/tf/train/slice_input_producer
        imgs_shuffled, lbls_shuffled = tf.compat.v1.train.slice_input_producer([all_imgs, all_lbls], shuffle=True, seed=seed) # Shuffling.
        batch_imgs, batch_lbls = tf.compat.v1.train.batch([imgs_shuffled, lbls_shuffled], batch_size)
    else:
        imgs_shuffled = tf.compat.v1.train.slice_input_producer([all_imgs], shuffle=True, seed=seed)
        batch_imgs = tf.compat.v1.train.batch(imgs_shuffled, batch_size=batch_size)
    return batch_imgs, batch_lbls














