#!/usr/bin/env python

# Copyright (c) 2018, Konstantinos Kamnitsas
#
# This program is free software; you can redistribute and/or modify
# it under the terms of the Apache License, Version 2.0. See the 
# accompanying LICENSE file or read the terms at:
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import absolute_import, division, print_function

import logging
LOG = logging.getLogger('main')

import tensorflow as tf
from cclp.neuralnet.models.activation_funcs import get_act_func



# Architectures to use for a feature extractor z.
# iot_fc hyperparameters
# h_dim1 = 60
# h_dim2 = 55
# h_dim3 = 50
h_dim1 = 260
h_dim2 = 240
h_dim3 = 200
h_dim4 = 170
h_dim5 = 150
h_dim6 = 128


def iot_fc( inputs,
            is_training,
            emb_z_size=30,
            emb_z_act="elu",
            l2_weight=1e-5,
            batch_norm_decay=0.99,
            seed=None):
    # inputs: [Batchsize x features]
    fwd = inputs
    tf.summary.scalar('min_int_net_in', tf.reduce_min(fwd), collections=[tf.GraphKeys.SUMMARIES] if is_training else ["eval_summaries"])
    tf.summary.scalar('max_int_net_in', tf.reduce_max(fwd), collections=[tf.GraphKeys.SUMMARIES] if is_training else ["eval_summaries"])
    
    # weights_initializer = tf.truncated_normal_initializer(stddev=0.01, seed=seed)
    # weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=seed, dtype=tf.float32)
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True, seed=seed, dtype=tf.float32)#An initializer that generates tensors with unit variance.
    weights_regularizer = tf.contrib.layers.l2_regularizer
    
    activation_fn = get_act_func(emb_z_act)
    
    USE_BN = batch_norm_decay != -1 # True if not -1. BN puts in tf.GraphKeys.UPDATE_OPS, needs to be put in train_op.
    BN_TRAINABLE = True
    
    # max_pool = tf.contrib.layers.max_pool2d
    # avg_pool = tf.contrib.layers.avg_pool2d
    dropout_f = tf.layers.dropout # tf.layers.dropout gets drop_rate. Others get keep_rate.
    
    # if is_training:
    #     fwd = fwd + tf.random_normal( tf.shape(fwd), mean=0.0, stddev=0.3, seed=seed, dtype=tf.float32 ) #? iGan, badGan, triGan use this and/or dropout
    # layer 1
    fwd = tf.contrib.layers.fully_connected(fwd,h_dim1, activation_fn=None,normalizer_fn=None,
                                            weights_initializer=weights_initializer,
                                            weights_regularizer=weights_regularizer(l2_weight),
                                            trainable=is_training,
                                            scope='fc_1')   
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn1_1" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    # layer 2
    fwd = tf.contrib.layers.fully_connected(fwd,h_dim2, activation_fn=None,normalizer_fn=None,
                                            weights_initializer=weights_initializer,
                                            weights_regularizer=weights_regularizer(l2_weight),
                                            trainable=is_training,
                                            scope='fc_2')   
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn1_2" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    
    # layer 3
    fwd = tf.contrib.layers.fully_connected(fwd,h_dim3, activation_fn=None,normalizer_fn=None,
                                            weights_initializer=weights_initializer,
                                            weights_regularizer=weights_regularizer(l2_weight),
                                            trainable=is_training,
                                            scope='fc_3')   
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn1_3" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    
    # layer 4
    fwd = tf.contrib.layers.fully_connected(fwd,h_dim4, activation_fn=None,normalizer_fn=None,
                                            weights_initializer=weights_initializer,
                                            weights_regularizer=weights_regularizer(l2_weight),
                                            trainable=is_training,
                                            scope='fc_4')   
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn1_4" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    
    # layer 5
    fwd = tf.contrib.layers.fully_connected(fwd,h_dim5, activation_fn=None,normalizer_fn=None,
                                            weights_initializer=weights_initializer,
                                            weights_regularizer=weights_regularizer(l2_weight),
                                            trainable=is_training,
                                            scope='fc_5')   
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn1_5" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = dropout_f( fwd, rate=0.1, training=is_training, noise_shape=(fwd.get_shape().as_list()[0], fwd.get_shape().as_list()[-1]), seed=seed ) # ala badGan. Doesnt do much, but to show there is no problem training with dropout here.
    # layer 6
    fwd = tf.contrib.layers.fully_connected(fwd,h_dim6, activation_fn=None,normalizer_fn=None,
                                            weights_initializer=weights_initializer,
                                            weights_regularizer=weights_regularizer(l2_weight),
                                            trainable=is_training,
                                            scope='fc_6')   
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn1_6" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    # output
    fwd = tf.layers.flatten(fwd)
    emb_z = fwd
    return emb_z

# need to code
def iot_cnn( inputs,
            is_training,
            emb_z_size=128,
            emb_z_act="elu",
            l2_weight=1e-5,
            batch_norm_decay=0.99,
            seed=None):
    # inputs: [Batchsize*features*1*1]
    # Architecture ala Triple GAN.
    # firstly, we should ascending dimension
    fwd = tf.expand_dims(inputs, -1)
    fwd = tf.expand_dims(fwd, -1)
    tf.summary.scalar('min_int_net_in', tf.reduce_min(fwd), collections=[tf.GraphKeys.SUMMARIES] if is_training else ["eval_summaries"])
    tf.summary.scalar('max_int_net_in', tf.reduce_max(fwd), collections=[tf.GraphKeys.SUMMARIES] if is_training else ["eval_summaries"])
    
    # weights_initializer = tf.truncated_normal_initializer(stddev=0.01, seed=seed)
    # weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=seed, dtype=tf.float32)
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True, seed=seed, dtype=tf.float32)#An initializer that generates tensors with unit variance.
    weights_regularizer = tf.contrib.layers.l2_regularizer
    
    activation_fn = get_act_func(emb_z_act)
    
    USE_BN = batch_norm_decay != -1 # True if not -1. BN puts in tf.GraphKeys.UPDATE_OPS, needs to be put in train_op.
    BN_TRAINABLE = True
    
    max_pool = tf.contrib.layers.max_pool2d
    avg_pool = tf.contrib.layers.avg_pool2d
    dropout_f = tf.layers.dropout # tf.layers.dropout gets drop_rate. Others get keep_rate.
    
    # if is_training:
    #     fwd = fwd + tf.random_normal( tf.shape(fwd), mean=0.0, stddev=0.3, seed=seed, dtype=tf.float32 ) #? iGan, badGan, triGan use this and/or dropout
    # fwd = tf.nn.conv1d(fwd,)    
    fwd = tf.contrib.layers.conv2d(fwd, 32, [3, 1], stride=2, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                                   weights_regularizer=weights_regularizer(l2_weight), scope='c1_1')
    print('c1_1 shape:',fwd.shape)
    
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn1_1" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = tf.contrib.layers.conv2d(fwd, 45, [3, 1], stride=2, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c1_2')
    print('c1_2 shape:',fwd.shape)
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn1_2" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = max_pool(fwd, [2, 1], scope='p1')  # 14
    print('p1 shape:',fwd.shape)
    
    fwd = tf.contrib.layers.conv2d(fwd, 64, [3, 1], stride=2, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c2_1')
    print('c2_1 shape:',fwd.shape)
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn2_1" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = tf.contrib.layers.conv2d(fwd, 80, [3, 1], stride=2, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c2_2')
    print('c2_2 shape:',fwd.shape)
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn_2_2" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = max_pool(fwd, [2, 1], scope='p2')  # 7
    print('p2 shape:',fwd.shape)
    fwd = tf.contrib.layers.conv2d(fwd, 100, [2, 1], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c3_1')
    print('c3_1 shape:',fwd.shape)
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn3_1" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    # Dropout should NOT be used right before the graph construction (emb_z). It makes the affinity matrix too sparse. Rather, at least a conv before, such as here for example.
    fwd = dropout_f( fwd, rate=0.1, training=is_training, noise_shape=(fwd.get_shape().as_list()[0], 1, 1, fwd.get_shape().as_list()[-1]), seed=seed ) # ala badGan. Doesnt do much, but to show there is no problem training with dropout here.
    fwd = tf.contrib.layers.conv2d(fwd, emb_z_size, [2, 1], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c3_2')
    print('c3_2 shape:',fwd.shape)
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn3_2" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    
    fwd = avg_pool(fwd, fwd.get_shape().as_list()[1:3], scope='p3')
    print('p3 shape:',fwd.shape)
    fwd = tf.layers.flatten(fwd)
    
    emb_z = fwd
    return emb_z


#----backup-------
def iot_cnn0( inputs,
            is_training,
            emb_z_size=80,
            emb_z_act="elu",
            l2_weight=1e-5,
            batch_norm_decay=0.99,
            seed=None):
    # inputs: [Batchsize*features*1*1]
    # Architecture ala Triple GAN.
    # firstly, we should ascending dimension
    fwd = tf.expand_dims(inputs, -1)
    fwd = tf.expand_dims(fwd, -1)
    tf.summary.scalar('min_int_net_in', tf.reduce_min(fwd), collections=[tf.GraphKeys.SUMMARIES] if is_training else ["eval_summaries"])
    tf.summary.scalar('max_int_net_in', tf.reduce_max(fwd), collections=[tf.GraphKeys.SUMMARIES] if is_training else ["eval_summaries"])
    
    # weights_initializer = tf.truncated_normal_initializer(stddev=0.01, seed=seed)
    # weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=seed, dtype=tf.float32)
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True, seed=seed, dtype=tf.float32)#An initializer that generates tensors with unit variance.
    weights_regularizer = tf.contrib.layers.l2_regularizer
    
    activation_fn = get_act_func(emb_z_act)
    
    USE_BN = batch_norm_decay != -1 # True if not -1. BN puts in tf.GraphKeys.UPDATE_OPS, needs to be put in train_op.
    BN_TRAINABLE = True
    
    max_pool = tf.contrib.layers.max_pool2d
    avg_pool = tf.contrib.layers.avg_pool2d
    dropout_f = tf.layers.dropout # tf.layers.dropout gets drop_rate. Others get keep_rate.
    
    # if is_training:
    #     fwd = fwd + tf.random_normal( tf.shape(fwd), mean=0.0, stddev=0.3, seed=seed, dtype=tf.float32 ) #? iGan, badGan, triGan use this and/or dropout
    # fwd = tf.nn.conv1d(fwd,)    
    fwd = tf.contrib.layers.conv2d(fwd, 32, [3, 1], stride=2, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                                   weights_regularizer=weights_regularizer(l2_weight), scope='c1_1')
    print('c1_1 shape:',fwd.shape)
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn1_1" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = tf.contrib.layers.conv2d(fwd, 32, [3, 1], stride=2, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c1_2')
    print('c1_2 shape:',fwd.shape)
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn1_2" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = max_pool(fwd, [2, 1], scope='p1')  # 14
    print('p1 shape:',fwd.shape)
    fwd = tf.contrib.layers.conv2d(fwd, 64, [3, 1], stride=2, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c2_1')
    print('c2_1 shape:',fwd.shape)
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn2_1" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = tf.contrib.layers.conv2d(fwd, 64, [3, 1], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c2_2')
    print('c2_2 shape:',fwd.shape)
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn_2_2" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = max_pool(fwd, [2, 1], scope='p2')  # 7
    print('p2 shape:',fwd.shape)
    fwd = tf.contrib.layers.conv2d(fwd, 70, [3, 1], stride=2, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c3_1')
    print('c3_1 shape:',fwd.shape)
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn3_1" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    # Dropout should NOT be used right before the graph construction (emb_z). It makes the affinity matrix too sparse. Rather, at least a conv before, such as here for example.
    fwd = dropout_f( fwd, rate=0.1, training=is_training, noise_shape=(fwd.get_shape().as_list()[0], 1, 1, fwd.get_shape().as_list()[-1]), seed=seed ) # ala badGan. Doesnt do much, but to show there is no problem training with dropout here.
    fwd = tf.contrib.layers.conv2d(fwd, emb_z_size, [3, 1], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                               weights_regularizer=weights_regularizer(l2_weight), scope='c3_2')
    print('c3_2 shape:',fwd.shape)
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn3_2" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    
    fwd = avg_pool(fwd, fwd.get_shape().as_list()[1:3], scope='p3')
    print('p3 shape:',fwd.shape)
    fwd = tf.layers.flatten(fwd)
    print('flatten shape:',fwd.shape)
    
    emb_z = fwd
    return emb_z

#------------------
#iot_cnn2
def iot_cnn2( inputs,
            is_training,
            emb_z_size=60,
            emb_z_act="elu",
            l2_weight=1e-5,
            batch_norm_decay=0.99,
            seed=None):
    # inputs: [Batchsize*features*1*1]
    # Architecture ala Triple GAN.
    # firstly, we should ascending dimension
    fwd = tf.expand_dims(inputs, -1)
    fwd = tf.expand_dims(fwd, -1)
    tf.summary.scalar('min_int_net_in', tf.reduce_min(fwd), collections=[tf.GraphKeys.SUMMARIES] if is_training else ["eval_summaries"])
    tf.summary.scalar('max_int_net_in', tf.reduce_max(fwd), collections=[tf.GraphKeys.SUMMARIES] if is_training else ["eval_summaries"])
    
    # weights_initializer = tf.truncated_normal_initializer(stddev=0.01, seed=seed)
    # weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=seed, dtype=tf.float32)
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True, seed=seed, dtype=tf.float32)#An initializer that generates tensors with unit variance.
    weights_regularizer = tf.contrib.layers.l2_regularizer
    
    activation_fn = get_act_func(emb_z_act)
    
    USE_BN = batch_norm_decay != -1 # True if not -1. BN puts in tf.GraphKeys.UPDATE_OPS, needs to be put in train_op.
    BN_TRAINABLE = True
    
    max_pool = tf.contrib.layers.max_pool2d
    avg_pool = tf.contrib.layers.avg_pool2d
    dropout_f = tf.layers.dropout # tf.layers.dropout gets drop_rate. Others get keep_rate.
    with tf.name_scope('cnn2'):
        # if is_training:
        #     fwd = fwd + tf.random_normal( tf.shape(fwd), mean=0.0, stddev=0.3, seed=seed, dtype=tf.float32 ) #? iGan, badGan, triGan use this and/or dropout 
        fwd = tf.contrib.layers.conv2d(fwd, 32, [3, 1], stride=2, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                                    weights_regularizer=weights_regularizer(l2_weight), scope='c1_1')
        # print('fwd1.name:'.format(fwd1.name))
        # fwd = fwd1

        print('c1_1 shape:',fwd.shape) # name:emb_z_name_scope_1/cnn2/c1_1/BiasAdd:0
        print('======================c1_1 name======================:',fwd.name)
        fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn1_1" ) if USE_BN else fwd
        
        fwd = activation_fn(fwd)
        
        fwd = tf.contrib.layers.conv2d(fwd, 32, [3, 1], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                                weights_regularizer=weights_regularizer(l2_weight), scope='c1_2')
        print('c1_2 shape:',fwd.shape)
        print('======================c1_2 name======================:',fwd.name)
        fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn1_2" ) if USE_BN else fwd
        fwd = activation_fn(fwd)
        fwd = max_pool(fwd, [2, 1], scope='p1')  # 14
        print('======================p1 shape:',fwd.shape)
        fwd = tf.contrib.layers.conv2d(fwd, 64, [3, 1], stride=2, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                                weights_regularizer=weights_regularizer(l2_weight), scope='c2_1')
        print('c2_1 shape:',fwd.shape)
        print('======================c2_1 name======================:',fwd.name)
        fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn2_1" ) if USE_BN else fwd
        fwd = activation_fn(fwd)
        fwd = tf.contrib.layers.conv2d(fwd, 64, [3, 1], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                                weights_regularizer=weights_regularizer(l2_weight), scope='c2_2')
        print('c2_2 shape:',fwd.shape)
        print('======================c2_2 name======================:',fwd.name)
        fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn_2_2" ) if USE_BN else fwd
        fwd = activation_fn(fwd)
        fwd = max_pool(fwd, [2, 1], scope='p2')  # 7
        print('=======================p2 shape:',fwd.shape)
        fwd = tf.contrib.layers.conv2d(fwd, 80, [3, 1], stride=2, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                                weights_regularizer=weights_regularizer(l2_weight), scope='c3_1')
        print('c3_1 shape:',fwd.shape)
        print('======================c3_1 name======================:',fwd.name)
        fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn3_1" ) if USE_BN else fwd
        fwd = activation_fn(fwd)
        # Dropout should NOT be used right before the graph construction (emb_z). It makes the affinity matrix too sparse. Rather, at least a conv before, such as here for example.
        fwd = dropout_f( fwd, rate=0.1, training=is_training, noise_shape=(fwd.get_shape().as_list()[0], 1, 1, fwd.get_shape().as_list()[-1]), seed=seed ) # ala badGan. Doesnt do much, but to show there is no problem training with dropout here.
        fwd = tf.contrib.layers.conv2d(fwd, emb_z_size, [3, 1], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                                weights_regularizer=weights_regularizer(l2_weight), scope='c3_2')
        print('c3_2 shape:',fwd.shape)
        print('======================c3_2 name======================:',fwd.name)
        fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, reuse=is_training, name="bn3_2" ) if USE_BN else fwd
        fwd = activation_fn(fwd)
        
        fwd = avg_pool(fwd, fwd.get_shape().as_list()[1:3], scope='p3')
        print('======================p3 shape:',fwd.shape)
        fwd = tf.layers.flatten(fwd,name='flatten')
        print('flatten shape', fwd.shape)
        print('======================flatten name======================:',fwd.name)
        emb_z = fwd
        print("name of emb_z:{}".format(emb_z.name)) 
        return emb_z


def get_feat_extractor_z_func( feat_extractor_z ):
    # Returns: Pointer to the function for the architecture of feature extractor Z.
    arch_func = None
    if feat_extractor_z == "mnist_cnn":
        arch_func = mnist_cnn
    elif feat_extractor_z == "svhn_cnn":
        arch_func = svhn_cnn
    elif feat_extractor_z == "cifar_cnn":
        arch_func = cifar_cnn
    elif feat_extractor_z == 'iot_fc':
        arch_func = iot_fc
    elif feat_extractor_z == 'iot_cnn':
        arch_func = iot_cnn
    elif feat_extractor_z == 'iot_cnn2':
        arch_func = iot_cnn2
    elif feat_extractor_z == 'iot_cnn0':
        arch_func = iot_cnn0
    return arch_func







