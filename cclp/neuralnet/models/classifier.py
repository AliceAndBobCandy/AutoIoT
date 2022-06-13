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


class Classifier(object):
    
    def __init__(self,
                 augment_tens_func,
                 normalize_tens_func,
                 embed_tens_func,
                 num_classes,
                 inp_shape,
                 l2_reg_classif,
                 classifier_dropout,
                 use_multi_task_model=0):
        
        # This is the main neural network class.
        # Initialize it by passing the partially defined functions that it should apply to each batch
        # Then, call forward_pass(batch, ...) to build the tf graph for the forward pass, where batch is a tensor from batch-generator.
        # augment_tens_func: pointer to partially defined augmentation function. Expects only batch-tensor.
        # normalize_tens_func: pointer to partially defined re-normalization (after augm) function. Expects only batch-tensor. Unused.
        # embed_tens_func: pointer to partially defined feature_extractor. Expects only batch & is_training
        # inp_shape: list [height, width, number of channels]
        # l2_reg_classif: l2 weight regularization, applied on the classification layer's weights.
        # classifier_dropout: float, drop rate applied right before the classification layer.
        
        self._augment_tens_func = augment_tens_func
        self._normalize_tens_func = normalize_tens_func
        self._embed_tens_func = embed_tens_func
        
        LOG.debug("Classifier.__init__: num_classes = "+str(num_classes)+", inp_shape = "+str(inp_shape))
        self._num_classes = num_classes
        self._inp_shape = inp_shape
        self._l2_reg_classif = l2_reg_classif
        self._classifier_dropout = classifier_dropout
        
        self._step = tf.Variable(0, trainable=False, name='global_step')
    
        # This will hold tensors for "families": labelled and unlabelled training data, or validation data.
        self.tensor_families = {}
        self.use_multi_task_model = use_multi_task_model
    def get_num_classes(self):
        return self._num_classes
    
    def get_t_step(self):
        return self._step
    
    def forward_pass(self, input, tensor_family, is_training, init_vars, seed=None, image_summary=False):
        # Forward pass through the feature extractor AND the classifier. Call this separately for labelled and unlabelled inputs.
        # tensor_family: "eval" or "train_sup" or "train_unsup". To keep track of the corresponding tensors...
        # tensor_families[tensor_family] = {"inp_tens":..., "emb_z_tens": ..., "logits_tens": .... }
        self.tensor_families[tensor_family] = {}
        tensor_fam = self.tensor_families[tensor_family]
        if input is None:
            tensor_fam["inp_tens"] = tf.placeholder("float32", [None] + list(self._inp_shape), tensor_family+'_in') # + inp_shape, 'test_in')
            tensor_fam["inp_tens_is_plchldr"] = True
        else:
            tensor_fam["inp_tens"] = input
            tensor_fam["inp_tens_is_plchldr"] = False
            
        tensor_fam["emb_z_tens"] = self._compute_emb_z_of_imgs(tensor_fam["inp_tens"], is_training=is_training, init_vars=init_vars, image_summary=image_summary)
        if self.use_multi_task_model==2 or self.use_multi_task_model==4:
            tensor_fam["logits_tens"],tensor_fam["logits_tens_2"] =  self._compute_logits_of_emb_z(tensor_fam["emb_z_tens"], is_training=is_training, init_vars=init_vars, seed=seed)
        else:
            tensor_fam["logits_tens"] = self._compute_logits_of_emb_z(tensor_fam["emb_z_tens"], is_training=is_training, init_vars=init_vars, seed=seed)
    
    def _compute_emb_z_of_imgs(self, batch, is_training, init_vars=False, image_summary=False):
       
        with tf.variable_scope('emb_z_name_scope',reuse=(not init_vars)):
            # batch: Tf tensor. Either from a batcher, or a placeholder that will be given values at inference time.
            batch_preproc = tf.cast(batch, tf.float32) #data format change
            # Apply augmentation if requested in config.
            if is_training:
                batch_preproc = self._augment_tens_func(inp_batch=batch_preproc)
            if image_summary:
                tf.summary.image('inputs_after_augm', batch_preproc, max_outputs=3, collections=[tf.GraphKeys.SUMMARIES] if is_training else ["eval_summaries"]) # I should make this "family"
            # Re-normalize, after augmentation, if requested in config.
            batch_preproc = self._normalize_tens_func(tens_batch=batch_preproc)
            # Do forward pass through the feature extractor z to get embedding z.
            emb_z = self._embed_tens_func(batch_preproc, is_training=is_training)
        return emb_z 
        
            
    def _compute_logits_of_emb_z(self, emb_z, is_training, init_vars=False, seed=None):
        # Adds classification layer.
       
        weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=seed, dtype=tf.float32)
        activation_fn = tf.nn.elu
        with tf.variable_scope('compute_logits_name_scope', reuse=(not init_vars)):
            emb_z = tf.layers.dropout( emb_z, rate=self._classifier_dropout, training=is_training, seed=seed, name='drop5' )
            if self.use_multi_task_model==1: # add two neurons at last
                logits = tf.contrib.layers.fully_connected(
                            emb_z,
                            self._num_classes+2,#！
                            weights_initializer=weights_initializer,
                            activation_fn=None,
                            weights_regularizer=tf.contrib.layers.l2_regularizer( self._l2_reg_classif ))
            elif self.use_multi_task_model==2:
                logits_1_1 = tf.contrib.layers.fully_connected(
                            emb_z,
                            60,
                            weights_initializer=weights_initializer,
                            activation_fn=None,
                            weights_regularizer=tf.contrib.layers.l2_regularizer( self._l2_reg_classif ))
                logits_1_2 = tf.contrib.layers.fully_connected(
                            logits_1_1,
                            self._num_classes,
                            weights_initializer=weights_initializer,
                            activation_fn=None,
                            weights_regularizer=tf.contrib.layers.l2_regularizer( self._l2_reg_classif ))
                logits_2_1 = tf.contrib.layers.fully_connected(
                            emb_z,
                            60,
                            weights_initializer=weights_initializer,
                            activation_fn=None,
                            weights_regularizer=tf.contrib.layers.l2_regularizer( self._l2_reg_classif ))
                logits_2_2 = tf.contrib.layers.fully_connected(
                            logits_2_1,
                            30,
                            weights_initializer=weights_initializer,
                            activation_fn=None,
                            weights_regularizer=tf.contrib.layers.l2_regularizer( self._l2_reg_classif ))
                logits_2_3 = tf.contrib.layers.fully_connected(
                            logits_2_2,
                            2,
                            weights_initializer=weights_initializer,
                            activation_fn=None,
                            weights_regularizer=tf.contrib.layers.l2_regularizer( self._l2_reg_classif ))       
                return logits_1_2, logits_2_3  
            elif self.use_multi_task_model == 4:
               
                logits_1_2 = tf.contrib.layers.fully_connected(
                            # logits_1_1,
                            emb_z,
                            self._num_classes-1,
                            weights_initializer=weights_initializer,
                            activation_fn=None,
                            weights_regularizer=tf.contrib.layers.l2_regularizer( self._l2_reg_classif ))
                print('name of logits 1_2:{}'.format(logits_1_2.name)) # compute_logits_name_scope/fully_connected/BiasAdd:0
               
                logits_2_3 = tf.contrib.layers.fully_connected(
                            # logits_2_2,
                            emb_z,
                            2,
                            weights_initializer=weights_initializer,
                            activation_fn=None,
                            weights_regularizer=tf.contrib.layers.l2_regularizer( self._l2_reg_classif ))    
                print("logits_2_3 name:{}".format(logits_2_3.name)) # compute_logits_name_scope/fully_connected_1/BiasAdd:0
                return logits_1_2, logits_2_3  
            elif self.use_multi_task_model == 3:
                logits = tf.contrib.layers.fully_connected(
                            emb_z,
                            self._num_classes+1,#！
                            weights_initializer=weights_initializer,
                            activation_fn=None,
                            weights_regularizer=tf.contrib.layers.l2_regularizer( self._l2_reg_classif ))   
            else:
                logits = tf.contrib.layers.fully_connected(
                            emb_z,
                            self._num_classes,
                            weights_initializer=weights_initializer,
                            activation_fn=None,
                            weights_regularizer=tf.contrib.layers.l2_regularizer( self._l2_reg_classif ))
                # add softmax layer
            # logits = tf.nn.softmax(logits,name='output_softmax')
            return logits


# Unused / deprecated.
def add_average(ema_inst, variable):
    # Adds moving average of the given tensor to the model.
    # The moving-av is updated at every optimization step via the OP added in the collection.
    # To get the average of the var, (eg for the graph-for-testing) call: average_variable.average() .
    # variable: a tensor Eg the training tensor variable of a kernel. The return of this is the kern that should be used at test.
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_inst.apply([variable]))
    # note that .apply can be called multiple times for the same ema instance, with different list of vars. It can handle many internally.
    average_variable = tf.identity( ema_inst.average(variable), name=variable.name[:-2] + '_avg' ) # just to give it a name.
    return average_variable
















