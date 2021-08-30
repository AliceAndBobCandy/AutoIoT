#!/usr/bin/env python

# Copyright (c) 2018, Konstantinos Kamnitsas
# changed by Linna Fan
# This program is free software; you can redistribute and/or modify
# it under the terms of the Apache License, Version 2.0. See the 
# accompanying LICENSE file or read the terms at:
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import absolute_import, division, print_function

import logging
LOG = logging.getLogger('main')

import numpy as np


def makeArrayOneHot(arrayWithClassInts, cardinality, axisInResultToUseForCategories): # cardinality is the num of classes.
    # arrayWithClassInts: np array of shape [batchSize, r, c, z], with sampled ints expressing classes for each of the samples.
    oneHotsArray = np.zeros( [cardinality] + list(arrayWithClassInts.shape), dtype=np.float32 )
    oneHotsArray = np.reshape(oneHotsArray, newshape=(cardinality, -1)) #Flatten all dimensions except the first.
    arrayWithClassInts = np.reshape(arrayWithClassInts, newshape=-1) # Flatten
    
    oneHotsArray[arrayWithClassInts, range(oneHotsArray.shape[1])] = 1
    oneHotsArray = np.reshape(oneHotsArray, newshape=[cardinality] + list(arrayWithClassInts.shape)) # CAREFUL! cardinality first!
    oneHotsArray = np.swapaxes(oneHotsArray, 0, axisInResultToUseForCategories) # in my implementation, axisInResultToUseForCategories == 1 usually.
    
    return oneHotsArray
    
    
def sample_uniformly(array_to_sample, num_samples, rng):
    if num_samples == -1: # return all elements.
        selected_items = array_to_sample
    else:
        if len(array_to_sample)>0:
            selected_items = rng.choice(a=array_to_sample, size=num_samples, replace=False)
    return selected_items


def sample_by_class( list_of_arrays_to_sample_by_class, num_samples_per_c, rng,upsampling=False,percent_lbl_samples=0,new_devices_labels=None):
    # num_samples_per_c: a single integer. Not a list of integers.
    # list_of_arrays_to_sample_by_class: list with number_of_classes arrays. Each array gives the indices to sample from, for class c.
    selected_items_list_by_class = [] # Will be a list of sublists. Each sublist, the indices of selected samples for c.
    num_classes = len(list_of_arrays_to_sample_by_class)
    if num_samples_per_c==0 and percent_lbl_samples>0: # sample according to percentage define in 'percent_lbl_samples' in cfg_iot.py
        num_each_classes = [len(item) for item in list_of_arrays_to_sample_by_class]
        for idx,item in enumerate(num_each_classes):
            if (new_devices_labels is not None) and (idx in new_devices_labels):
                num_samples_c = item
            else:
                num_samples_c = int(percent_lbl_samples*item)
            if num_samples_c < 1:
                num_samples_c = 1
            if len(list_of_arrays_to_sample_by_class[idx])>0:
                selected_items_c = sample_uniformly(list_of_arrays_to_sample_by_class[idx], num_samples_c, rng)
                selected_items_list_by_class.append(selected_items_c)     
        return selected_items_list_by_class      
    if upsampling==False:
        for c in range(num_classes):
            selected_items_c = sample_uniformly(list_of_arrays_to_sample_by_class[c], num_samples_per_c, rng)
            selected_items_list_by_class.append(selected_items_c)           
    else: # if the data is not enough for some classes, the lacked data will be sampled from other classes.mainly used in construct testing dataset
        num_each_classes = [len(item) for item in list_of_arrays_to_sample_by_class]
        if np.min(num_each_classes) >= num_samples_per_c: # if the data is enough, upsampling will not be executed.
            for c in range(num_classes):
                selected_items_c = sample_uniformly(list_of_arrays_to_sample_by_class[c], num_samples_per_c, rng)
                selected_items_list_by_class.append(selected_items_c)
        else:
            lack_counts = 0 # lack_count is the num of lacked samples in classes fewer than num_samples_per_c
            lack_kind = 0 # how many classes lack data
            for item in num_each_classes:
                if item < num_samples_per_c:
                    lack_counts += num_samples_per_c-item
                    lack_kind += 1
            other_classes_extra_samples_num = lack_counts//(num_classes-lack_kind)
            left_sample_num = lack_counts - other_classes_extra_samples_num*(num_classes-lack_kind) #because of the interger problem, there may be data not enough 
            num_samples_per_c_ajusted = [] # sample number ajusted for each class
            last = 0
            for idx,item in enumerate(num_each_classes):
                if item < num_samples_per_c:
                    num_samples_per_c_ajusted.append(item) #not enough, get all for this class
                else:
                    num_samples_per_c_ajusted.append(num_samples_per_c+other_classes_extra_samples_num)#!!!may have problem, because some classes may not have enough item+other_classes_extra_samples_num samples
                    last = idx
            num_samples_per_c_ajusted[last] += left_sample_num #left samples got from the last enough class
            for c in range(num_classes):
                selected_items_c = sample_uniformly(list_of_arrays_to_sample_by_class[c], num_samples_per_c_ajusted[c], rng)
                selected_items_list_by_class.append(selected_items_c)
    return selected_items_list_by_class



        
        