# Implementation of discriminator according to the infocom2020 discriminator 
# the effect is bad

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.training import saver as tf_saver
import matplotlib.pyplot as plt
from configs import local_dirs_to_data
from cclp.sampling import samplers
from cclp.neuralnet.models.discriminator import FC_model, FC_model2, FC_model3, CNN_model
from cclp.data_utils import compute_utils
import random
from sklearn.utils import resample
from cclp.frontend.logging.utils import datetime_str as timestr
import json
import os
from configs.iot.cfg_iot import out_path

threshold = 0.8 # the threshold split the data into new_devices_flag = 0(old data), 1(confidence score<threshold), 2(confidence_score>threshold)
EPOCHS = 30
train_ratio = 0.7
train_per_class_per_batch = 128

class New_devices(object):
    def __init__(self,sessionNModelFlags,trainerFlags):
        self.pathToDataFolder = local_dirs_to_data.iot
        self.new_devices_list = sessionNModelFlags['new_devices_list']
        self.new_devices_postfix = '_'.join(str(item) for item in self.new_devices_list)
        self.main_model_path = out_path + 'trainTf/'
        self.discriminator_model_path = 'D:/IoT/IoT_detection/code/fln/semi_supervised_4/ssl_compact_clustering-master2/ssl_compact_clustering-master/output/iot/discriminator_model/'
        self.val_data = self.generate_val_data()
        self.val_samples = self.val_data[:,1:]
        self.val_labels = self.val_data[:,0]
        self.num_classes = int(np.max(self.val_labels[self.val_labels>=0]))+1
        self.sample_shape = self.val_samples.shape[1:]
        self.val_batch_size = sessionNModelFlags["eval_batch_size"]
        self.val_new_devices_flag = None
        self.seed = sessionNModelFlags["seed"]
        self.new_devices_list = sessionNModelFlags['new_devices_list']
        self.prob_fig = self.main_model_path + self.new_devices_postfix + '/fig_prob'
        if not os.path.exists(self.prob_fig):
            os.makedirs(self.prob_fig)
        self.new_devices_samples = None # the confidence score is higher than threshold
        self.old_new_device_label_dict = self.get_old_new_device_label_dict()
        self.test_new_devices()

    def get_old_new_device_label_dict(self):
        f = open(self.pathToDataFolder + self.new_devices_postfix + '/old_new_device_idx_dict.json')
        self.old_new_device_label_dict = json.load(f)
        return self.old_new_device_label_dict

    def generate_val_data(self):
        data = pd.read_csv(self.pathToDataFolder + self.new_devices_postfix + '/old_devices_test_data.csv')
        data_new_train = pd.read_csv(self.pathToDataFolder + self.new_devices_postfix + '/new_devices_train_data.csv')
        data_new_test = pd.read_csv(self.pathToDataFolder + self.new_devices_postfix + '/new_devices_test_data.csv')
        data = pd.concat([data,data_new_train],axis=0)
        data = pd.concat([data,data_new_test],axis=0)
        return data.values

    # read merged_devices_data, output probablity {label:[prob_distribu1,prob_distribu2,...]}, minus label stands for new devices type, -1000 is 0 before
    def test_new_devices(self):     
        path = self.main_model_path + self.new_devices_postfix
        with open(path + '/model_name','r') as f:
            model_name = f.readline().strip('\n')

        with tf.Session() as sess:
            saver = tf.compat.v1.train.import_meta_graph(self.main_model_path + self.new_devices_postfix + '/' + model_name + '.meta')
            saver.restore(sess, tf.train.latest_checkpoint(self.main_model_path + self.new_devices_postfix))
            graph = tf.compat.v1.get_default_graph()
            # need input and pred value
            # model = graph.get_operation_by_name('model.forward_')
            input_placeholder = sess.graph.get_tensor_by_name('eval_in:0')
            prediction = sess.graph.get_tensor_by_name('compute_logits_name_scope/fully_connected/BiasAdd:0')
            eval_pred_logits = None # list of embeddings for each val batch: [ [batchSize, 10], .... [bs, 10] ]       
                      
            for i in range(0, len(self.val_samples), self.val_batch_size):         
                pred_logits = sess.run(prediction,feed_dict = {input_placeholder:self.val_samples[i:i+self.val_batch_size]})
                val_new_device_flags = compute_utils.get_new_devices_flag(pred_logits,self.val_labels[i:i+self.val_batch_size],threshold)
                if (eval_pred_logits is None):
                    eval_pred_logits = pred_logits                   
                    self.val_new_devices_flag = val_new_device_flags         
                else:    
                    eval_pred_logits = np.concatenate([eval_pred_logits,pred_logits],axis = 0)
                    self.val_new_devices_flag = np.concatenate([self.val_new_devices_flag,val_new_device_flags],axis = 0)
            
        eval_prob_distri = compute_utils.eval_probability_distribution(eval_pred_logits)
        compute_utils.observe_prob_distri(eval_prob_distri,self.val_labels,self.new_devices_list,self.prob_fig,self.old_new_device_label_dict)
        print("finish getting prob distribution of new devices") 

    def resample_data_c(self,data_c,add_num = 4000):
        data_resampled = resample(data_c, 
                                replace=True,     # sample with replacement
                                n_samples=add_num,    # to match majority class
                                random_state=self.seed) # reproducible results
        return data_resampled
    def train_val_split(self,data_samples,data_labels,train_ratio):
        train_data_samples, train_data_labels, val_samples, val_labels = None, None, None, None
        for i in range(2): # 0 stands for old, 1 stands for new
            bool_data_c = data_labels == i
            if i==1:
                train_size = int(4000 * train_ratio)
            else:
                train_size = int(np.sum(bool_data_c) * train_ratio) #???
            if train_size < 1:
                train_size = 1
            data_c, label_c = data_samples[bool_data_c], data_labels[bool_data_c]
            if i==1:
                data_c = self.resample_data_c(data_c,4000)
                label_c = np.array([1] * 4000)
            indices = range(len(data_c))
            train_indices = random.sample(indices, train_size)
            
            val_indices = np.array(indices)[np.invert(np.isin(indices,train_indices))]
            train_data_c, train_label_c, val_data_c, val_label_c = data_c[train_indices], label_c[train_indices], data_c[val_indices], label_c[val_indices]
            if train_data_samples is None:
                train_data_samples = train_data_c
                train_data_labels = train_label_c
            else:
                train_data_samples = np.concatenate([train_data_samples,train_data_c])
                train_data_labels = np.concatenate([train_data_labels,train_label_c])
            if val_samples is None:
                val_samples = val_data_c
                val_labels = val_label_c
            else:
                val_samples = np.concatenate([val_samples, val_data_c])
                val_labels = np.concatenate([val_labels, val_label_c])
        return train_data_samples, train_data_labels, val_samples, val_labels

    # judge new devices again, use old data and new data train discriminator, construct a MLP and train it use val_samples and val_new_devices_flag
    def discriminator_create(self,train=False): 
        data_samples_bool = self.val_new_devices_flag!=2
        data_samples = self.val_samples[data_samples_bool]
        data_labels = self.val_new_devices_flag[data_samples_bool]
        # split data_samples, data_labels into train and val
        train_data_samples, train_data_labels, val_data_samples, val_data_labels = self.train_val_split(data_samples,data_labels,train_ratio)
        print("label 0 has {} samples, and label 1 has {} samples".format(train_data_samples.shape[0],val_data_samples.shape[0]))
        # shuffle
        train_data = np.concatenate([train_data_labels.reshape(-1,1),train_data_samples],axis = 1)
        np.random.shuffle(train_data)
        train_data_samples, train_data_labels = train_data[:,1:], train_data[:,0]

        self.new_devices_samples = self.val_samples[np.invert(data_samples_bool)]
        if train == False:
            return

        n_batch_size = 128
        n_batch = train_data_samples.shape[0]//n_batch_size
        
        graph = tf.Graph()
        with graph.as_default():
            print("creating graph for the discriminator")
            # train_step, ph_x, ph_y, accuracy, loss, lr, prediction_2 = FC_model2(sample_shape=self.sample_shape, num_classes=2, is_training=True,seed = self.seed )
            train_step, ph_x, ph_y, accuracy, loss, lr, prediction_2 = CNN_model(sample_shape=self.sample_shape, num_classes=2, is_training=True,seed = self.seed )
            saver = tf_saver.Saver(max_to_keep=1)
        with tf.Session(graph = graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(EPOCHS):               
                sess.run(tf.assign(lr, 0.0001*(0.99**(epoch//100))))
                for i in range(0,train_data_samples.shape[0],n_batch_size):
                    sess.run(train_step, feed_dict={
                        ph_x: train_data_samples[i:i+n_batch_size],
                        ph_y: train_data_labels[i:i+n_batch_size]
                    })                  
                # validation 
                if epoch % 20 == 0:
                        learning_rate = sess.run(lr)
                        print('epoch:{},iteration:{}'.format(epoch,i))
                        [accuracy_value, loss_value] = sess.run([accuracy,loss],feed_dict={
                            ph_x: train_data_samples,
                            ph_y: train_data_labels
                        }) 
                        print('train accuracy:{}, loss:{}'.format(accuracy_value, loss_value))
                        [accuracy_value2, loss_value2] = sess.run([accuracy,loss],feed_dict={
                            ph_x: val_data_samples,
                            ph_y: val_data_labels
                        }) 
                        print('val accuracy:{}, loss:{}, lr = {}'.format(accuracy_value2, loss_value2, str(learning_rate)))    
            # SAVE MODEL changed by fln
            filepath_to_save = self.discriminator_model_path # + 'model-{}'.format(timestr)
            # LOG.info("[SAVE] Saving model at: "+str(filepath_to_save))
            print("save model at "+str(filepath_to_save))
            saver.save(sess, filepath_to_save, global_step=EPOCHS, write_meta_graph=True, write_state=True ) # creates the subfolder if doesnt exist. Appends step if given.        
    
    def discriminator_judge(self):
        tf.compat.v1.reset_default_graph() 
        with tf.Session() as sess:
            saver = tf.compat.v1.train.import_meta_graph(self.discriminator_model_path + '-30.meta')
            saver.restore(sess, tf.train.latest_checkpoint(self.discriminator_model_path))
            graph = tf.compat.v1.get_default_graph()
    
            ph_x = sess.graph.get_tensor_by_name('discriminator_place_holder_x:0')
            prediction = sess.graph.get_tensor_by_name('Softmax:0')
            prediction_softmax = sess.run(prediction,feed_dict={
                                            ph_x: self.new_devices_samples
                                        })
            prediction = prediction_softmax.argmax(axis = 1)
            accuracy = np.sum(prediction)/len(prediction)
            print('accuracy of the new devices found by the discriminator is:{}'.format(accuracy))

    # retrain the main CNN model use new found devices data
    def retrain_model(self):
        pass

        
    