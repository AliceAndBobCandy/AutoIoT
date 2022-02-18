#!/usr/bin/env python
import random
import os
import inspect, os.path
filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
pardir = os.path.dirname(os.path.dirname(path).replace("\\","/"))
# Variables needed for pre-setting up the session.
dataset_name = 'tmc' # 
session_type = 'test' # can be train/test
# ========== Variables needed for the session i24,2tself. ========
# === Variables that are read from the cmd line too. ===
# WARN: Values given in cmd line overwrite these given below.
out_path = pardir + '/output/iot/'
device = 1 # GPU device number,-1 is to use CPU
model_to_load = None # To start from pretrained model/continue training.
plot_save_emb = 0 # Plot embedding if emb_size is 2D. 0: No. 1: save. 2: plot. 3: save & plot.

# === Variables not given from command line ===
# --- Network architecture ---
# feat_extractor_z is iot_fc / iot_cnn

model = {'feat_extractor_z': 'iot_cnn2', 'emb_z_size': 80, 'emb_z_act': 'elu',
         'batch_norm_decay': 0.99, 'l2_reg_feat': 1e-4, 'l2_reg_classif': 1e-4, 'classifier_dropout': 0.5}
# parameters added for data unsampling
upsampling = True # if the labeled data is less than num_least,whether to upsampling or not
upsample_niot = -1 # -1 stands for no upsample for non-iot
num_least = 100 # num. of resampled data in each class 
epsilon = 1e-1 # in order to prevent overfitting, the resampled data is added a disturbance epsilon
prob = 0 # probability that a col is added epsilon
disturbe = 0 # 0 stands for no disturb, 1 stands for SMOTE


#------------------------------------------------------------------------
threshold = 0.7 # multi label model threshold, default value is 0.7
# --- Validation, labelled and unlabelled folds for training ---
dataset = 'iot' # mnist, svhn, cifar
val_on_test = False  # If True, validate on test dataset. Else, validate on subset of training data.
num_val_samples = 4726  # How many samples to use for validation when training. -1 to use all. #4726 tmc,5353 yt
num_lbl_samples = 750  # do not used if percent_lbl_samples is defined. (all classes data) How many labelled data to learn from. -1 to use all.some class may not have enough samples
percent_lbl_samples = 0.08 #label data percentage in each class, 0 stands for not use this percentage but use num_lbl_samples
percent_lbl_samples_small = 0.08
lbl_samples_small = 40
train_ratio = 0.7
num_unlbl_samples = -1  # How many unlabelled data to learn from. -1 to use all.
unlbl_overlap_val = False  # If True and val_on_test=False, unlabelled samples may overlap with validation.
unlbl_overlap_lbl = False  # If True, unlabelled samples can overlap with labelled. If False, they are disjoint.
# --- Batch sampling, normalization, augmentation ---
n_lbl_per_class_per_batch = 20 # How many labelled samples per class in a batch. if some classed have instances less than 10, then use as more data as possible.
n_unlbl_per_batch = 100 # How many unlabelled samples in a batch.
norm_imgs_pre_aug = None # do not need here,None, zscoreDb, zscorePerCase, center0sc1, rescale01, zca, zca_center0sc1.
augm_params = {}

# Augmentation options:
# augm_params = { "reflect":{"apply": True},
#    "color_inv": {"apply": False, "params": {"p":0.5}},
#    "rand_crop": {"apply": False, "params": {'transf':[2,2]}},
#    "blur": {"apply": False, "params": {'p':0.5, 's':1.0, 'sc':1}} }

seed = 1 # for preproduce result

# --- Training loop ---
#added
# multi task model
use_multi_task_model = 4 #1:multi label, 2:multi task 0:original model, 3:single neuron added at last, 4: multi task 24IoT + IoT/NoT
#--------------------------
#-----dynamic cclp loss weight---------------
dynamic_loss_weight = False
#------------------------------------------------------------------------
max_iters = 2000 # Max training iterations
val_during_train = True # Whether to validate performance every now and then.
val_interval = 200 # Every how many training steps to validate performance.
# Learning rate schedule
lr_sched_type = 'expon_decay' # 'expon_decay' or 'piecewise'
lr_expon_init = 1e-3 # Only for expon. Initial LR.
lr_expon_decay_factor = 0.333  # Only for expon. How much to decrease.
lr_expon_decay_steps = 2000  # Only for expon. How often to decrease.
lr_piecewise_boundaries = None # Only for expon. When to change LR.
lr_piecewise_values = None # Only for expon. Initial and following values.

# --- Compact Clustering via Label Propagation (CCLP) ---
cc_weight = 5 # Weight w in: Ltotal = Lsup + w*Lcclp . Set to 0 to disable CCLP.
cc_steps = 3 # Length of longest chain to optimize. Set to 0 to disable CCLP.
# cc_loss_on = (cc_steps > 0) or (cc_weight > 0) # Set to False to disable.
cc_loss_on = True
# cc_loss_distance = 'warsserstein' # wasserstein stands for wasserstein loss, else it is crossentropy warsserstein, crossentropy, ...
cc_loss_distance = 'crossentropy'
# Params for creating the graph.
cc_sim_metric = "dot" # dot or L2, similarity metric for creating the graph.
cc_l2_sigmas_init = 1.0 # Only for L2. float or list of floats per dim.
cc_l2_sigmas_trainable = True # Only for L2. Whether to learn the sigmas.
cc_l2_sigmas_lr_multipl = 1.0 # Only for L2.
# Secondary configs for CCLP.
cc_sum_over_chains = True # If False, only the longest chain is optimized.
cc_e_smooth = 0.00001
cc_optim_smooth_mtx = True
#===========================================find new devices=================================================
new_devices_list = [0,1] # if None, use all data, else the training set should exclude this list
# new_devices_list = None
if dataset_name == 'tmc':
    niot_label = 24
else:
    niot_label = 45
record_data_index = False
record_best_score = False

#================================================1st round modification======================================
use_cnn_layer_for_cluster = False # when cluster, we use the emb features after cnn layer
merge_cnn_layer_methods = 'sum' # or 'mean'