#!/usr/bin/env python
#
# Copyright (c) 2018, Konstantinos Kamnitsas
# modified by Linna Fan
# This program is free software; you can redistribute and/or modify
# it under the terms of the Apache License, Version 2.0. See the 
# accompanying LICENSE file or read the terms at:
# http://www.apache.org/licenses/LICENSE-2.0

# To see the help and options, in command line try:
# python ./run.py -h
# Example usage from the command line:
# python ./run.py ./configs/mnist/cfg_mnist100.py -o ./output/ -dev 0
# read config/cmd parameters and call train
# this file is modified by Linna Fan
from __future__ import absolute_import, division, print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import sys

import logging

from cclp.frontend.logging.loggers import setup_loggers
from cclp.frontend.config_flags import TrainSessionNModelFlags, TrainerFlags
from cclp.routines import train
from cclp.routines.test_new_devices_multi_instance import New_devices # ks test查找新设备
# from cclp.routines.test import New_devices # 使用discriminator思路找到新设备。
os.environ['KMP_WARNINGS'] = 'off'

def setup_arg_parser() :
    parser = argparse.ArgumentParser( prog='Compact Clustering via Label Propagation (CCLP)', formatter_class=argparse.RawTextHelpFormatter,
                                      description=  "This software accompanies the research presented in:\n"+\
                                                    "Kamnitsas et al, Semi-supervised learning via compact latent space clustering, ICML 2018.\n"+\
                                                    "We hope our work aids you in your endeavours.\n"+\
                                                    "For questions and feedback contact: konstantinos.kamnitsas12@ic.ac.uk" )
    # Required
    parser.add_argument(dest='cfg_file', type=str, help="[REQUIRED] Path to config file.")
    # Optional cfg that overrides values given in config file.
    parser.add_argument("-o", dest='out_path', type=str, help="Path for output folder.")
    parser.add_argument("-dev", dest='device', type=int, help="Which device to use. Give -1 to use CPU. Give number (0,1,2...) to use the correcpoding GPU.")
    parser.add_argument("-load", dest='model_to_load', type=str, help="At the beginning of training, load model parameters from pretrained model.\n"+\
                                                                      "Give \"self\" to load last checkpoint from own output dir, eg to continue training.\n"+\
                                                                      "Else, give path to any pre-trained model.")
    parser.add_argument("-pe", dest='plot_save_emb', type=int, help="Plot embedding of training batch (only works for 2D emb): 0=dont plot dont save. 1=save. 2=plot. 3=save and plot.")
    # Optionals, cmd line only.
    parser.add_argument("-ll", dest='cmd_loglevel', type=str, default="d", help="d=debug, i=info, etc. Default: [d]")
    #------------------------- fln add-------------------------------
    parser.add_argument("-new_devices_list",dest='new_devices_list',type=str,default=None)
    parser.add_argument("-use_cnn_layer_for_cluster",dest='use_cnn_layer_for_cluster',type=str,default=None)
    parser.add_argument("-merge_cnn_layer_methods",dest='merge_cnn_layer_methods',type=str,default=None)
    parser.add_argument("-upsampling",dest='upsampling',type=bool,default=None)
    parser.add_argument("-session_type",dest='session_type',type=str,default=None)
    parser.add_argument("-dataset_name",dest='dataset_name',type=str,default=None)
    return parser

def override_file_cfg_with_cmd_cfg(cfg, cmd_args):
    cfg['out_path'] = os.path.abspath( cmd_args.out_path if cmd_args.out_path is not None else cfg['out_path'] )
    cfg['plot_save_emb'] = cmd_args.plot_save_emb if cmd_args.plot_save_emb is not None else cfg['plot_save_emb']
    cfg['device'] = cmd_args.device if cmd_args.device is not None else cfg['device']
    cfg['model_to_load'] = cmd_args.model_to_load if cmd_args.model_to_load is not None else cfg['model_to_load'] # if "self", it will change later in config_flags.py
    # 1st round modification
    # dispose cmd_args.new_devices_list
    if cmd_args.new_devices_list is not None:
        new_devices_list_cmd = cmd_args.new_devices_list.split(',')
        new_devices_list_cmd = [int(item) for item in new_devices_list_cmd]
        cmd_args.new_devices_list = new_devices_list_cmd

    cfg['new_devices_list'] = cmd_args.new_devices_list if cmd_args.new_devices_list is not None else cfg['new_devices_list']
    print('cmd_args.use_cnn_layer_for_cluster',cmd_args.use_cnn_layer_for_cluster)
    cfg['use_cnn_layer_for_cluster'] = cmd_args.use_cnn_layer_for_cluster if cmd_args.use_cnn_layer_for_cluster is not None else cfg['use_cnn_layer_for_cluster']
    if cfg['use_cnn_layer_for_cluster'] == '0':
        cfg['use_cnn_layer_for_cluster'] = False
    elif cfg['use_cnn_layer_for_cluster'] == '1':
        cfg['use_cnn_layer_for_cluster'] = True
    cfg['merge_cnn_layer_methods'] = cmd_args.merge_cnn_layer_methods if cmd_args.merge_cnn_layer_methods is not None else cfg['merge_cnn_layer_methods']
    cfg['upsampling'] = cmd_args.upsampling if cmd_args.upsampling is not None else cfg['upsampling']
    if cfg['upsampling'] == '0':
        cfg['upsampling'] = False
    elif cfg['upsampling'] == '1':
        cfg['upsampling'] = True
    cfg['session_type'] = cmd_args.session_type if cmd_args.session_type is not None else cfg['session_type']
    cfg['dataset_name'] = cmd_args.dataset_name if cmd_args.dataset_name is not None else cfg['dataset_name']
    if cfg['device'] is None:
        raise ValueError('Invalid device given in file and cmd: %s' % cfg['device'])
    
# def check_output_dir_ok( out_path, model_to_load ):# not used?
#     if os.path.exists(out_path) and model_to_load != "self":
#         user_input = None
#         try:
#             user_input = raw_input("The output directory exists already, and [-load] was NOT specified.\n"+\
#                                    "We may corrupt another session.\n" +\
#                                    "Output path given: "+str(out_path)+"\n"+\
#                                    "Are you sure you want to continue? [y/n] : ")
#             while user_input not in ['y','n']: 
#                 user_input = raw_input("Please specify 'y' or 'n': ")
#         except:
#             print("\nERROR:\tOutput directory already exists. Tried to ask for user input whether to continue, but failed."+\
#                   "\n\tReason unknown (nohup? remote?)."+\
#                   "\n\tSpecify other dir and retry."+\
#                   "\n\tExiting."); exit(1)
#         if user_input == 'y':
#             pass
#         else:
#             print("Exiting as requested."); exit(0)
    
def create_session_folders(main_out_dir): # create output dir according to main_out_dir
    # Must be given absolute dir.
    # Create only the main folder (and its parent if not existing).
    parent_out_folder = os.path.abspath( os.path.join(main_out_dir, os.pardir) )
    if not os.path.exists(parent_out_folder):
        os.mkdir(parent_out_folder)
    if not os.path.exists(main_out_dir):
        os.mkdir(main_out_dir)
    
def print_param(config_file):
    # print('session_type:',cfg['session_type'])
    # print('new_devices_list:',cfg['new_devices_list'])
    # print('use_cnn_layer_for_cluster:',cfg['use_cnn_layer_for_cluster'])
    # print('merge_cnn_layer_methods:',cfg['merge_cnn_layer_methods'])
    # print('upsampling:',cfg['upsampling'])
    for key,value in config_file.items():
        print(key,value)

if __name__ == '__main__':
    
    parser = setup_arg_parser()
    args = parser.parse_args()
    print('args',args)
    
    cfg_file = os.path.abspath( args.cfg_file )
    cfg = {}
    exec(open(cfg_file).read(), cfg) # Now the cfg is populated.
    # Override file-config with command-line config if given.
    override_file_cfg_with_cmd_cfg(cfg, args)
    
    # Check whether output dir exists or not, and set it up
    # check_output_dir_ok( out_path=cfg['out_path'], model_to_load=cfg['model_to_load'] )
    # print parameters in cfg_file 
    print_param(cfg)
    create_session_folders( cfg['out_path'] )
    
    # Setup logger
    setup_loggers( cmd_loglevel=args.cmd_loglevel, logfile=os.path.join( cfg['out_path'], "logs") )
    log = logging.getLogger('main') # created in loggers.py.
    log.info("\n======================== Starting new session ==========================\n")
    
    # Print command line args.
    log.info("Command line arguments given: \n"+str(args))
    
    # Setup cuda
    if cfg['device'] == -1: # Set it up for CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else: # Goes to corresponding
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['device'])
    
  

    try:
        if cfg['session_type'] == 'train':
            sessionNModelFlags = TrainSessionNModelFlags(cfg) # parameters about session model
            sessionNModelFlags['retrain'] = False
            trainerFlags = TrainerFlags(cfg) # parameters about trainer
            log.info("================ PRINTING CONFIGURATION ====================")
            sessionNModelFlags.print_params() #put session, model parameters into sessionNModelFlags class
            trainerFlags.print_params() # put training parameters into trainerFlags class
            train.train( sessionNModelFlags=sessionNModelFlags, trainerFlags=trainerFlags ) # core
        # ==========================this part is added by Linna Fan====================================
        elif cfg['session_type'] == 'test': # test new device          
            sessionNModelFlags = TrainSessionNModelFlags(cfg) # parameters about session model
            sessionNModelFlags['retrain'] = True
            trainerFlags = TrainerFlags(cfg) # parameters about trainer
            log.info("================start testing new device finding==================")
            new_devices = New_devices(sessionNModelFlags, trainerFlags)

            # new_devices.test_new_devices()
            # discriminator cannot be used 
            # new_devices.discriminator_create(train=True)
            # new_devices.discriminator_judge()
            # new_devices.retrain_model()
        else:
            raise ValueError('Invalid session type: %s' % cfg['session_type'])
    except Exception as e:
        log.exception("Got exception during main process..!")
            
    log.info("Process finished.")
    logging.shutdown() # Closes all handlers everywhere.

    
    
    
    
    
    
    
    
    
    
    
    
