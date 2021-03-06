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

import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import colors as c
from mpl_toolkits.mplot3d import Axes3D
# This should be used only if the embedding / latent space is configured to be 2-dimensional. For visualisation.
# Size of embedding configured in configs -> model = { .... "emb_z_size": 2 ...}
# Activate plotting by setting in configs -> plot_save_emb = 0: No. 1: save. 2: plot. 3: save & plot.

# cMap = c.ListedColormap(['y','b','r','g','m','navy', 'c', 'cornflowerblue', 'gold', 'darkorange','k'])
cMap = c.ListedColormap(['black','grey','lightcoral', 'brown', 'maroon', 'red', 'sienna', 'chocolate', 'saddlebrown',
 'orange', 'yellow', 'olivedrab', 'yellowgreen', 'darkolivegreen', 'teal', 'deepskyblue',
'slategray', 'cornflowerblue', 'navy', 'blue', 'blueviolet','indigo', 'violet','pink','deeppink']) #'plum'

# cMap = c.ListedColormap(['#f42a2a', '#f86132', '#f48218', '#fdbb12', '#fbf108', '#ccf622', '#9af922',
#                          '#53fb04', '#19fb06', '#0cf932', '#24fe7b', '#27faae', '#17f7dc', '#14defa', 
#                          '#29affa', '#267bfb', '#2b4dfb', '#1905fe', '#702ff9', '#a336f8', '#ca27f3', 
#                          '#fa1af1', '#f813b8', '#f81a85', '#f42658'])                       
def normalize_emb_axis_to_range(list_of_embs, target_range=[0,100], dtype=None):
    # emb_of_samples: numpy array of size: [batchSize, embedding_size]
    # target_range: [min, max]. Commonly [0,100], eg to visualize them in an image of size 100x100 pixels.
    
    #min_int_old = np.min(emb_of_samples, axis=(0), keepdims=True).astype("float32")
    #max_int_old = np.max(emb_of_samples, axis=(0), keepdims=True).astype("float32")
    min_int_old = min( [np.min(emb).astype("float32") for emb in list_of_embs] )
    max_int_old = max( [np.max(emb).astype("float32") for emb in list_of_embs] )
    
    LOG.debug("VISUALIZE EMBEDDING: normalization of the int range: min_int_old = "+str(min_int_old)+", max_int_old = "+str(max_int_old))
    min_int_new = target_range[0]
    max_int_new = target_range[1]
    
    list_of_normed_embs = []
    for emb in list_of_embs:
        normed_emb = emb.astype("float32")
        normed_emb = (normed_emb - min_int_old) / (max_int_old - min_int_old) # now it's from 0 to 1
        normed_emb = normed_emb*(max_int_new - min_int_new) + min_int_new
    
        if dtype is not None:
            normed_emb = normed_emb.astype(dtype)
            
        list_of_normed_embs.append(normed_emb)
        
    return list_of_normed_embs
    
    
def emb_array_to_img_arr(normed_emb_train_sup, norm_emb_train_unsup, nparr_t_sup_labels, axis_range,
                         train_step, save_emb, plot_emb, output_folder):
    
    if not plot_emb:
        # Change backend, so that it does not need the X-server for the monitor. Otherwise calling imshow will throw error.
        # Guide: https://stackoverflow.com/questions/19518352/tkinter-tclerror-couldnt-connect-to-display-localhost18-0
        # Agg for png, other backends available: https://matplotlib.org/faq/usage_faq.html#what-is-a-backend
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
        
    LOG.debug("normed_emb_train_sup.shape = "+str(normed_emb_train_sup.shape))
    LOG.debug("norm_emb_train_unsup.shape = "+str(norm_emb_train_unsup.shape))
    
    assert normed_emb_train_sup.shape[1] == 2 and norm_emb_train_unsup.shape[1] == 2
    assert normed_emb_train_sup.shape[0] == nparr_t_sup_labels.shape[0]
    img_width = axis_range[1] - axis_range[0] + 1
    img_shape = [img_width]*2
    
    # Create background image:
    img_bg = np.zeros(img_shape)
    cmap_bg = plt.get_cmap('gray_r') # 0 = white
    plt.imshow(img_bg, cmap=cmap_bg, interpolation='none', vmin=0, vmax=1)
    
    # Create MASKED image of labeled: https://stackoverflow.com/questions/36910410/values-at-coordinates-to-image
    img_sup = np.ma.array( np.ones( img_shape )*(-1) )
    img_sup[normed_emb_train_sup[:,0], normed_emb_train_sup[:,1]] = nparr_t_sup_labels
    # set mask:
    img_sup.mask = ( img_sup == -1 )
    # Colormaps: https://matplotlib.org/examples/color/colormaps_reference.html
    # https://matplotlib.org/devdocs/api/_as_gen/matplotlib.colors.Colormap.html#matplotlib.colors.Colormap
    cmap_sup = plt.get_cmap('jet') # "gray_r"
    # Custom colormap:
    # https://stackoverflow.com/questions/9707676/defining-a-discrete-colormap-for-imshow-in-matplotlib
    # https://stackoverflow.com/questions/36377638/how-to-map-integers-to-colors-in-matplotlib
    plt.imshow(img_sup, cmap=cmap_sup, interpolation='none', vmin=np.min(nparr_t_sup_labels), vmax=np.max(nparr_t_sup_labels))
    plt.colorbar() #plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[0, 5, 10])
    
    # Create image of unlabeled
    img_unsup = np.ma.array( np.zeros( img_shape ))
    img_unsup[norm_emb_train_unsup[:,0], norm_emb_train_unsup[:,1]] = 1
    # set mask:
    img_unsup.mask = ( img_unsup == 0 )
    cmap_unsup = plt.get_cmap('gray_r') # 1 will be black
    plt.imshow(img_unsup, cmap=cmap_unsup, interpolation='none', vmin=0, vmax=1)
    
    if save_emb:
        assert output_folder is not None
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        plt.savefig( os.path.join(output_folder, "tr_emb_step_"+str(train_step)+".png" ) )
        
    if plot_emb:
        try:
            plt.show()
        except:
            LOG.warn("Tried to plot but something went wrong. Probably nohup, remote use or piped output? Continuing...")
            
    plt.close()
    
def plot_2d_emb( emb_train_sup, emb_train_unsup, nparr_t_sup_labels, train_step, save_emb=True, plot_emb=False, output_folder=None ):
    target_range = [0,100]
    normed_emb_train_sup, norm_emb_train_unsup = normalize_emb_axis_to_range( [emb_train_sup, emb_train_unsup], target_range=target_range, dtype="int")
    emb_array_to_img_arr(normed_emb_train_sup, norm_emb_train_unsup, nparr_t_sup_labels, axis_range=target_range,
                         train_step=train_step, save_emb=save_emb, plot_emb=plot_emb, output_folder=output_folder)

#----------------------------------------------------------------------------------------
def plt_scatter(encoded_imgs, label, azim, elev, title="latent_space", save=True):
    # title=None
    ticks = np.arange(0,25)
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], encoded_imgs[:, 2], c=label, cmap=cMap) #plt.cm.get_cmap('jet', 25)
    # p = ax.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], encoded_imgs[:, 2], c=label, cmap=plt.cm.get_cmap('jet', 25),edgecolors='none') #plt.cm.get_cmap('jet', 25)
    ax.view_init(azim=azim,elev=elev)
    # ax.set_title(title,fontsize=20)
    fig.colorbar(p, ticks=ticks,fraction=0.046, pad=0.04, aspect=20)
    fig.tight_layout()
    if save:
        fig.savefig(title + ".png")

def plt_scatter2d(encoded_imgs, label, azim, elev, title="latent_space", save=True):
    ticks = np.arange(0,25)
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    p = ax.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1],
                   c=label,cmap=cMap)#cmap=plt.cm.get_cmap('jet', 25)
    ax.set_title(title,fontsize=20)
    fig.colorbar(p, ticks=ticks,fraction=0.046, pad=0.04, aspect=20)
    fig.tight_layout()
    if save:
        fig.savefig(title + ".png")


def plot_tsne(eval_emb_all,val_samples,val_labels):   
    tsne2 = TSNE(n_components=3, learning_rate=100,perplexity=50)
    eval_samples_tsne = tsne2.fit_transform(val_samples)   
    plt_scatter(val_samples,val_labels, azim=48, elev=20, title="eval_samples")
    tsne1 = TSNE(n_components=3, learning_rate=100,perplexity=50)
    eval_emb_tsne = tsne1.fit_transform(eval_emb_all)   
    plt_scatter(eval_emb_tsne,val_labels, azim=48, elev=20, title="eval_emb_all")

    

    
    
    
    