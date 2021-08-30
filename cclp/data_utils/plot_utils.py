# author: Linna Fan

from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np

# List of device in the dataset, use to plot pictures later
# **THIS CELL DOES NOT PARTICIPATE THE TRAINING OF THE SYSTEM**
# device_list = [
#     "Amazon Echo [0]","Belkin wemo motion sensor [1]","Belkin Wemo switch [2]",
#     "Blipcare Blood Pressure meter [3]","Dropcam [4]","HP Printer [5]",
#     "iHome [6]","Insteon Camera [7]","Insteon Camera [8]",
#     "Light Bulbs LiFX Smart Bulb [9]","Nest Dropcam [10]","NEST Protect smoke alarm [11]",
#     "Netatmo weather station [12]","Netatmo Welcome [13]","PIX-STAR Photo-frame [14]",
#     "Samsung SmartCam [15]","Smart Things [16]","TP-Link Day Night Cloud camera [17]",
#     "TP-Link Smart plug [18]","Triby Speaker [19]","Withings Aura smart sleep sensor [20]",
#     "Withings Smart Baby Monitor [21]","Withings Smart scale [22]","Withings Baby Monitor [23]",
#     "Non-IoT Device [24]"
# ]
# device_list = range(46)
def plot_confusion_matrix(cm, normalize=True, title=None, cmap=plt.cm.Blues, save=False):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    device_list = list(range(cm.shape[0]))
    classes = device_list
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    plt.figure(figsize=(30, 30), dpi=300)
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 20)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    ax.figure.colorbar(im,fraction=0.046, pad=0.04, aspect=20)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.set_ylabel('True label',fontsize=18)
    ax.set_xlabel('Predicted label',fontsize=18)
    ax.set_title(title, fontsize=20)
    ttl = ax.title
    ttl.set_position([.5, 1.02]) #标题距离
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    # fig.tight_layout()
    
    if save:
        fig.savefig(title)
    return ax