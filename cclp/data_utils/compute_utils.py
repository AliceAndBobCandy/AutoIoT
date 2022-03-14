from __future__ import absolute_import, division, print_function

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os
import math
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
import pandas as pd 
from scipy.stats import kstest,ks_2samp
from matplotlib import colors as c
from statsmodels.distributions.empirical_distribution import ECDF
import random
# from yellowbrick.cluster import KElbowVisualizer
import json
from sklearn.cluster import DBSCAN
# from bayes_opt import BayesianOptimization


font={
    'family':'Arial',
    'weight':'medium',
      'size':13
    }

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
device_list = [
    "Amazon Echo","Belkin Wemo motion sensor","Belkin Wemo switch",
    "Blipcare Blood Pressure meter","Dropcam","HP Printer",
    "iHome","Insteon Camera A","Insteon Camera B",
    "Light Bulbs LiFX Smart Bulb","Nest Dropcam","NEST Protect smoke alarm",
    "Netatmo weather station","Netatmo Welcome","PIX-STAR Photo-frame",
    "Samsung SmartCam","Smart Things","TP-Link Day Night Cloud camera",
    "TP-Link Smart plug","Triby Speaker","Withings Aura smart sleep sensor",
    "Withings Smart Baby Monitor","Withings Smart scale","Withings Baby Monitor",
    "Non-IoT Device"
]
device_list = ['Google OnHub', 'Samsung SmartThings Hub', 'Philips HUE Hub', 'Insteon Hub', 'Sonos', 
'Securifi Almond', 'Nest Camera', 'Belkin WeMo Motion Sensor', 'LIFX Virtual Bulb', 'Belkin WeMo Switch', 
'Amazon Echo', 'Belkin Netcam', 'Ring Doorbell', 'Roku TV', 'Roku 4', 'Amazon Fire TV', 'nVidia Shield', 
'Apple TV (4th Gen)', 'Belkin WeMo Link', 'Netgear Arlo Camera', 'D-Link DCS-5009L Camera', 
'Logitech Logi Circle', 'Canary', 'Piper NV [23]', 'Withings Home', 'WeMo Crockpot', 'MiCasaVerde VeraLite',
'Chinese Webcam', 'August Doorbell Cam', 'TP-Link WiFi Plug', 'Chamberlain myQ Garage Opener', 
'Logitech Harmony Hub', 'Caseta Wireless Hub', 'Google Home Mini', 'Google Home', 'Bose SoundTouch 10', 
'Harmon Kardon Invoke', 'Apple HomePod', 'Roomba', 'Samsung SmartTV', 'Koogeek Lightbulb', 
'TP-Link Smart WiFi LED Bulb', 'Wink 2 Hub', 'Nest Cam IQ', 'Nest Guard', 'Non-IoT devices']


cMap = ['red', 'coral', 'cornflowerblue', 'deepskyblue']
# compute softmax prob for each line, logits is the logits value of neurons. logits: np 
def eval_probability_distribution(logits):
    cols_ = logits.shape[1]
    prob_distri = None
    for logit_line in logits:
        prob_line = np.exp(logit_line - np.max(logit_line))/np.sum(np.exp(logit_line - np.max(logit_line)))
        if prob_distri is None:
            prob_distri = prob_line.reshape([-1,cols_])
        else:
            prob_distri = np.concatenate([prob_distri,prob_line.reshape([-1,cols_])],axis=0)
    return prob_distri

# observe prob_distri of devices, especially the new devices.
def observe_prob_distri(prob_distri, labels, new_devices_list, path,old_new_label_dict):
    font1 = {
        'family':'Arial',
        'weight':'medium',
        'size':20
    }
    if not os.path.exists(path):
        os.makedirs(path)
    # observe new_devices
    new_devices_list_ = [-1000 if item == 0 else -1*item for item in new_devices_list]
   
    new_old_label_dict = {}
    for key,value in old_new_label_dict.items():
        new_old_label_dict[value] = key
    for item in new_devices_list_:
        if -1000 in new_devices_list_:
            new_old_label_dict[-1000] = 0
        else:
            new_old_label_dict[item] = -1*item

    for idx, device_idx in enumerate(new_devices_list_):
        
        labels_bool = (labels == device_idx)
        device_prob_distri = prob_distri[labels_bool]
        device_name = device_list[int(new_old_label_dict[device_idx])]       
        max_inf = np.max(device_prob_distri,axis = 1)
        max_inf = np.around(max_inf,decimals = 2)
        max_count = Counter(max_inf)
        #
        plt.rcParams.update(plt.rcParamsDefault)
        plt.figure(figsize=(5.2,4.5),dpi=300)        
        plt.tick_params(labelsize=16)
        ax = plt.gca()       
        plt.xlim(0,1.1)       
        plt.bar(max_count.keys(),max_count.values(),width = 0.01)
        # plt.title(device_name,fontdict=font)
        plt.xlabel('Max probability',fontdict=font1)
        plt.ylabel('Count',fontdict=font1)
        # # 坐标轴刻度字体加粗
        # for i in range(len(ax.get_xticklabels())):
        #     ax.get_xticklabels()[i].set_fontweight("bold")
        # for i in range(len(ax.get_yticklabels())):
        #     ax.get_yticklabels()[i].set_fontweight("bold")
            # ax.get_yticklabels()[i].set_fontweight("bold")
        # fontproperties = {'family':'Arial','size': 6} 
        # ax.set_xticks(np.arange(0,1,0.1))
        # ax.set_xticklabels(ax.get_xticks(), fontproperties) 
        # ax.set_yticklabels(ax.get_yticks(), fontproperties) 
        plt.tight_layout()
        plt.savefig(path + '/{}.pdf'.format(device_name))
        
        plt.close()
    old_devices_list = [int(item) for item in labels if item >=0]
    old_devices_list = np.unique(old_devices_list)

    for idx, device_idx in enumerate(old_devices_list):
        # fig,ax = plt.subplots()
        labels_bool = (labels == device_idx)
        device_prob_distri = prob_distri[labels_bool]
        device_name = device_list[int(new_old_label_dict[device_idx])]
        max_inf = np.max(device_prob_distri,axis = 1)
        max_inf = np.around(max_inf,decimals = 1)
        max_count = Counter(max_inf)
        # ax.set_xticks([0,1,0.1])
        plt.figure(figsize=(5.2,4.5),dpi=300)
        ax = plt.gca() 
        plt.tick_params(labelsize=16)
        plt.xlim(0,1.1)
        plt.bar(max_count.keys(),max_count.values(),width = 0.01)
        # plt.title(device_name,fontdict=font)
        plt.xlabel('Max probability',fontdict=font1)
        plt.ylabel('Count',fontdict=font1)
        # for i in range(len(ax.get_xticklabels())):
        #     ax.get_xticklabels()[i].set_fontweight("bold")
        # for i in range(len(ax.get_yticklabels())):
        #     ax.get_yticklabels()[i].set_fontweight("bold")
        plt.tight_layout()
        plt.savefig(path + '/{}.pdf'.format(device_name))
        
        # plt.show()
        print(path)
        plt.close() 
    return
    # for device_idx,maxs in device_max_dict.items():
    #     max_dict = Counter(maxs)    
    # observe old_devices

# return a numpy list containing flag (0:old,1:new_devices,2:not decided)
def get_new_devices_flag(pred_logits,labels,threshold):
    # labels > 0 corresponding to old data, else decide its flag by threshold  
    new_device_flags = []
    cols_ = pred_logits.shape[1]
    for pred_logit,label in zip(pred_logits,labels):
        if label > 0:
            new_device_flags.append(0)
        else:
            M = np.max(pred_logit)
            max_prob = 1/np.sum(np.exp(pred_logit - M))
            if max_prob < threshold:
                new_device_flags.append(1)
            else:
                new_device_flags.append(2)
    # new_device_flags = np.array(new_device_flags).reshape([-1,1])
    return new_device_flags

def get_max_probs(pred_logits):
    result = []
    for pred_logits_ele in pred_logits:
        M = np.max(pred_logits_ele)
        max_prob = 1/np.sum(np.exp(pred_logits_ele-M))
        result.append(round(max_prob,2))
    return result
def get_threshold(label_prob_dict):
    result = []
    for label,values in label_prob_dict.items():
        if len(values) > 0:
            theta_cur = np.percentile(list(values),1)
            result.append(theta_cur)
    return max(result)

# type=1:pca, type=2:t-sne, type=3:VAE
def dimension_reduction(data,type):
    if type == 1:
        pca = PCA(n_components=0.95)
        reduced_data = pca.fit_transform(data)
        print("dimension of reduced data:{}".format(len(reduced_data[0])))
        if len(reduced_data[0]) == 1:
            print('the dimension reduced to 1, it should be at least 2')
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(data)
    elif type == 2:
        tsne = TSNE(n_components=3) #,learning_rate=100,perplexity=50
        reduced_data = tsne.fit_transform(data)
    else:
        pass
    return reduced_data

def get_optimal_k(sses,path=None):
    slopes = []
    for idx in range(len(sses)-1):
        slope = sses[idx+1]-sses[idx]
        slopes.append(slope)
    if path is not None:
        X = range(1,len(slopes)+1)
        plt.xlabel('sector i:i+1')
        plt.ylabel('slope')
        
        plt.plot(X,slopes,'o-',color = 'blue',label='slope')
        plt.savefig(path + 'slope.png')


    increases = []
    for idx in range(len(slopes)-1):
        increase = slopes[idx+1]-slopes[idx]
        increases.append(increase)
    # for idx in range(len(increases)):
    #     if increases[idx] < 0.1:
    #         print("{} is the optimal K".format(idx+1))
            # return idx+1
    if path is not None:
        X = range(1,len(increases)+1)
        plt.xlabel('slope i:i+1',fontdict=font)
        plt.ylabel('slope increase',fontdict=font)
        plt.plot(X,increases,'o-',color = 'orange',label = 'increase of slope')
        plt.savefig(path + 'increase.png')

    increases_delta = []
    for idx in range(len(increases_delta)-1):
        delta = increases[idx+1] - increases[idx]
        increases_delta.append(delta)
    # print(increases_delta)
    if path is not None:
        X = range(1,len(increases_delta)+1)
        plt.xlabel('sector i:i+1',fontdict=font)
        plt.ylabel('increase_delta',fontdict=font)
        plt.plot(X,increases_delta,'o-')
        plt.savefig(path + 'increase_delta.png',color='red',label='delta of increase')

    first = 0
    for idx in range(len(increases_delta)-1):
        if increases_delta[idx] > 0 and increases_delta[idx] > 0.3 and increases_delta[idx + 1] < 0:
            first = idx
            break
    print("{} is the optimal K".format(first + 4))

def get_optimal_k_from_slope(slopes,cnn_type=None,use_cnn_layer_for_cluster=False):
    print('use_cnn_layer_for_cluster',use_cnn_layer_for_cluster)
    if use_cnn_layer_for_cluster is False:
        # threshold = -100
        threshold = -115.9
    else:
        threshold = -300
    # print('threshold',threshold)
    # find the first i that |slopes[i]|>100 & |slopes[i+1]|<100
    result = 0
    find_flag = False
    for i in range(len(slopes)-1):
        if slopes[i] < threshold and slopes[i+1] > threshold:
            result = i
            find_flag = True
            break
    if find_flag is True:
        return result + 2
    else:
        return 1

def train_test_split(data,ratio):
    train = None
    test = None
    label_set = np.unique(data['label'])
    for label in label_set:
        data_c = data[data['label']==label]
        train_size = np.max([math.floor(ratio*data_c.shape[0]),1])
        train_c = data_c[:train_size]
        test_c = data_c[train_size:]
        if train is None:
            train = train_c
        else:
            train = pd.concat([train,train_c],axis=0)
        if test is None:
            test = test_c
        else:
            test = pd.concat([test,test_c],axis=0)
    return train,test

# compute probability distribution according to logits, return a list
def compute_probs_dist_through_logits(logits_all):
    result = []
    for logits in logits_all:
        M = np.max(logits)
        prob_iot = np.exp(logits[0]-M)/np.sum(np.exp(logits-M))
        result.append([prob_iot,1-prob_iot])
    return result

# dispose cnn emb, cnn_emb: {label:'c1_1':[],'c1_2':[],...'c3_2':[],'flatten':[]}, after dispose, each instance should corresponse to 1-dim vector
def dispose_cnn_emb(cnn_emb,merge_cnn_layer_type='sum'):
    cnn_layer_types = ['c1_1','c1_2','c2_1','c2_2','c3_1','c3_2','flatten']
    # cnn_layer_types = ['c3_1']
    data_compressed = {type:None for type in cnn_layer_types}
    for label in cnn_emb.keys():
        if label >= 0:
            continue
        data_c = cnn_emb[label]
        for cnn_layer in cnn_layer_types:
            data_c_cnn_type = np.array(data_c[cnn_layer]) # one type layer tensor, format: [num_instance,channels,1,length]
            if cnn_layer != 'flatten':
                data_c_cnn_type = np.squeeze(data_c_cnn_type,axis=2)
                if merge_cnn_layer_type == 'sum':
                    data_c_cnn_type = np.sum(data_c_cnn_type,axis=1) # [num_instance,length]
                else:
                    data_c_cnn_type = np.mean(data_c_cnn_type,axis=1) # [num_instance,length]
            if data_compressed[cnn_layer] is None:
                data_compressed[cnn_layer] = data_c_cnn_type
            else:
                data_compressed[cnn_layer] = np.concatenate([data_compressed[cnn_layer],data_c_cnn_type],axis=0)
    return data_compressed


def rf_cv(epsilon, minimum_samples):
    """
    :param epsilon: 两个点之间的最大距离，就是阈值
    :param minimum_samples: 作为核心点的一个点在一个邻域内的样本数量，就是成为一个类的最少的点的数量
    """
    # db = DBSCAN(eps=epsilon, min_samples=int(minimum_samples), leaf_size=leaf_size).fit(data)
    db = DBSCAN(eps=epsilon, min_samples=int(minimum_samples)).fit(data_bo)
    if db.labels_.max() == -1 or db.labels_.max() == 0 or db.labels_.max() == db.labels_.shape[0] - 1:
        return -1
    score = silhouette_score(data_bo, db.labels_, metric='euclidean')
    return score


data_bo = None
# return relabeled data
def relabel_data(data_X,K,path,new_labels,data,type=1,cnn_type=None,merge_cnn_layer_method=' ',use_cnn_layer_for_cluster=False,logger=None):
    font1={
    'family':'Arial',
    'weight':'medium',
      'size':18
    }
    if type == 1: # elbow + kmeans
        SSE = []  # 存放每次结果的误差平方和
        Scores = []
        for k in range(1,K):
            estimator = KMeans(n_clusters=k)  # 构造聚类器
            estimator.fit(data_X)
            # SSE法
            sse_value = sum(np.min(cdist(data_X,estimator.cluster_centers_,'euclidean'),axis=1))
            # # SSE.append(estimator.inertia_)
            SSE.append(sse_value)
            # 轮廓系数法
            # Scores.append(silhouette_score(data_X,estimator.labels_,metric='euclidean'))

        X = range(1,K)
        path_ = path.split('sse')[0] + '/'
        plt.figure(figsize=(5,4),dpi=300)
        plt.tick_params(labelsize=14)
        plt.xlabel('Number of cluser: k',fontdict=font1)
        plt.ylabel('Minimal square error',fontdict=font1)
        plt.plot(X,SSE,label='SSE',markersize=5,marker='o')
        
        plt.grid(ls='--')
        # plt.ylabel('silhouette_score')
        # plt.plot(X,Scores,'o-')
        plt.vlines(2,0,400,colors='black',linestyles='dashed') # changed
        plt.tight_layout()
        plt.savefig(path_ + 'elbow_{}_{}.pdf'.format(cnn_type,merge_cnn_layer_method))
        # plt.show()

        # find optimal k
        # k = get_optimal_k(SSE,path_)
        slopes = []
        for idx in range(len(SSE)-1):
            slope = SSE[idx+1]-SSE[idx]
            slopes.append(slope)
        optimal_k = get_optimal_k_from_slope(slopes,cnn_type,use_cnn_layer_for_cluster)
        if path is not None:
            X = range(2,len(slopes)+2)
            plt.xlabel('sector i:i+1')
            plt.ylabel('slope')
            plt.plot(X,slopes,'o-',color = 'blue',label='slope')
            
            plt.savefig(path_ + 'slope_{}.png'.format(cnn_type))
            plt.close()
        # record sse and slope inf
        with open(path_ + 'sse_slope_inf.txt','a') as f:
            f.write('find optimal cluster num k: elbow\n')
            f.write('cnn_layer_type:{}\n'.format(cnn_type))
            f.write('sse(1-10):{}\n'.format(SSE))
            f.write('slope(1-10):{}\n'.format(slopes))
            f.close()
        # relabel the data according to kmeans
        new_labels = np.array(new_labels)
        estimator = KMeans(n_clusters=optimal_k)  # 构造聚类器
        estimator.fit(data_X)
        pred_labels = estimator.labels_ # new_labels
        pred_labels_set = list(np.unique(pred_labels))
        # find label map relationship of new_labels and kmeans labels
        if optimal_k == len(pred_labels_set):
            pred_new_label_dict = {} # store pred_label: new labels dict, it is the final res
            new_pred_label_dict = {}
            used_new_label = [] 
            new_label_to_attribute = []
            new_label_in_pred_label_ratio = {}
            pred_label_with_new_label_ratio = {}
            for pred_label in pred_labels_set:
                logger.info(f"******** pred_label:{pred_label}")
                pred_new_label_dict[pred_label] = []
                pred_label_indices = np.where(pred_labels==pred_label)
                new_label_corres = new_labels[pred_label_indices]
                if len(np.unique(new_label_corres)) > 1:
                    logger.info(f"***** pred label {pred_label} cover {np.unique(new_label_corres)} true labels")
                # compute the ratio in this cluster to the num of all new label instances
                for new_label in np.unique(new_label_corres):
                    num_this = np.sum(new_label_corres==new_label)
                    num_all = np.sum(new_labels == new_label)
                    logger.info(f"new label:{new_label}, num instances in this cluster/all new label instances:{num_this/num_all}")
                    if new_label not in new_label_in_pred_label_ratio.keys():
                        new_label_in_pred_label_ratio[new_label] = {}
                    new_label_in_pred_label_ratio[new_label][pred_label] = num_this/num_all
                    if pred_label not in pred_label_with_new_label_ratio.keys():
                        pred_label_with_new_label_ratio[pred_label] = {}
                    pred_label_with_new_label_ratio[pred_label][new_label] = num_this/num_all
                # put the max ratio new label into pred label
                items_ = pred_label_with_new_label_ratio[pred_label]
                if len(items_) == 1:
                    new_label_ = list(items_.keys())[0]
                    used_new_label.append(new_label_)
                    pred_new_label_dict[pred_label].append(new_label_)
                    assert new_label_ not in new_pred_label_dict.keys(),f"new label {new_label_} has been distributed to other cluster before"
                    new_pred_label_dict[new_label_] = pred_label
                    
                        
                    # pred_labels[pred_labels==pred_label] = new_label_
                    if new_label_ in new_label_to_attribute:
                        new_label_to_attribute.remove(new_label_)
                else: # find max ratio new_label_
                    items_ = sorted(items_.items(),key=lambda x:x[1],reverse=True)
                    keys_ = [items_[i][0] for i in range(len(items_))]
                    keys_ = [item for item in keys_ if item not in used_new_label]
                    used_new_label.append(keys_[0])
                    new_label_to_attribute += keys_[1:]
                    pred_new_label_dict[pred_label].append(keys_[0])
                    assert keys_[0] not in new_pred_label_dict.keys(),f"new label {keys_[0]} has been distributed to other cluster before"
                    new_pred_label_dict[keys_[0]] = pred_label
                        
                    # pred_labels[pred_labels==pred_label] = keys_[0]
                    if keys_[0] in new_label_to_attribute:
                        new_label_to_attribute.remove(keys_[0])
                
            # dispose left new labels
            for new_label_ in new_label_to_attribute:
                items_ = new_label_in_pred_label_ratio[new_label_]
                items_ = sorted(items_.items(),key=lambda x:x[1],reverse=True)
                key = items_[0][0]
                pred_new_label_dict[key].append(new_label_)
                assert new_label_ not in new_pred_label_dict.keys(),f"new label {keys_[0]} has been distributed to other cluster before"
                new_pred_label_dict[new_label_] = key
                    
            logger.info(f'pred_label_new_label_dict:{pred_new_label_dict}')

                # corres_label = max(list(new_label_corres),key=list(new_label_corres).count)
                # pred_new_label_dict[pred_label] = corres_label
                # pred_labels[pred_labels==pred_label]=corres_label

            print('============= cnn type:{}==============='.format(cnn_type))
            # compute the label prediction accracy, ==========================!!!!!!!!!!!!!!!
            # for true_label in np.unique(new_labels):
            #     true_label_indices = new_labels==true_label
            #     true_num_c = np.sum(true_label_indices)
            #     print('true_label:{},num:{}'.format(true_label,np.sum(true_label_indices)))
            #     true_pred_label_indices = true_label_indices & (pred_labels == true_label)
            #     true_pred_num_c = np.sum(true_pred_label_indices)
            #     print('true_pred_label:{},num:{},ratio:{}'.format(true_label,np.sum(true_pred_num_c),true_pred_num_c*1.0/true_num_c))

            # construct new pred labels
            new_pred_labels = pred_labels
            for true_label in np.unique(new_labels):
                pred_ = new_pred_label_dict[true_label]
                marked_label = pred_new_label_dict[pred_][0]
                true_label_bool = (new_labels == true_label)
                pred_labels[true_label_bool] = marked_label


            # for pred_label,new_label in pred_new_label_dict.items():                    
            #     new_pred_labels[new_pred_labels==pred_label] = new_label[0]


            # concat new label and data_X, then save the data
            labels_ = new_pred_labels.reshape([-1,1])
            data_r = np.concatenate([labels_,data],axis=1)
            print("optimal_k is {}".format(optimal_k))
            return optimal_k, pd.DataFrame(data_r)
        else:
            return optimal_k, None
    # elif type == 1.1: # elbow contained in yellowbrick
    #     model = KMeans()
    #     visualizer = KElbowVisualizer(model,k=(1,K+1))
    #     visualizer.fit(data_X,timings=False)
    #     visualizer.show()

    elif type == 2: # BIC + kmeans
        BIC = []
        SSE = []
        for k in range(2,K):
            estimator = KMeans(n_clusters=k)
            estimator.fit(data_X)
            sse_value = math.sqrt(sum(np.min(cdist(data_X,estimator.cluster_centers_,'euclidean'),axis=1)))/data_X.shape[0]
            bic = k*math.log(data_X.shape[0]) + data_X.shape[0] * math.log(math.sqrt(sse_value))
            BIC.append(bic)
        delta_BIC = []
        for i in range(len(BIC)-1):
            delta_BIC.append(BIC[i+1]-BIC[i])
        return BIC.index(min(BIC)) + 2
        
    elif type == 3: # AIC + Kmeans
        AIC = []
        SSE = []
        for k in range(2,K):
            estimator = KMeans(n_clusters=k)
            estimator.fit(data_X)
            sse_value = sum(np.min(cdist(data_X,estimator.cluster_centers_,'euclidean'),axis=1))/data_X.shape[0]
            aic = k*2 + data_X.shape[0] * math.log(sse_value)
            AIC.append(aic)
        delta_AIC = []
        for i in range(len(AIC)-1):
            delta_AIC.append(AIC[i+1]-AIC[i])
        return AIC.index(min(AIC)) + 2
    
    # elif type == 4: # dbscan + sihouette_score + bayesian optimization
    #     global data_bo
    #     data_bo = data_X
    #     pbounds = {'epsilon': (0.00000001, 0.1),
    #            'minimum_samples': (2, 10),
    #         #    'leaf_size': (20, 40),
    #            }
    #     optimizer = BayesianOptimization(
    #         f=rf_cv,  # 黑盒目标函数
    #         pbounds=pbounds,  # 取值空间
    #         verbose=2,  # verbose = 2 时打印全部，verbose = 1 时打印运行中发现的最大值，verbose = 0 将什么都不打印
    #         random_state=1,
    #     )

    #     # init_points: 随机搜索的步数, n_iter: # 执行贝叶斯优化迭代次数
    #     optimizer.maximize(init_points=1, n_iter=1)
    #     # 得到最优的参数
    #     optimize_params = optimizer.max['params']
    #     dbscan_model = DBSCAN(eps=optimize_params['epsilon'], min_samples=optimize_params['minimum_samples'])
    #     dbscan_model.fit(data_X)
    #     pred_labels = dbscan_model.labels_  # 得到优化后的标签
    #     optimal_k = len(np.unique(pred_labels))
    #     pred_labels_set = list(np.unique(pred_labels))
    #     # find label map relationship of new_labels and kmeans labels
    #     new_labels = np.array(new_labels)
    #     if optimal_k == len(np.unique(new_labels)):
    #         pred_new_label_dict = {}
    #         for pred_label in pred_labels_set:
    #             pred_label_indices = np.where(pred_labels==pred_label)
    #             new_label_corres = new_labels[pred_label_indices]
    #             corres_label = max(list(new_label_corres),key=list(new_label_corres).count)
    #             pred_new_label_dict[pred_label] = corres_label
    #             # pred_labels = pred_labels.replace(pred_label,corres_label)
    #             pred_labels[pred_labels==pred_label]=corres_label
    #         # concat new label and data_X, then save the data
    #         labels_ = pred_labels.reshape([-1,1])
    #         data_r = np.concatenate([labels_,data],axis=1)
    #         print("optimal_k is {}".format(optimal_k))
    #         return optimal_k, pd.DataFrame(data_r)
    #     else:
    #         return optimal_k, None


    elif type == 5: # silhouette_score + kmeans
        Scores = []
        for k in range(2,K):
            estimator = KMeans(n_clusters=k)  # 构造聚类器
            estimator.fit(data_X)
            # 轮廓系数法
            Scores.append(silhouette_score(data_X,estimator.labels_,metric='euclidean'))

        X = range(2,K)
        path_ = path.split('sse')[0] + '/'
        plt.figure(figsize=(6,5),dpi=300)
        plt.tick_params(labelsize=9)
        plt.xlabel('Number of cluser: k',fontdict=font)
        plt.ylabel('Silhouette score',fontdict=font)
        plt.plot(X,Scores,label='silhouette_score',markersize=5,marker='o')
        
        plt.grid(ls='--')
        plt.savefig(path_ + 'silhouette_score_{}_{}.png'.format(cnn_type,merge_cnn_layer_method))
        plt.close()
        # find optimal k
        optimal_k = Scores.index(max(Scores)) + 2
        
        # relabel the data according to kmeans
        new_labels = np.array(new_labels)
        estimator = KMeans(n_clusters=optimal_k)  # 构造聚类器
        estimator.fit(data_X)
        pred_labels = estimator.labels_ # new_labels
        pred_labels_set = list(np.unique(pred_labels))
        # find label map relationship of new_labels and kmeans labels
        if optimal_k == len(np.unique(new_labels)):
            pred_new_label_dict = {}
            for pred_label in pred_labels_set:
                pred_label_indices = np.where(pred_labels==pred_label)
                new_label_corres = new_labels[pred_label_indices]
                corres_label = max(list(new_label_corres),key=list(new_label_corres).count)
                pred_new_label_dict[pred_label] = corres_label
                # pred_labels = pred_labels.replace(pred_label,corres_label)
                pred_labels[pred_labels==pred_label]=corres_label
            # concat new label and data_X, then save the data
            labels_ = pred_labels.reshape([-1,1])
            data_r = np.concatenate([labels_,data],axis=1)
            print("optimal_k is {}".format(optimal_k))
            return optimal_k, pd.DataFrame(data_r)
        else:
            return optimal_k, None



# return cluster num of data, data:pd, type=1:kmeans, type=2:dbscan, type=3:KNN, filtered_cnn_emb: use the emb of cnn layer for cluster
# filtered_cnn_emb: {label:{'c1_1':[],'c1_2':[],...}
def get_new_type_num_from_cluster(data1,type,path=None,filtered_cnn_emb=None,merge_cnn_layer_method=' ',use_cnn_layer_for_cluster=False,logger=None):
    data_o = None
    new_labels = []
    for label in data1.keys():
        if label < 0: # new devices
            if data_o is None:
                data_o = data1[label]
                # data = data[:instance_num]
            else:
                if len(data1[label]) > 0:
                    data1_c = data1[label]
                    # data1_c = data1_c[:instance_num]
                    data_o = np.concatenate([data_o,data1_c],axis=0)
            new_labels += [label]*len(data1[label])
            # new_labels += [label]*instance_num
    K = 11
    # data = data.values[:,1:]
    #============================== use cnn layer emb ===============================
    if not filtered_cnn_emb is None:
        types = ['c1_1','c1_2','c2_1','c2_2','c3_1','c3_2','flatten']
        # types = ['c3_1']
        optimal_k_cnn_layers = {type:0 for type in types}
        data_r_cnn_layers = {type:None for type in types}

        data_multi_cnn_layer = dispose_cnn_emb(filtered_cnn_emb,merge_cnn_layer_method)
        for key,data in data_multi_cnn_layer.items():
            if key == 'c3_1':
                data_X = dimension_reduction(data,1)
                optimal_k, data_r = relabel_data(data_X,K,path,new_labels,data_o,type,cnn_type=key,merge_cnn_layer_method=merge_cnn_layer_method,use_cnn_layer_for_cluster=use_cnn_layer_for_cluster)
                optimal_k_cnn_layers[key] = optimal_k
                data_r_cnn_layers[key] = data_r
        return optimal_k_cnn_layers, data_r_cnn_layers
    else:
        data_X = dimension_reduction(data_o,1)
        optimal_k, data_r = relabel_data(data_X,K,path,new_labels,data_o,type,None,' ',use_cnn_layer_for_cluster,logger)
        return optimal_k, data_r
    
    
    
#-------------judge new devices by multi instances------------
def get_delta_s(seq):
    a = 1
    b = 0.2
    theta = 0.7
    Y = sum(seq)/len(seq)
    if Y > theta:       
        return a*math.pow(Y+b,len(seq))
    else:
        return a*math.pow((1-Y+b),len(seq))*(-1)

# def compute_score(seq):
#     result = s0
#     for idx in range(len(seq)):
#         delta = delta_s(seq[:idx+1])
#         # print(delta)
#         result += delta
#         print(result)
#     return result
def rearrange_labels(old_data_train,new_data_train,old_data_test,new_data_test,path):

        # 
        new_data_train_labels = np.unique(new_data_train['label'])
        print(f'new data train labels num:{len(new_data_train_labels)}')
        old_data_train_labels = np.unique(old_data_train['label'])
        print(f'old data train labels num:{len(old_data_train_labels)}')

        new_rearange_label_dict = {}   
        new_labels = np.unique(new_data_train['label']) # fmodify
        old_labels = np.unique(old_data_train['label'])
        label_non_iot_cur = np.max(old_labels)
        label_non_iot_new = len(new_labels) + np.max(old_labels)
        # self.old_label_new_label_dict[label_non_iot_cur] = label_non_iot_new
        old_data_train['label'] = old_data_train['label'].replace(label_non_iot_cur,label_non_iot_new)
        old_data_test['label'] = old_data_test['label'].replace(label_non_iot_cur,label_non_iot_new)
        blank_list = list(range(int(label_non_iot_cur),int(label_non_iot_new)))
        for blank, new_device_label in zip(blank_list,new_labels):
            # self.old_label_new_label_dict[new_device_label] = blank
            new_data_train['label'] = new_data_train['label'].replace(new_device_label,blank)
            new_data_test['label'] = new_data_test['label'].replace(new_device_label,blank)
            new_rearange_label_dict[new_device_label] = blank
        data_train = pd.concat([old_data_train,new_data_train],axis=0)
        data_test = pd.concat([old_data_test,new_data_test],axis=0)
        # save dict
        with open(path + '/new_rearange_label_dict.json','w') as f:
            json.dump(new_rearange_label_dict,f)
            f.close()

        return data_train, data_test


# compute KS test result for 2 instances and compare with D table, cur_value is the tested instances, known_values is used to compare.
def compute_KS(cur_values,known_values,type = 0.05): 
    d_statistic,critical = 0,0
    if type == 0.05:
        critical_value = {}
        criticals_5 = [0.975,0.842,0.708,0.624,0.565,0.521,0.486,0.457,0.432,0.410,0.391,0.375,0.361,0.349,0.338,0.328,0.318,0.309,0.301,0.294,0.27,0.24,0.23]
        for i in range(1,21):
            critical_value[i] = criticals_5[i-1]
        for i in range(21,26):
            critical_value[i] = 0.27
        for i in range(26,31):
            critical_value[i] = 0.24
        for i in range(30,36):
            critical_value[i] = 0.23
        for i in range(36,1000):
            critical_value[i] = round(1.36/math.sqrt(i),2)
        d_statistic, pvalue = ks_2samp(np.array(cur_values),np.array(known_values))  
        critical = critical_value[len(cur_values)]
        return [d_statistic,critical]
    elif type == 0.01:
        critical_value = {}
        criticals_1 = [0.995,0.929,0.828,0.773,0.669,0.618,0.577,0.543,0.514,0.490,0.468,0.450,0.433,0.418,0.404,0.392,0.381,0.371,0.363,0.356]
        for i in range(1,21):
            critical_value[i] = criticals_1[i-1]
        for i in range(21,26):
            critical_value[i] = 0.32
        for i in range(26,31):
            critical_value[i] = 0.29
        for i in range(30,36):
            critical_value[i] = 0.27
        for i in range(36,1000):
            critical_value[i] = round(1.63/math.sqrt(i),2)
        d_statistic, pvalue = ks_2samp(np.array(cur_values),np.array(known_values))  
        critical = critical_value[len(cur_values)]
        return [d_statistic,critical]

# cMap = ['red', 'coral', 'gold', 'c']
def plot_cdf(known_probs,max_prob_label_dict,new_old_label_dict,path):
    font1={
    'family':'Arial',
    'weight':'medium',
      'size':16
    }
    label_set = [-1000,-1,0]

    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(5,4.5),dpi=300)
    plt.tick_params(labelsize=14)
    plt.xlim(0,1.1)
    plt.ylim(0,1.1)
    ecdf_known = ECDF(known_probs)
    plt.plot(ecdf_known.x,ecdf_known.y,'--',linewidth=1.5,color='orange',label='Known types of devices',alpha=0.8)
    plt.xlabel('Max probability value',fontdict=font1)
    plt.ylabel('Probability',fontdict=font1)
    # plt.savefig(path + '/{}.jpg'.format(device_name))
    # plt.show()
    for idx,label in enumerate(label_set):
        ecdf = ECDF(max_prob_label_dict[label])
        ecdf_x = list(ecdf.x)
        ecdf_y = list(ecdf.y)
        # padding the lacked value
        max_value = max(ecdf(ecdf.x))
        for i in range(20):
            added_value = max_value + (1-max_value)*np.random.random()
            ecdf_x.append(added_value)
            ecdf_y.append(1)
        min_value = min(ecdf(ecdf.x))
        for i in range(20):
            added_value = min_value*np.random.random()
            ecdf_x.insert(0,added_value)
            ecdf_y.insert(0,0)
        plt.plot(ecdf_x,ecdf_y,linewidth=1.5,color=cMap[idx],label=device_list[new_old_label_dict[label]],alpha=0.8)
    plt.legend(bbox_to_anchor=(0.26, 1.02), loc=3, borderaxespad=0, fontsize=12) #prop={'size':6},
    # plt.legend(loc=2, fontsize=12) 
    plt.tight_layout()
    plt.savefig(path)


