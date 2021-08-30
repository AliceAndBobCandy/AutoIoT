@author:fln
this project is based on "Semi-Supervised Learning via Compact Latent Space Clustering". We change and apply it in IoT identification. Part of the codes is the same as "Semi-Supervised Learning via Compact Latent Space Clustering"

add dir
output/:store output files
data/:the data downloaded


the change needed when test yourthings:
1. configs/iot/cfg_iot.py:
dataset_name = 'yourthings'
upsampling = True
num_val_samples = 5353
2. run_batch use yourthings new_devices_list_all
