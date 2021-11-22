@author:fln
Part of this project is based on "Semi-Supervised Learning via Compact Latent Space Clustering". We change and apply it in IoT identification. Thanks for their codes.

## functions of dir
output/: store output files
data/: put the features file extracted from traffic

## environment:
```tensorflow 1.14
numpy 1.19.2
scikit-learn 0.23.2
python 3.7
```

## the change needed when test yourthings:
1. configs/iot/cfg_iot.py:
```
dataset_name = 'yourthings'
upsampling = False
```


## running method:
```
python run.py configs/iot/cfg_iot.py
``` 


