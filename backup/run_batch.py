# this is a running code for batch of tests
# author Linna Fan
import os
import time
from cclp.data_utils.log_utils import init_log
from configs.iot.cfg_iot import out_path

dataset = 'unsw' # unsw/yt
logger = init_log(out_path + 'mylogs/test_known_new_iden_for_n_min_max.log','w')
# os.chmod(out_path + 'mylogs', 0o777)
# unsw
new_devices_list_all_tmc = ['0,1','2,4','5,6','7,9','12,13','14,15','16,17','18,19','20,21','1,17,18','9,19,20','2,14,16','0,4,21','1,7,9,18','12,15,16,19','5,14,17,20','2,4,13,21']
# new_devices_list_all_tmc = ['2,14,16','0,4,21','1,7,9,18','12,15,16,19','5,14,17,20','2,4,13,21']
# yourthings
new_devices_list_all_yt = ['0,1','2,3','4,5','6,7','8,9','10,11','12,13','14,15','16,17','18,19','20,21','22,23','24,25','26,27',
'28,29','30,31','32,33','34,35','36,37','38,39','40,41','42,43','8,27,36','6,25,35','11,38,40','2,20,39','8,22,32,38','29,36,39,42',
'0,21,30,41','2,12,23,26']

if dataset == 'yt':
    print(len(new_devices_list_all_yt))

#-----------------------------------------------------------------------------
if dataset == 'unsw':
    for idx,new_devices_list in enumerate(new_devices_list_all_tmc):  
        # # train, session_type = 'train'
        # time11 = time.time()
        # try:
        #     print('tmc start training {}'.format(new_devices_list))
        #     os.system('python run.py -session_type {} -upsampling {} -dataset_name {} -merge_cnn_layer_methods {} -use_cnn_layer_for_cluster {} -new_devices_list {} configs/iot/cfg_iot.py'.format('train',True,'tmc','sum',True,new_devices_list))
        # except Exception as e:
        #     print('there is something wrong when training {}'.format(new_devices_list))
        # time12 = time.time()
        # with open('time_information.txt','a') as f:
        #     f.write('tmc {}, whole training time of {}:{} (sum aggregate)\n'.format(idx,new_devices_list,time12-time11))
        #     f.close()


        # test
        time21 = time.time()
        try:
            print('tmc start new devices disposing {}'.format(new_devices_list))
            os.system('python run.py configs/iot/cfg_iot.py -session_type {} -upsampling {} -dataset_name {}  -new_devices_list {}'.format('test','1','tmc',new_devices_list))
        except Exception as e:
            print('there is something wrong when testing {}'.format(new_devices_list))
        time22 = time.time()
        with open('time_information.txt','a') as f:
            f.write('tmc {}, whole new devices identification cost time of {}:{} (sum aggregate)\n'.format(idx,new_devices_list,time22-time21))
            f.close()

else:
    for idx,new_devices_list in enumerate(new_devices_list_all_yt):  
        # train, session_type = 'train'
        # time11 = time.time()
        # try:
        #     print('yt start training {}'.format(new_devices_list))
        #     os.system('python run.py -session_type {} -upsampling {} -dataset_name {} -merge_cnn_layer_methods {} -use_cnn_layer_for_cluster {} -new_devices_list {} configs/iot/cfg_iot.py'.format('train','1','yourthings','sum','1',new_devices_list))
        # except Exception as e:
        #     print('there is something wrong when training {}'.format(new_devices_list))
        # time12 = time.time()
        # with open('time_information.txt','a') as f:
        #     f.write('yt {}, whole training time of {}:{} (sum aggregate)\n'.format(idx,new_devices_list,time12-time11))
        #     f.close()


        # test
        time21 = time.time()
        try:
            logger.info(f'***** group:{idx+1},new_devices:{new_devices_list}')
            # os.system('python run.py configs/iot/cfg_iot.py -session_type {} -upsampling {} -dataset_name {} -merge_cnn_layer_methods {} -use_cnn_layer_for_cluster {} -new_devices_list {}'.format('test','0','yourthings','sum','1',new_devices_list))
            os.system('python run.py configs/iot/cfg_iot.py -session_type {} -upsampling {} -dataset_name {} -new_devices_list {}'.format('test','0','yourthings',new_devices_list))
        except Exception as e:
            logger.info('there is something wrong when testing group {}, new device list {}'.format(idx,new_devices_list))
        time22 = time.time()
        with open('time_information.txt','a') as f:
            logger.info('yt group {}, time_dur:{} \n'.format(idx,time22-time21))
            

