
import os
import glob
from configs.iot.cfg_iot import out_path


path = out_path + 'trainTf'
folders = glob.glob(path + '/*')
folders = sorted(folders)
for idx,folder in enumerate(folders):
    if idx < 2:
        continue
    if os.path.exists(folder + '/checkpoint'):
        f = open(folder + '/checkpoint','r')
        res = ""
        line = f.readline()
        while(line):
            front,_,end = line.split(':')
            end_last = end.split("\\")[-1]
            line_new = front + ': "' + end_last
            res = res + line_new
            line = f.readline()
        f.close()
        g = open(folder + '/checkpoint','w')
        g.write(res)
        g.close()

