# Provide the paths to the databases in the filesystem.


import inspect, os.path

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
pardir = os.path.dirname(path).replace("\\","/")
iot = pardir + '/data/iot/'