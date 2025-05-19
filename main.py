import sys
from os.path import abspath
import time
from imp import reload
from conf import OptionConf
from loader import load_data
import numpy as np

reload(sys)

def executeCMD(config=None,method=None,from_disk=False):
    if config == None:
        print('Invalid configuration')
        exit(-1)

    if method == None:
        print('Invalid model')
        exit(-1)

    exec('from '+method+' import '+method)
    
    conf = OptionConf(config)
    train_data,test_data = load_data(conf,from_disk=from_disk)
    
    if conf.contains('social'):
        my_method = eval(method+'(conf,train_data,test_data,conf["social"])')
    else:
        my_method = eval(method+'(conf,train_data,test_data)')

    start_time = time.time()
    my_method.execute()
    end_time = time.time()
    print("Running time: %f s" % (end_time - start_time))

if __name__ == '__main__':
    method_name = 'XSimGCL'
    config_file = 'submission/sub/XSimGCL.conf'
    executeCMD(config=config_file, method=method_name, from_disk=True)

