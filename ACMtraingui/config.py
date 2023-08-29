import os
import re

def load_cfg(path):
    cfg_file = open(path, 'r')
    configtxt = cfg_file.read()
    datadir = datadir = re.sub(r'\\', '/', os.path.dirname(os.path.abspath(path)))+'/../../../data'
    cfg = eval(configtxt) # this is ugly since eval is used (make sure only trusted strings are evaluated)
    cfg_file.close()
    #sys.path.insert(0,os.path.dirname(path))
    #from labeling_gui_cfg import cfg
    #sys.path.remove(os.path.dirname(path))
    return cfg


def save_cfg(path, cfg):
    file_cfg = open(path, 'w')
    file_cfg.write('{\n')
    for key in cfg.keys():
        if isinstance(cfg[key], str):
            file_cfg.write('\'' + key + '\'' + ': ' + '\'' + str(cfg[key]) + '\'' + ',\n')
        else:
            file_cfg.write('\'' + key + '\'' + ': ' + str(cfg[key]) + ',\n')
    file_cfg.write('}\n')
    file_cfg.close()
    return cfg
