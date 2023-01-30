from pathlib import Path
import os  # Required for eval below :(


def load_cfg(path):
    with open(path, 'r') as cfg_file:
        configtxt = cfg_file.read()
        cfg = eval(configtxt)  # this is ugly since eval is used (make sure only trusted strings are evaluated)
    return cfg


def save_cfg(path: Path, cfg):
    with open(path, 'w') as file_cfg:
        file_cfg.write('{\n')
        for key in cfg.keys():
            if isinstance(cfg[key], str):
                line = f"  '{key}': '{cfg[key]}',\n"
            else:
                line = f"  '{key}': {cfg[key]},\n"
            file_cfg.write(line)
        file_cfg.write('}\n')
    return cfg
