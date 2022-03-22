# ACMtraingui

GUI to label frames for training of ACM-dlcdetect, by Arne Monsees

## Installation
(You do not need to clone this repository.)
1. [Install Anaconda](https://docs.anaconda.com/anaconda/install/)
2. Start Anaconda Prompt (Windows) / terminal (linux) and navigate into repository directory
3. Create conda environment `conda env create -f https://raw.githubusercontent.com/bbo-lab/ACM-traingui/master/environment.yml`

## Update 
1. Start Anaconda Prompt (Windows) / terminal (linux) and navigate into repository directory
2. Update with `conda env update -f https://raw.githubusercontent.com/bbo-lab/ACM-traingui/master/environment.yml --prune`.

## Running
1. Start Anaconda Prompt (Windows) / terminal (linux) and navigate into repository directory
2. Switch to environment `conda activate bbo_acm-traingui`
3. Run with `python -m ACMtraingui [options ...]`

## Options
### Assistant mode
Run with `python -m ACMtraingui [base data directory]`.
This starts a GUI in drone mode, for the use by assistants with limited options to influence how the program runs and were it saves. This expects the following file structure:
```
[base data directory]/data/users/{user1,user2,...}/labeling_gui_cfg.py
[base data directory]/users/
```
{user1,user2,...} will be presented in a selection dialog on startup. Marking results will be placed in `[base data directory]/users/`

### Master mode
Run with `python -m ACMtraingui [configdir] --master`.
This starts a GUI in master mode. Only do this if you know what you are doing.

### Check mode
Run with 
```
python -m ACMtraingui [directory of labels.npz] --check [bbo_calibcam calibration npy] # to use specified npy or
python -m ACMtraingui [directory of labels.npz] --check [labeling_gui_cfg.py folder] # to use standardCalibrationFile from labeling_gui_cfg.py or
python -m ACMtraingui [directory of labels.npz] --check '-' # or
python -m ACMtraingui [directory of labels.npz] --check # to use backup labeling_gui_cfg.py in labels.npy folder. (Will often fail due to different paths between checker und labeler, as relative pathes are resolved, here).
```

This gives sorted text output of 3d and reprojections errors. Reporjection errors above 5-10px usually indicate errors in labeling and respective frames have to be checked.

### Join mode
Run with `python -m ACMtraingui [configdir of ACM-dlcdetect] --check [multiple directories containing labels.npz files] `.
This joins all marked labels in the labels.npz files into the labels.npz file in the dlcdetect configuration. Marked labels overwrite existing labels framewise.


## TODO
- Document config
- Document sketch file (2d sketch of animal. If not presented, 3d wireframe is shown instead)
- Document [model file](https://github.com/bbo-lab/ACM/blob/main/INPUTS.md#modelnpy)
