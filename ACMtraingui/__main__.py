import argparse
import os
import sys
import numpy as np

def main():
    # Parse inputs
    parser = argparse.ArgumentParser(description="ACM-traingui - Simple GUI to .")
    parser.add_argument('INPUT_PATH', type=str, help="Directory with detect job configuration")
    parser.add_argument('--labels', type=str, required=False, nargs='*', default=None, help="If given, merges labes.npz in given dirs into labels.npz file specified in INPUT_PATH config file")
    parser.add_argument('--check', type=str, required=False, nargs='?', default=None, const='-', help="Prints sorted list of square errors for labels in INPUT_PATH/labels.npz. Supply either calibration file, a path to a labeling_gui_cfg.py or '-'/nothing to load labeling_gui_cfg.py in directory of labels.npy")
    parser.add_argument('--master', required=False, action="store_true", help="Switches between master mode and worker mode")

    args = parser.parse_args()
    input_path = os.path.expanduser(args.INPUT_PATH)
    print(input_path)
    if args.labels is not None:
        # Load config
        sys.path.insert(0,input_path)
        import dlcdetectConfig as cfg

        if os.path.isfile(cfg.filePath_labels):
            labels = np.load(cfg.filePath_labels, allow_pickle=True)['arr_0'].item()
        else:
            labels = dict()

        for labelsdir in args.labels:
            print(labelsdir+"/labels.npz")
            try:
                labels_new = np.load(labelsdir+"/labels.npz", allow_pickle=True)['arr_0'].item()
                print(list(labels_new.keys()))
                print()
                labels = {**labels, **labels_new} # labels | labels_new # Python 3.9+
            except:
                print("Error loading")

        np.savez(cfg.filePath_labels, labels)
        print()
        print(f"{len(labels.keys())} frames labelled")
    elif args.check is not None:
        import calibcamlib

        if args.check == '-':
            from .config import load_cfg
            cfg = load_cfg(input_path+'/labeling_gui_cfg.py') # This will load a copy, might fail since paths are replaced
            calibfile = cfg['standardCalibrationFile']
        elif os.path.isdir(args.check): # This is supposed to be filled with the config directory
            from .config import load_cfg
            cfg = load_cfg(args.check+'/labeling_gui_cfg.py')
            calibfile = cfg['standardCalibrationFile']
        else: # This is supposed to be filled with the path of the calib file
            calibfile = args.check[0]

        cs = calibcamlib.Camerasystem.from_calibcam_file(calibfile)
        labels = np.load(input_path+"/labels.npz",allow_pickle=True)['arr_0'].item()

        frame = [];
        marker = [];
        dists = [];
        mdist = [];
        pdists = [];
        mpdist = [];
 
        for i in labels.keys():
            for m in labels[i].keys():
                (X,P,V) = cs.triangulate_3derr(labels[i][m][:,np.newaxis,:])
                ds = calibcamlib.helper.calc_3derr(X,P,V)[1]

                x = cs.project(X)
                pd = np.sum((labels[i][m][:,np.newaxis,:]-x)**2,axis=2).T[0]

                if np.any(~np.isnan(ds)):
                    frame.append(i)
                    marker.append(m)
                    dists.append(ds)
                    mdist.append(np.nanmax(ds))
                    pdists.append(pd)
                    mpdist.append(np.nanmax(pd))

        idx = np.argsort(np.asarray(mpdist))[::-1]
        mdists = np.asarray(mdist)[idx]
        frames = np.asarray(frame)[idx]
        markers = np.asarray(marker)[idx]
        distss = np.asarray(dists)[idx]
        pdistss = np.asarray(pdists)[idx]

        for i,md in enumerate(mdists):
            print(f'{frames[i]:6} | {markers[i]:>25} | {np.nanmax(distss[i]):5.2f} | {list(map("{:5.2f}".format,distss[i]))} | {np.nanmax(pdistss[i]):6.1f} | {list(map("{:7.2f}".format,pdistss[i]))}')

    elif args.master:
        from . import labeling_gui
        labeling_gui.main(master=True,configFile=input_path)
    else:
        from . import labeling_gui
        labeling_gui.main(master=False,drive=input_path)
    return

if __name__ == '__main__':
    main()
