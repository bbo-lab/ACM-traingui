import argparse
import os
import sys
import numpy as np

def main():
    # Parse inputs
    parser = argparse.ArgumentParser(description="ACM-traingui - Simple GUI to .")
    parser.add_argument('INPUT_PATH', type=str, help="Directory with detect job configuration")
    parser.add_argument('--labels', type=str, required=False, nargs='*', default=None)

    args = parser.parse_args()
    input_path = os.path.expanduser(args.INPUT_PATH)

    # Load config
    # TODO change config system, e.g. pass around a dictionary instead of importing the config everywhere, requiring the sys.path.insert
    sys.path.insert(0,input_path)
    print(input_path)

    if args.labels is None:
        from . import labeling_gui
        labeling_gui.main()
    else:
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
                labels = labels_new | labels # Python 3.9+
            except:
                print("Error loading")

        np.savez(cfg.filePath_labels, labels)
        print()
        print(f"{len(labels.keys())} frames labelled")

    return

if __name__ == '__main__':
    main()
