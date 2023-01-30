import argparse
import os
import sys
import numpy as np
import re
from pathlib import Path


def main():
    # Parse inputs
    parser = argparse.ArgumentParser(description="ACM-traingui - Simple GUI to .")
    parser.add_argument('INPUT_PATH', type=str, help="Directory with detect job configuration")
    parser.add_argument('--labels', type=str, required=False, nargs='*', default=None,
                        help="If given, merges labes.npz in given dirs into labels.npz file specified in INPUT_PATH "
                             "config file")
    parser.add_argument('--merge', type=str, required=False, nargs='*', default=None,
                        help="If given, merges given labes.npz into labels.npz file specified in INPUT_PATH")
    parser.add_argument('--strict', required=False, action="store_true",
                        help="With --labels, merges only frames where frames were labeled in all cameras")
    parser.add_argument('--check', type=str, required=False, nargs='?', default=None, const='-',
                        help="Prints sorted list of square errors for labels in INPUT_PATH/labels.npz. Supply either "
                             "calibration file, a path to a labeling_gui_cfg.py or '-'/nothing to load "
                             "labeling_gui_cfg.py in directory of labels.npy")
    parser.add_argument('--master', required=False, action="store_true",
                        help="Switches between master mode and worker mode")

    args = parser.parse_args()
    input_path = os.path.expanduser(args.INPUT_PATH)
    print(input_path)
    if args.labels is not None:
        # Load config
        if os.path.isdir(input_path):
            sys.path.insert(0, input_path)
            import dlcdetectConfig as cfg
            labels_file = cfg.filePath_labels
        else:
            labels_file = input_path

        if os.path.isfile(labels_file):
            labels = np.load(labels_file, allow_pickle=True)['arr_0'].item()
        else:
            labels = dict()

        for labelsdir in args.labels:
            try:
                labels_new = np.load(labelsdir + "/labels.npz", allow_pickle=True)['arr_0'].item()
            except:
                print(f"Error loading {labelsdir + '/labels.npz'}")
                continue

            delframes = []
            if args.strict:
                for frameidx, labels_dict in labels_new.items():
                    camnans = np.all(np.isnan(np.concatenate(list(labels_dict.values()), 1)), axis=1)
                    if np.any(camnans):
                        delframes.append(frameidx)

                for frame_idx in delframes:
                    labels_new.pop(frame_idx, None)

            userframes = list(labels_new.keys())

            m = re.search('(?<=/pose/user/)[A-Za-z0-9_-]+', labelsdir)
            print()
            print(f'{m[0]}: {len(userframes)} - {userframes}')
            if len(delframes) > 0:
                print(f'Not considering {len(delframes)} frames due to incomplete marking: {delframes}')

            labels = {**labels_new, **labels}  # labels_new | labels # Python 3.9+

        np.savez(labels_file, labels)
        print()
        print(f"{len(labels.keys())} frames labelled")

    if args.merge is not None:
        labels_files = [Path(lf) for lf in args.merge]
        labeler = [lf.parent.parent.name for lf in labels_files]
        target_file = Path(input_path)

        labels = [np.load(lf, allow_pickle=True)["arr_0"][()] for lf in labels_files]

        for l, p in zip(labels, labeler):
            if "labeler" not in l:
                # add from path
                l["labeler"] = dict.fromkeys(l["fr_times"].keys(), labeler.index(p))
            else:
                # rewrite to global index list
                for frame_idx, pp in l["labeler"].items():
                    try:
                        p_idx = labeler.index(l["labeler_list"][pp])
                    except ValueError:
                        labeler.append(l["labeler_list"][pp])
                        p_idx = labeler.index(l["labeler_list"][pp])

                    l["labeler"][frame_idx] = p_idx

        if not target_file.is_file():
            target_labels = labels[0]
        else:
            target_labels = np.load(target_file, allow_pickle=True)["arr_0"][()]

        for l in labels[1:]:
            for frame_idx in l["fr_times"]:
                if frame_idx not in target_labels["fr_times"] or \
                        target_labels["fr_times"][frame_idx] < l["fr_times"][frame_idx]:
                    for f in ["fr_times", "labeler"]:
                        target_labels[f][frame_idx] = l[f][frame_idx]
                    for f in l["labels"]:
                        if frame_idx in l["labels"][f]:
                            target_labels["labels"][f][frame_idx] = l["labels"][f][frame_idx]
        np.savez(target_file, target_labels)

    elif args.check is not None:
        import calibcamlib
        if args.check == '-':
            from ACMtraingui.config import load_cfg
            cfg = load_cfg(
                input_path + '/labeling_gui_cfg.py')  # This will load a copy, might fail since paths are replaced
            calib_file = cfg['standardCalibrationFile']
        elif os.path.isdir(args.check):  # This is supposed to be filled with the config directory
            from ACMtraingui.config import load_cfg
            cfg = load_cfg(args.check + '/labeling_gui_cfg.py')
            calib_file = cfg['standardCalibrationFile']
        else:  # This is supposed to be filled with the path of the calib file
            calib_file = args.check
        print(calib_file)

        cs = calibcamlib.Camerasystem.from_calibcam_file(calib_file)
        labels = np.load(input_path + "/labels.npz", allow_pickle=True)['arr_0'].item()

        frame = []
        marker = []
        dists = []
        mdist = []
        pdists = []
        mpdist = []

        for i in labels.keys():
            for m in labels[i].keys():
                (X, P, V) = cs.triangulate_3derr(labels[i][m][:, np.newaxis, :])
                ds = calibcamlib.helper.calc_3derr(X, P, V)[1]

                x = cs.project(X)
                pd = np.sum((labels[i][m][:, np.newaxis, :] - x) ** 2, axis=2).T[0]

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

        for i, md in enumerate(mdists):
            print(
                f'{frames[i]:6} | {markers[i]:>25} | {np.nanmax(distss[i]):5.2f} | {list(map("{:5.2f}".format, distss[i]))} | {np.nanmax(pdistss[i]):6.1f} | {list(map("{:7.2f}".format, pdistss[i]))}')

    elif args.master:
        from ACMtraingui import labeling_gui
        labeling_gui.main(Path('.'), config_file=input_path, master=True)
    else:
        from ACMtraingui import labeling_gui
        labeling_gui.main(Path(input_path), master=False)
    return


if __name__ == '__main__':
    main()
