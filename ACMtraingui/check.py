import os
import numpy as np
def check_triangulation_error(check_arg, input_path):
    import calibcamlib
    if check_arg == '-':
        from ACMtraingui.config import load_cfg
        cfg = load_cfg(
            input_path + '/labeling_gui_cfg.py')  # This will load a copy, might fail since paths are replaced
        calib_file = cfg['standardCalibrationFile']
    elif os.path.isdir(check_arg):  # This is supposed to be filled with the config directory
        from ACMtraingui.config import load_cfg
        cfg = load_cfg(check_arg + '/labeling_gui_cfg.py')
        calib_file = cfg['standardCalibrationFile']
    else:  # This is supposed to be filled with the path of the calib file
        calib_file = check_arg
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