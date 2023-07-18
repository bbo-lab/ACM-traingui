from pathlib import Path
import numpy as np

version = 0.3


def update(labels, labeler="_unknown"):
    assert labels["version"] <= version, "Please update ACM traingui"

    # Before versioning
    if "version" not in labels or labels["version"] <= 0.2:
        if "labeler" not in labels:
            labels["labeler"] = {}
        if "labeler_list" not in labels:
            labels["labeler_list"] = [labeler]
        if labeler not in labels["labeler_list"]:
            labels["labeler_list"].append(labeler)

        labeled_frame_idxs = get_labeled_frame_idxs(labels)
        labeler_idx = labels["labeler_list"].index(labeler)

        for f_idx in labeled_frame_idxs:
            if f_idx not in labels["labeler"]:
                labels["labeler"][f_idx] = labeler_idx
            if f_idx not in labels["fr_times"]:
                labels["fr_times"][f_idx] = 0

    labels["version"] = version
    return labels


def get_labeled_frame_idxs(labels):
    labeled_frame_idxs = []
    for label in labels["labels"]:
        labeled_frame_idxs += list(labels["labels"][label].keys())

    return sorted(list(set(labeled_frame_idxs)))


def merge(labels_list: list, labeler_list=None, target_file=None):
    # Load data from files
    labels_files = None
    if isinstance(labels_list[0], str):
        labels_files = [Path(lf).expanduser().resolve() for lf in labels_list]
    elif isinstance(labels_list[0], Path):
        labels_files = labels_list
    else:
        assert target_file is not None, "target_file is only supported if labels_list contains paths"

    # Normalize path of target_file
    if isinstance(target_file, str):
        target_file = Path(target_file).expanduser().resolve()

    # Add target files as first labels file is existing
    if target_file is not None and target_file.is_file():
        labels_files.insert(0, target_file)

    # Load data from labels files
    if labels_files is not None:
        labels_list = [np.load(lf.as_posix(), allow_pickle=True)["arr_0"][()] for lf in labels_files]

        if labeler_list is None:
            labeler_list = [lf.parent.parent.name for lf in labels_files]

    # Update and copy
    labels_list = [update(labels.copy(), labeler=labeler) for labels, labeler in zip(labels_list, labeler_list)]

    # Add unknown labeler to list if nothing else is provided. This will only be used for old labels files that do not
    # contain labeler infos
    if labeler_list is None:
        labeler_list = ["_unknown" for _ in labels_list]

    labeler_list_all = make_global_labeler_list(labels_list, labeler_list)

    # Merge file-wise
    target_labels = labels_list[0]
    for labels in labels_list[1:]:
        labeled_frame_idxs = get_labeled_frame_idxs(target_labels)

        # Create all necessary labels in target
        for label in labels["labels"]:
            if label not in target_labels["labels"]:
                target_labels["labels"] = {}

        # Walk through frames
        for frame_idx in get_labeled_frame_idxs(labels):
            if frame_idx not in labeled_frame_idxs:
                # Frame is not yet labeled, everything can be copied over
                for f in ["fr_times", "labeler"]:
                    target_labels[f][frame_idx] = labels[f][frame_idx]
                for label in labels["labels"]:
                    if frame_idx in labels["labels"][label]:
                        target_labels["labels"][label][frame_idx] = labels["labels"][label][frame_idx]
            else:
                for label in labels["labels"]:
                    # Label not present for this frame in source
                    if frame_idx not in labels["labels"][label]:
                        continue

                    # Label not present for this frame in target, initialize
                    if frame_idx not in target_labels["labels"][label]:
                        target_labels["labels"][label][frame_idx] = \
                            np.full(labels["labels"][label][frame_idx].shape, np.nan)

                    target_cam_mask = ~np.any(np.isnan(target_labels["labels"][label][frame_idx]), axis=1)
                    source_cam_mask = ~np.any(np.isnan(labels["labels"][label][frame_idx]), axis=1)
                    replace_mask = source_cam_mask
                    if target_labels["fr_times"][frame_idx] > labels["fr_times"][frame_idx]:
                        # Frame is newer in target
                        replace_mask = np.logical_and(replace_mask, ~target_cam_mask)
                    target_labels["labels"][label][frame_idx][replace_mask] = \
                        labels["labels"][label][frame_idx][replace_mask]

                if target_labels["labeler"][frame_idx] != labels["labeler"][frame_idx]:
                    target_labels["labeler"][frame_idx] = labeler_list_all.index("_various")

                # Update time
                target_labels["fr_times"][frame_idx] = labels["fr_times"][frame_idx]

    sort_dictionaries(target_labels)

    if target_file is not None:
        np.savez(target_file, target_labels)
    return target_labels


def sort_dictionaries(target_labels):
    # Sort dictionaries
    for f in ["fr_times", "labeler"]:
        target_labels[f] = dict(sorted(target_labels[f].items()))
    for label in target_labels["labels"]:
        target_labels["labels"][label] = dict(sorted(target_labels["labels"][label].items()))


def make_global_labeler_list(labels_list, labeler_list=None):
    # This changes labeler_list!!!
    # Create a new global list of all labelers
    if labeler_list is None:
        labeler_list = []

    labeler_list_all = labeler_list
    for labels in labels_list:
        if labels is None:
            continue
        if "labeler_list" in labels:
            labeler_list_all += labels["labeler_list"]
    labeler_list_all.append("_various")
    labeler_list_all.append("_unknown")
    labeler_list_all = sorted(list(set(labeler_list_all)))
    print(labeler_list_all)
    # Rewrite to global index list
    for labels, p in zip(labels_list, labeler_list):
        if labels is None:
            continue
        for frame_idx, pp in labels["labeler"].items():
            labels["labeler"][frame_idx] = labeler_list_all.index(labels["labeler_list"][pp])
        labels["labeler_list"] = labeler_list_all
    return labeler_list_all


def combine_cams(labels_list: list, target_file=None):
    # Normalize path of target_file
    if isinstance(target_file, str):
        target_file = Path(target_file).expanduser().resolve()

    # Load data from files
    for i_l, label in enumerate(labels_list):
        if isinstance(label, str):
            if label == "None":
                labels_list[i_l] = None
                continue
            labels_list[i_l] = Path(label)
        if isinstance(label, Path):
            labels_list[i_l] = labels_list[i_l].expanduser().resolve()
        labels_list[i_l] = np.load(labels_list[i_l].as_posix(), allow_pickle=True)["arr_0"][()]

    # Update and copy
    labels_list = [update(labels.copy()) if labels is not None else None for labels in labels_list]

    target_labels = get_empty_labels()
    target_labels['labeler_list'] = make_global_labeler_list(labels_list)

    for cam_idx, labels in enumerate(labels_list):
        if labels is None:
            continue

        for frame_idx in get_labeled_frame_idxs(labels):
            if frame_idx in target_labels["fr_times"]:
                target_labels["fr_times"][frame_idx] = min(target_labels["fr_times"][frame_idx],
                                                           labels["fr_times"][frame_idx])
            else:
                target_labels["fr_times"][frame_idx] = labels["fr_times"][frame_idx]

            target_labels["labeler"][frame_idx] = target_labels['labeler_list'].index("_various")

            for label in labels["labels"]:
                if label not in target_labels["labels"]:
                    target_labels["labels"][label] = {}
                    
                if frame_idx not in target_labels["labels"][label]:
                    target_labels["labels"][label][frame_idx] = np.full((len(labels_list), 2), np.nan)

                if frame_idx in labels["labels"][label]:
                    target_labels["labels"][label][frame_idx][cam_idx] = labels["labels"][label][frame_idx]

    sort_dictionaries(target_labels)

    if target_file is not None:
        np.savez(target_file, target_labels)
    return target_labels


def get_empty_labels():
    return {
        'labels': {},
        'fr_times': {},
        'labeler_list': [],
        'labeler': {},
        'version': version,
    }