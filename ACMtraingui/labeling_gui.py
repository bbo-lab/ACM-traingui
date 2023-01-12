#!/usr/bin/env python3

import copy
import numpy as np
import os
import sys
import calibcamlib
import time
import typing
from typing import List, Dict, Optional

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication, QIntValidator, QCursor
from PyQt5.QtWidgets import QAbstractItemView, \
    QApplication, \
    QComboBox, \
    QDialog, \
    QFrame, \
    QFileDialog, \
    QGridLayout, \
    QLabel, \
    QLineEdit, \
    QListWidget, \
    QMainWindow, \
    QPushButton, \
    QSizePolicy

from matplotlib import colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.image import AxesImage
from matplotlib.figure import Figure

from pathlib import Path

import imageio
# noinspection PyUnresolvedReferences
from ccvtools import rawio

from ACMtraingui.config import load_cfg, save_cfg


def rodrigues2rotmat_single(r):
    theta = np.power(r[0] ** 2 + r[1] ** 2 + r[2] ** 2, 0.5)
    u = r / (theta + np.abs(np.sign(theta)) - 1.0)
    # row 1
    rotmat_00 = np.cos(theta) + u[0] ** 2 * (1.0 - np.cos(theta))
    rotmat_01 = u[0] * u[1] * (1.0 - np.cos(theta)) - u[2] * np.sin(theta)
    rotmat_02 = u[0] * u[2] * (1.0 - np.cos(theta)) + u[1] * np.sin(theta)

    # row 2
    rotmat_10 = u[0] * u[1] * (1.0 - np.cos(theta)) + u[2] * np.sin(theta)
    rotmat_11 = np.cos(theta) + u[1] ** 2 * (1.0 - np.cos(theta))
    rotmat_12 = u[1] * u[2] * (1.0 - np.cos(theta)) - u[0] * np.sin(theta)

    # row 3
    rotmat_20 = u[0] * u[2] * (1.0 - np.cos(theta)) - u[1] * np.sin(theta)
    rotmat_21 = u[1] * u[2] * (1.0 - np.cos(theta)) + u[0] * np.sin(theta)
    rotmat_22 = np.cos(theta) + u[2] ** 2 * (1.0 - np.cos(theta))

    rotmat = np.array([[rotmat_00, rotmat_01, rotmat_02],
                       [rotmat_10, rotmat_11, rotmat_12],
                       [rotmat_20, rotmat_21, rotmat_22]], dtype=np.float64)

    return rotmat


def calc_dst(m_udst, k):
    x_1 = m_udst[:, 0] / m_udst[:, 2]
    y_1 = m_udst[:, 1] / m_udst[:, 2]

    r2 = x_1 ** 2 + y_1 ** 2

    x_2 = x_1 * (1.0 + k[0] * r2 + k[1] * r2 ** 2 + k[4] * r2 ** 3) + 2.0 * k[2] * x_1 * y_1 + k[3] * (
            r2 + 2.0 * x_1 ** 2)

    y_2 = y_1 * (1.0 + k[0] * r2 + k[1] * r2 ** 2 + k[4] * r2 ** 3) + k[2] * (r2 + 2.0 * y_1 ** 2) + 2.0 * k[
        3] * x_1 * y_1

    n_points = np.size(m_udst, 0)
    ones = np.ones(n_points, dtype=np.float64)
    m_dst = np.concatenate([[x_2], [y_2], [ones]], 0).T
    return m_dst


def read_video_meta(reader):
    header = reader.get_meta_data()
    header['nFrames'] = len(reader)  # len() may be Inf for formats where counting frames can be expensive
    if 1000000000000000 < header['nFrames']:
        header['nFrames'] = reader.count_frames()

    # Add required headers that are not normally part of standard video formats but are required information
    if "sensor" in header:
        header['offset'] = tuple(header['sensor']['offset'])
        header['sensorsize'] = tuple(header['sensor']['size'])
    else:
        print("Infering sensor size from image and setting offset to 0!")
        header['sensorsize'] = (reader.get_data(0).shape[1], reader.get_data(0).shape[0], reader.get_data(0).shape[2])
        header['offset'] = tuple(np.asarray([0, 0]))

    return header


# look at: Rational Radial Distortion Models with Analytical Undistortion Formulae, Lili Ma et al.
# source: https://arxiv.org/pdf/cs/0307047.pdf
# only works for k = [k1, k2, 0, 0, 0]
def calc_udst(m_dst, k):
    assert np.all(k[2:] == 0.0), 'ERROR: Undistortion only valid for up to two radial distortion coefficients.'

    x_2 = m_dst[:, 0]
    y_2 = m_dst[:, 1]

    # use r directly instead of c
    n_points = np.size(m_dst, 0)
    p = np.zeros(6, dtype=np.float64)
    p[4] = 1.0

    x_1 = np.zeros(n_points, dtype=np.float64)
    y_1 = np.zeros(n_points, dtype=np.float64)
    for i_point in range(n_points):
        cond = (np.abs(x_2[i_point]) > np.abs(y_2[i_point]))
        if cond:
            c = y_2[i_point] / x_2[i_point]
            p[5] = -x_2[i_point]
        else:
            c = x_2[i_point] / y_2[i_point]
            p[5] = -y_2[i_point]
        #        p[4] = 1
        p[2] = k[0] * (1.0 + c ** 2)
        p[0] = k[1] * (1.0 + c ** 2) ** 2
        sol = np.real(np.roots(p))
        # use min(abs(x)) to make your model as accurate as possible
        sol_abs = np.abs(sol)
        if cond:
            x_1[i_point] = sol[sol_abs == np.min(sol_abs)][0]
            y_1[i_point] = c * x_1[i_point]
        else:
            y_1[i_point] = sol[sol_abs == np.min(sol_abs)][0]
            x_1[i_point] = c * y_1[i_point]
    m_udst = np.concatenate([[x_1], [y_1], [m_dst[:, 2]]], 0).T
    return m_udst


# ATTENTION: hard coded
def sort_label_sequence(seq):
    num_order = list(['tail', 'spine', 'head'])
    left_right_order = list(['shoulder', 'elbow', 'wrist', 'paw_front', 'finger',
                             'side',
                             'hip', 'knee', 'ankle', 'paw_hind', 'toe'])
    #
    labels_num = list()
    labels_left = list()
    labels_right = list()
    for label in seq:
        label_split = label.split('_')
        if 'right' in label_split:
            labels_right.append(label)
        elif 'left' in label_split:
            labels_left.append(label)
        else:
            labels_num.append(label)
    labels_num = sorted(labels_num)
    labels_left = sorted(labels_left)
    labels_right = sorted(labels_right)
    #
    labels_num_sorted = list([[] for _ in num_order])
    labels_left_sorted = list([[] for _ in left_right_order])
    labels_right_sorted = list([[] for _ in left_right_order])
    for label in labels_num:
        label_split = label.split('_')
        label_use = '_'.join(label_split[1:-1])
        index = num_order.index(label_use)
        labels_num_sorted[index].append(label)
    labels_num_sorted = list([i for j in labels_num_sorted for i in j])
    for label in labels_left:
        label_split = label.split('_')
        label_use = '_'.join(label_split[1:-1])
        label_use_split = label_use.split('_')
        if 'left' in label_use_split:
            label_use_split.remove('left')
        label_use = '_'.join(label_use_split)
        index = left_right_order.index(label_use)
        labels_left_sorted[index].append(label)
    labels_left_sorted = list([i for j in labels_left_sorted for i in j])
    for label in labels_right:
        label_split = label.split('_')
        label_use = '_'.join(label_split[1:-1])
        label_use_split = label_use.split('_')
        if 'right' in label_use_split:
            label_use_split.remove('right')
        label_use = '_'.join(label_use_split)
        index = left_right_order.index(label_use)
        labels_right_sorted[index].append(label)
    labels_right_sorted = list([i for j in labels_right_sorted for i in j])
    #
    seq_ordered = labels_num_sorted + labels_left_sorted + labels_right_sorted
    return seq_ordered


class SelectUserWindow(QDialog):
    def __init__(self, drive: Path, parent=None):
        super(SelectUserWindow, self).__init__(parent)
        self.drive = drive

        self.setGeometry(0, 0, 256, 128)
        self.center()
        self.setWindowTitle('Select User')

        self.user_list = self.get_user_list(drive)

        self.selecting_layout = QGridLayout()

        self.selecting_field = QComboBox()
        self.selecting_field.addItems(self.user_list)
        self.selecting_field.setSizePolicy(QSizePolicy.Expanding,
                                           QSizePolicy.Preferred)
        self.selecting_layout.addWidget(self.selecting_field)

        self.selecting_button = QPushButton('Ok')
        self.selecting_button.clicked.connect(self.accept)
        self.selecting_button.setSizePolicy(QSizePolicy.Expanding,
                                            QSizePolicy.Preferred)
        self.selecting_layout.addWidget(self.selecting_button)

        self.setLayout(self.selecting_layout)

    @staticmethod
    def get_user_list(drive):
        user_list = sorted(os.listdir(drive / 'pose/data/user'))
        return user_list

    def center(self):
        qr = self.frameGeometry()
        cp = QGuiApplication.primaryScreen().geometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def get_user(self):
        user_id = self.selecting_field.currentIndex()
        user = self.user_list[user_id]
        return user

    @staticmethod
    def start(drive, parent=None):
        selecting = SelectUserWindow(drive=drive, parent=parent)
        exit_sel = selecting.exec_()
        user = selecting.get_user()
        return user, exit_sel == QDialog.Accepted


class MainWindow(QMainWindow):
    def __init__(self, drive: Path, file_config=None, master=True, parent=None):
        super(MainWindow, self).__init__(parent)

        # Parameters
        self.drive = drive
        self.master = master

        if self.master:
            if file_config is None:
                file_config = 'labeling_gui_cfg.py'  # use hard coded path here
            self.cfg = load_cfg(file_config)
        else:
            if os.path.isdir(self.drive):
                self.user, correct_exit = SelectUserWindow.start(drive)
                if correct_exit:
                    file_config = self.drive / 'pose' / 'data' / 'user' / self.user / 'labeling_gui_cfg.py'
                    self.cfg = load_cfg(file_config)
                else:
                    sys.exit()
            else:
                print('ERROR: Server is not mounted')
                sys.exit()

        self.model = None

        # Files
        self.standardCalibrationFile = Path(self.cfg['standardCalibrationFile'])
        self.standardOriginCoordFile = Path(self.cfg['standardOriginCoordFile'])
        self.standardModelFile = Path(self.cfg['standardModelFile'])
        self.standardLabelsFile = Path(self.cfg['standardLabelsFile'])
        self.standardSketchFile = Path(self.cfg['standardSketchFile'])
        self.standardLabelsFolder = None

        # Data load status
        self.recordingIsLoaded = False
        self.calibrationIsLoaded = False
        self.originIsLoaded = False
        self.modelIsLoaded = False
        self.labelsAreLoaded = False
        self.sketchIsLoaded = False

        # Loaded data
        self.cameras: List[Dict] = []
        self.calibration = None
        self.camera_system = None
        self.origin_coord = None
        self.sketch = None

        # Controls
        self.controls = {
            'canvases': {},
            'figs': {},
            'axes': {},
            'plots': {},
            'frames': {},
            'grids': {},
            'lists': {},
            'fields': {},
            'labels': {},
            'buttons': {},
            'texts': {},
            'status': {  # control statuses - FIXME: should be derived from buttons with method
                'toolbars_zoom': False,
                'toolbars_pan': False,
                'label3d_select': False,
            }
        }

        self.labels = {
            'labels': {},
            'fr_times': {},
            'version': 0.2,
        }

        # Sketch zoom stuff
        self.sketch_zoom_dy = None
        self.sketch_zoom_dx = None
        self.sketch_zoom_scale = 0.1

        # Toolbars
        self.toolbars = list()

        # Stuff ... stuff
        self.labels3d_sequence = None
        self.cid = None
        self.cidSketch = None
        self.label2d_max_err = []

        self.dx = int(128)
        self.dy = int(128)
        self.vmin = int(0)
        self.vmax = int(127)
        self.dFrame = self.cfg['dFrame']

        self.minPose = self.cfg['minPose']
        self.maxPose = self.cfg['maxPose']
        self.pose_idx = self.minPose

        self.i_cam = self.cfg['cam']

        self.labels2d_all = dict()
        self.labels2d = dict()
        self.clickedLabel2d = np.array([np.nan, np.nan], dtype=np.float64)
        self.clickedLabel2d_pose = self.get_pose_idx()
        self.selectedLabel2d = np.full((len(self.cameras), 2), np.nan, dtype=np.float64)

        self.clickedLabel3d = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        self.selectedLabel3d = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

        self.dxyz_lim = float(0.4)
        self.dxyz = float(0.01)
        self.labels3d = dict()

        self.colors = []
        self.init_colors()

        self.autoSaveCounter = int(0)

        self.setGeometry(0, 0, 1024, 768)
        self.showMaximized()

        self.init_files_folders()
        self.set_controls()
        self.set_layout()

        self.plot2d_ini()

        self.setFocus()
        self.setWindowTitle('Labeling GUI')
        self.show()

        if not (self.cfg['list_fastLabelingMode']):
            self.self.controls['lists']['fast_labeling_mode'].setCurrentIndex(self.cfg['cam'])
        if self.cfg['button_fastLabelingMode_activate']:
            self.button_fast_labeling_mode_press()
        if self.cfg['button_centricViewMode_activate']:
            self.button_centric_view_mode_press()
        if self.cfg['button_reprojectionMode_activate']:
            self.button_reprojection_mode_press()
        if self.cfg['button_sketchMode_activate']:
            self.button_sketch_mode_press()

    def init_files_folders(self):
        standard_recording_folder = Path(self.cfg['standardRecordingFolder'])

        # create folder structure / save backup / load last pose
        if not self.master:
            self.init_assistant_folders(standard_recording_folder)
            self.init_autosave()
            save_cfg(self.standardLabelsFolder / 'labeling_gui_cfg.py', self.cfg)
            self.restore_last_pose_idx()

        if self.cfg['autoLoad']:
            rec_file_names = sorted(
                [standard_recording_folder / i for i in self.cfg['standardRecordingFileNames']]
            )
            self.load_recordings_from_names(rec_file_names)

            self.load_calibrations()
            self.load_origin()
            # self.load_model()
            self.load_sketch()
            self.load_labels()

    def load_labels(self, labels_file: typing.Optional[Path] = None):
        if labels_file is None:
            labels_file = self.standardLabelsFile

        # load labels
        if not self.master:
            labels_file = self.standardLabelsFolder / 'labels.npz'

        if labels_file.is_file():
            labels = np.load(labels_file.as_posix(), allow_pickle=True)['arr_0'][()]
            if 'version' in labels:
                self.labels = labels
                self.labelsAreLoaded = True
            else:
                print(f'WARNING: Autoloading failed. Legacy labels file {labels_file} not loaded.')
        else:
            print(f'WARNING: Autoloading failed. Labels file {labels_file} does not exist.')

    def load_sketch(self, sketch_file: typing.Optional[Path] = None):
        if sketch_file is None:
            sketch_file = self.standardSketchFile

        # load sketch
        if sketch_file.is_file():
            self.sketch = np.load(sketch_file.as_posix(), allow_pickle=True)[()]

            sketch = self.get_sketch()
            self.sketch_zoom_dx = np.max(np.shape(sketch)) * self.sketch_zoom_scale
            self.sketch_zoom_dy = np.max(np.shape(sketch)) * self.sketch_zoom_scale

            self.sketchIsLoaded = True
        else:
            print(f'WARNING: Autoloading failed. Sketch file {self.standardSketchFile} does not exist.')

        for label_name in self.get_sketch_labels():
            if label_name not in self.labels['labels']:
                self.labels['labels'][label_name] = {}

    def get_sketch(self):
        return self.sketch['sketch']

    def get_sketch_labels(self):
        return self.sketch['sketch_label_locations']

    def get_sketch_label_coordinates(self):
        return np.array(list(self.get_sketch_labels().values()), dtype=np.float64)

    def load_origin(self, origin_file: typing.Optional[Path] = None):
        if origin_file is None:
            origin_file = self.standardOriginCoordFile

        if origin_file.is_file():
            self.originIsLoaded = True
            self.origin_coord = np.load(origin_file.as_posix(), allow_pickle=True)[()]
        else:
            print(f'WARNING: Autoloading failed. Origin/Coord file {origin_file} does not exist.')

    def get_origin_coord(self):
        return self.origin_coord['origin'], self.origin_coord['coord']

    def load_calibrations(self, calibrations_file: typing.Optional[Path] = None):
        if calibrations_file is None:
            calibrations_file = self.standardCalibrationFile

        if calibrations_file.is_file():
            self.camera_system = calibcamlib.Camerasystem.from_calibcam_file(calibrations_file.as_posix())
            self.calibrationIsLoaded = True

            self.calibration = np.load(calibrations_file.as_posix(), allow_pickle=True)[()]
        else:
            print(f'WARNING: Autoloading failed. Calibration file {calibrations_file} does not exist.')

    # noinspection PyPep8Naming
    def get_calibration_params(self):
        A_val = self.calibration['A_fit']
        A = np.zeros((len(self.cameras), 3, 3), dtype=np.float64)
        for i in range(len(self.cameras)):
            A[i][0, 0] = A_val[i, 0]
            A[i][0, 2] = A_val[i, 1]
            A[i][1, 1] = A_val[i, 2]
            A[i][1, 2] = A_val[i, 3]
            A[i][2, 2] = 1.0

        return {
            'A': A,
            'k': self.calibration['k_fit'],
            'rX1': self.calibration['rX1_fit'],
            'RX1': self.calibration['RX1_fit'],
            'tX1': self.calibration['tX1_fit'],
        }

    def load_recordings_from_names(self, names):
        # load recording
        if np.all([i_file.is_file() for i_file in names]):
            self.recordingIsLoaded = True

            cameras = []
            for file_name in names:
                if file_name:
                    reader = imageio.get_reader(file_name)
                    header = read_video_meta(reader)
                    cam = {
                        'file_name': file_name,
                        'reader': reader,
                        'header': header,
                        'x_lim_prev': (0, header['sensorsize'][0]),
                        'y_lim_prev': (0, header['sensorsize'][1]),
                        'rotate': False,
                    }
                    cameras.append(cam)
                else:
                    print(f'WARNING: Invalid recording file {file_name}')

            self.cameras = cameras

        else:
            print(f'WARNING: Autoloading failed. Recording files do not exist: {names}')

    def get_n_poses(self):
        return [cam["header"]["nFrames"] for cam in self.cameras]

    def get_x_res(self):
        return [cam["header"]["sensorsize"][0] for cam in self.cameras]

    def get_y_res(self):
        return [cam["header"]["sensorsize"][1] for cam in self.cameras]

    def restore_last_pose_idx(self):
        # last pose
        file_exit_status = self.standardLabelsFolder / 'exit_status.npy'
        if file_exit_status.is_file():
            exit_status = np.load(file_exit_status.as_posix(), allow_pickle=True)[()]
            self.set_pose_idx(exit_status['i_pose'])

    def init_autosave(self):
        # autosave
        autosavefolder = self.standardLabelsFolder / 'autosave'
        if not autosavefolder.is_dir():
            os.makedirs(autosavefolder)
        save_cfg(autosavefolder / 'labeling_gui_cfg.py', self.cfg)
        # file = self.standardLabelsFolder / 'labels.npz'
        # if file.is_file():
        #     labels_save = np.load(file.as_posix(), allow_pickle=True)['arr_0'][()]
        #     np.savez(autosavefolder / 'labels.npz', labels_save)

    def init_assistant_folders(self, standard_recording_folder):
        # folder structure
        userfolder = self.drive / 'pose' / 'user' / self.user
        if not userfolder.is_dir():
            os.makedirs(userfolder)
        resultsfolder = userfolder / standard_recording_folder.name
        if not resultsfolder.is_dir():
            os.makedirs(resultsfolder)
        self.standardLabelsFolder = resultsfolder
        # backup
        backupfolder = self.standardLabelsFolder / 'backup'
        if not backupfolder.is_dir():
            os.mkdir(backupfolder)
        file = self.standardLabelsFolder / 'labeling_gui_cfg.py'
        if file.is_file():
            cfg_old = load_cfg(file)
            save_cfg(backupfolder / 'labeling_gui_cfg.py', cfg_old)
        file = self.standardLabelsFolder / 'labels.npz'
        if file.is_file():
            labels_old = np.load(file.as_posix(), allow_pickle=True)['arr_0'][()]
            np.savez(backupfolder / 'labels.npz', labels_old)

    def init_colors(self):
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        # Sort colors by hue, saturation, value and name.
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                        for name, color in colors.items())
        sorted_names = [name for hsv, name in by_hsv]
        for i in range(24, -1, -1):
            self.colors = self.colors + sorted_names[i::24]

    def get_pose_idx(self):
        return self.pose_idx

    def set_pose_idx(self, pose_idx):
        self.pose_idx = pose_idx
        if (self.pose_idx < self.minPose) or (self.pose_idx > self.maxPose):
            self.pose_idx = self.minPose

    def set_layout(self):
        # frame main
        frame_main = QFrame()
        self.setStyleSheet("background-color: black;")
        layout_grid = QGridLayout()
        layout_grid.setSpacing(10)
        frame_main.setMinimumSize(512, 512)
        # frame for 2d views
        self.controls['frames']['views2d'].setStyleSheet("background-color: white;")
        layout_grid.setRowStretch(0, 2)
        layout_grid.setColumnStretch(0, 2)
        layout_grid.addWidget(self.controls['frames']['views2d'], 0, 0, 2, 4)

        self.controls['grids']['views2d'].setSpacing(0)
        self.controls['frames']['views2d'].setLayout(self.controls['grids']['views2d'])

        # frame for 3d model
        self.controls['frames']['views3d'].setStyleSheet("background-color:  white;")
        layout_grid.setRowStretch(0, 1)
        layout_grid.setColumnStretch(4, 1)
        layout_grid.addWidget(self.controls['frames']['views3d'], 0, 4)

        self.controls['grids']['views3d'].setSpacing(0)
        self.controls['frames']['views3d'].setLayout(self.controls['grids']['views3d'])

        # frame for controls
        self.controls['frames']['controls'].setStyleSheet("background-color: white;")
        layout_grid.setRowStretch(1, 1)
        layout_grid.setColumnStretch(4, 1)
        layout_grid.addWidget(self.controls['frames']['controls'], 1, 4)
        # add to grid
        frame_main.setLayout(layout_grid)
        self.setCentralWidget(frame_main)

    # 2d plots
    def plot2d_update(self):
        self.plot2d_draw_labels(self.controls['axes']['2d'])
        self.controls['figs']['2d'].canvas.draw()

    def plot2d_draw_labels(self, ax):
        # ax.lines = list()

        cam_idx = self.i_cam
        frame_idx = self.get_pose_idx()
        current_label_name = self.get_current_label()
        for label_name in self.labels['labels']:
            if frame_idx not in self.labels['labels'][label_name]:
                continue

            point = self.labels['labels'][label_name][frame_idx][cam_idx, :]

            if current_label_name == label_name:
                plotparams = {
                    'color': 'darkgreen',
                    'markersize': 4,
                    'zorder': 3,
                }
            else:
                plotparams = {
                    'color': 'cyan',
                    'markersize': 3,
                    'zorder': 2,
                }

            if '2d' not in self.controls['plots']:
                self.controls['plots']['2d'] = {}
            if label_name in self.controls['plots']['2d']:
                self.controls['plots']['2d'][label_name].remove()

            self.controls['plots']['2d'][label_name] = ax.plot([point[0]], [point[1]],
                                                               marker='o',
                                                               **plotparams,
                                                               )[0]

    def get_current_label(self):
        current_label_item = self.controls['lists']['labels'].currentItem()
        if current_label_item is not None:
            current_label_name = current_label_item.text()
        else:
            current_label_name = self.controls['lists']['labels'].item(0).text()
        return current_label_name

    def plot2d_plot(self, ax, i_cam):
        reader = self.cameras[i_cam]["reader"]
        img = reader.get_data(self.pose_idx)

        if self.controls['plots']['image2d'] is None:
            self.controls['plots']['image2d'] = ax.imshow(img,
                                                          aspect=1,
                                                          cmap='gray',
                                                          vmin=self.vmin,
                                                          vmax=self.vmax)
            ax.legend('',
                      facecolor=self.colors[i_cam % np.size(self.colors)],
                      loc='upper left',
                      bbox_to_anchor=(0, 1))
            ax.axis('off')
        else:
            self.controls['plots']['image2d'].set_array(img)
            self.controls['plots']['image2d'].set_clim(self.vmin, self.vmax)

        x_res = self.get_x_res()
        y_res = self.get_y_res()
        ax.set_xlim(0.0, x_res[i_cam] - 1)
        ax.set_ylim(0.0, y_res[i_cam] - 1)

        if self.cfg['invert_xaxis']:
            ax.invert_xaxis()
        if self.cfg['invert_yaxis']:
            ax.invert_yaxis()
        #
        self.plot2d_draw_labels(ax)

    def plot2d_ini(self):
        for i in reversed(range(self.controls['grids']['views2d'].count())):
            widget_to_remove = self.controls['grids']['views2d'].itemAt(i).widget()
            self.controls['grids']['views2d'].removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)

        if self.controls['status']['toolbars_zoom']:
            self.button_zoom_press()
        if self.controls['status']['toolbars_pan']:
            self.button_pan_press()

        fig = Figure(tight_layout=True)
        fig.clear()
        self.controls['figs']['2d'] = fig
        canvas = FigureCanvasQTAgg(fig)
        canvas.setParent(self.controls['frames']['views2d'])
        ax = fig.add_subplot('111')
        ax.clear()
        self.controls['axes']['2d'] = ax
        self.plot2d_plot(self.controls['axes']['2d'], self.i_cam)

        self.controls['grids']['views2d'].addWidget(canvas, 0, 0)

        toolbar = NavigationToolbar2QT(canvas, self)
        toolbar.hide()
        self.toolbars = list([toolbar])

        self.controls['frames']['views2d'].setCursor(QCursor(QtCore.Qt.CrossCursor))

        fig.canvas.mpl_connect('button_press_event',
                               lambda event: self.plot2d_click(event))

    def plot2d_draw_fast(self):
        if self.controls['status']['toolbars_zoom']:
            self.button_zoom_press()
        if self.controls['status']['toolbars_pan']:
            self.button_pan_press()

        self.plot2d_plot(self.controls['axes']['2d'], self.i_cam)

    def plot2d_click(self, event):
        if not self.controls['status']['toolbars_zoom'] and not self.controls['status']['toolbars_pan']:
            ax = event.inaxes

            # Initialize array
            cam_idx = self.i_cam
            frame_idx = self.get_pose_idx()
            label_name = self.get_current_label()

            if label_name not in self.labels['labels']:
                self.labels['labels'][label_name] = {}

            if frame_idx not in self.labels['labels'][label_name]:
                self.labels['labels'][label_name][frame_idx] = np.full((len(self.cameras), 2), np.nan, dtype=np.float64)
                self.labels['fr_times'][frame_idx] = time.time()

            if ax is not None:
                # Left mouse - create
                if event.button == 1:
                    x = event.xdata
                    y = event.ydata
                    if (x is not None) and (y is not None):
                        coords = np.array([x, y], dtype=np.float64)
                        self.labels['labels'][label_name][frame_idx][cam_idx] = coords
                        self.plot2d_update()

                        if self.controls['status']['button_sketchMode']:
                            self.sketch_update()

                # Right mouse - delete
                elif event.button == 3:
                    self.labels['labels'][label_name][frame_idx][cam_idx, :] = np.nan
                    self.plot2d_update()

                    if self.controls['status']['button_sketchMode']:
                        self.sketch_update()

    # sketch
    def sketch_init(self):
        for i in reversed(range(self.controls['grids']['views3d'].count())):
            widget_to_remove = self.controls['grids']['views3d'].itemAt(i).widget()
            self.controls['grids']['views3d'].removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)

        sketch = self.get_sketch()

        self.controls['figs']['sketch'].clear()

        # full
        ax_sketch_dims = [0 / 3, 1 / 18, 1 / 3, 16 / 18]
        self.controls['axes']['sketch'] = self.controls['figs']['sketch'].add_axes(ax_sketch_dims)
        self.controls['axes']['sketch'].clear()
        self.controls['axes']['sketch'].grid(False)
        self.controls['axes']['sketch'].imshow(sketch)
        self.controls['axes']['sketch'].axis('off')
        self.controls['axes']['sketch'].set_title('Full:',
                                                  ha='center', va='center',
                                                  zorder=0)

        self.controls['axes']['sketch'].set_xlim(
            [-self.sketch_zoom_dx / 2, np.shape(sketch)[1] + self.sketch_zoom_dx / 2])
        self.controls['axes']['sketch'].set_ylim(
            [-self.sketch_zoom_dy / 2, np.shape(sketch)[0] + self.sketch_zoom_dy / 2])
        self.controls['axes']['sketch'].invert_yaxis()
        # zoom
        ax_sketch_zoom_dims = [1 / 3, 5 / 18, 2 / 3, 12 / 18]
        self.controls['axes']['sketch_zoom'] = self.controls['figs']['sketch'].add_axes(ax_sketch_zoom_dims)
        self.controls['axes']['sketch_zoom'].imshow(sketch)
        self.controls['axes']['sketch_zoom'].set_xlabel('')
        self.controls['axes']['sketch_zoom'].set_ylabel('')
        self.controls['axes']['sketch_zoom'].set_xticks(list())
        self.controls['axes']['sketch_zoom'].set_yticks(list())
        self.controls['axes']['sketch_zoom'].set_xticklabels(list())
        self.controls['axes']['sketch_zoom'].set_yticklabels(list())
        self.controls['axes']['sketch_zoom'].set_title('Zoom:',
                                                       ha='center', va='center',
                                                       zorder=0)
        self.controls['axes']['sketch_zoom'].grid(False)

        self.controls['axes']['sketch_zoom'].set_xlim(
            [np.shape(sketch)[1] / 2 - self.sketch_zoom_dx, np.shape(sketch)[1] / 2 + self.sketch_zoom_dx])
        self.controls['axes']['sketch_zoom'].set_ylim(
            [np.shape(sketch)[0] / 2 - self.sketch_zoom_dy, np.shape(sketch)[0] / 2 + self.sketch_zoom_dy])
        self.controls['axes']['sketch_zoom'].invert_yaxis()
        # text
        self.controls['texts']['sketch'] = self.controls['figs']['sketch'].text(
            ax_sketch_dims[0] + ax_sketch_dims[2] / 2,
            ax_sketch_dims[1] / 2,
            'Label {:02d}:\n{:s}'.format(0, ''),
            ha='center', va='center',
            fontsize=18,
            zorder=2)

        self.controls['labels']['sketch'] = {}
        self.controls['labels']['sketch_zoom'] = {}

        self.sketch_init_labels()
        self.sketch_update()
        self.controls['grids']['views3d'].addWidget(self.controls['canvases']['sketch'])

    def sketch_init_labels(self):
        dot_params = {
            'color': 'darkgreen',
            'marker': '.',
            'markersize': 2,
            'alpha': 1.0,
            'zorder': 2,
        }
        circle_params = {
            'color': 'darkgreen',
            'marker': 'o',
            'markersize': 40,
            'markeredgewidth': 4,
            'fillstyle': 'none',
            'alpha': 2 / 3,
            'zorder': 2,
        }

        for ln in ['sketch_dot', 'sketch_circle', 'sketch_zoom_dot', 'sketch_zoom_circle']:
            if ln in self.controls['plots']:
                self.controls['plots'][ln].remove()
        self.controls['plots']['sketch_dot'] = self.controls['axes']['sketch'].plot(
            [np.nan], [np.nan], **dot_params)[0]
        self.controls['plots']['sketch_circle'] = self.controls['axes']['sketch'].plot(
            [np.nan], [np.nan], **circle_params)[0]
        self.controls['plots']['sketch_zoom_dot'] = self.controls['axes']['sketch_zoom'].plot(
            [np.nan], [np.nan], **dot_params)[0]
        self.controls['plots']['sketch_zoom_circle'] = self.controls['axes']['sketch_zoom'].plot(
            [np.nan], [np.nan], **circle_params)[0]

        for label_name, label_location in self.get_sketch_labels().items():
            if label_name in self.controls['labels']['sketch']:
                self.controls['labels']['sketch'][label_name].remove()
            self.controls['labels']['sketch'][label_name] = \
                self.controls['axes']['sketch'].plot([label_location[0]], [label_location[1]],
                                                     marker='o',
                                                     color='orange',
                                                     markersize=3,
                                                     zorder=1)[0]

            if label_name in self.controls['labels']['sketch_zoom']:
                self.controls['labels']['sketch_zoom'][label_name].remove()
            self.controls['labels']['sketch_zoom'][label_name] = \
                self.controls['axes']['sketch_zoom'].plot([label_location[0]],
                                                          [label_location[1]],
                                                          marker='o',
                                                          color='orange',
                                                          markersize=5,
                                                          zorder=1)[0]

    def sketch_update(self):
        sketch_labels = self.get_sketch_labels()
        label_names = list(self.labels['labels'].keys())

        for label_name in label_names:
            self.controls['labels']['sketch'][label_name].set(color='orange')
            self.controls['labels']['sketch_zoom'][label_name].set(color='orange')

        current_label_name = self.get_current_label()
        if current_label_name is None or current_label_name not in sketch_labels:
            return

        (x, y) = sketch_labels[current_label_name]

        self.controls['plots']['sketch_dot'].set_data([x], [y])
        self.controls['plots']['sketch_circle'].set_data([x], [y])
        # zoom
        self.controls['plots']['sketch_zoom_dot'].set_data([x], [y])
        self.controls['plots']['sketch_zoom_circle'].set_data([x], [y])
        self.controls['axes']['sketch_zoom'].set_xlim([x - self.sketch_zoom_dx, x + self.sketch_zoom_dx])
        self.controls['axes']['sketch_zoom'].set_ylim([y - self.sketch_zoom_dy, y + self.sketch_zoom_dy])
        self.controls['axes']['sketch_zoom'].invert_yaxis()

        self.controls['canvases']['sketch'].draw()
        self.button_zoom_press(tostate=["off"])

    # controls
    def set_controls(self):
        controls = self.controls

        # controls
        controls['frames']['controls'] = QFrame()

        # 3d view
        # controls['figs']['3d'] = Figure(tight_layout=True)
        # controls['axes']['3d'] = controls['figs']['3d'].add_subplot(111, projection='3d')
        # controls['canvases']['3d'] = FigureCanvasQTAgg(controls['figs']['3d'])
        controls['frames']['views3d'] = QFrame()
        controls['grids']['views3d'] = QGridLayout()

        # camera view
        controls['figs']['2d'] = []
        controls['axes']['2d'] = []
        controls['frames']['views2d'] = QFrame()
        controls['grids']['views2d'] = QGridLayout()
        controls['plots']['image2d']: Optional[AxesImage] = None

        # sketch view
        controls['figs']['sketch'] = Figure()
        controls['canvases']['sketch'] = FigureCanvasQTAgg(controls['figs']['sketch'])
        controls['labels']['sketch'] = []
        controls['texts']['sketch'] = None

        # sketch zoom view
        controls['labels']['sketch_zoom'] = []

        controls_layout_grid = QGridLayout()
        row = 0
        col = 0

        button_load_recording = QPushButton()
        if self.recordingIsLoaded:
            button_load_recording.setStyleSheet("background-color: green;")
        else:
            button_load_recording.setStyleSheet("background-color: darkred;")
        button_load_recording.setText('Load Recording')
        button_load_recording.clicked.connect(self.button_load_recording_press)
        controls_layout_grid.addWidget(button_load_recording, row, col)
        button_load_recording.setEnabled(self.cfg['button_loadRecording'])
        controls['buttons']['load_recording'] = button_load_recording
        col = col + 1

        button_load_model = QPushButton()
        if self.modelIsLoaded:
            button_load_model.setStyleSheet("background-color: green;")
        else:
            button_load_model.setStyleSheet("background-color: darkred;")
        button_load_model.setText('Load Model')
        button_load_model.clicked.connect(self.button_load_model_press)
        controls_layout_grid.addWidget(button_load_model, row, col)
        button_load_model.setEnabled(self.cfg['button_loadModel'])
        controls['buttons']['load_model'] = button_load_model
        col = col + 1

        button_load_labels = QPushButton()
        if self.labelsAreLoaded:
            button_load_labels.setStyleSheet("background-color: green;")
        else:
            button_load_labels.setStyleSheet("background-color: darkred;")
        button_load_labels.setText('Load Labels')
        button_load_labels.clicked.connect(self.button_load_labels_press)
        controls_layout_grid.addWidget(button_load_labels, row, col)
        button_load_labels.setEnabled(self.cfg['button_loadLabels'])
        controls['buttons']['load_labels'] = button_load_labels
        row = row + 1
        col = 0

        button_load_calibration = QPushButton()
        if self.calibrationIsLoaded:
            button_load_calibration.setStyleSheet("background-color: green;")
        else:
            button_load_calibration.setStyleSheet("background-color: darkred;")
        button_load_calibration.setText('Load Calibration')
        button_load_calibration.clicked.connect(self.button_load_calibration_press)
        controls_layout_grid.addWidget(button_load_calibration, row, col)
        button_load_calibration.setEnabled(self.cfg['button_loadCalibration'])
        controls['buttons']['load_calibration'] = button_load_calibration
        col = col + 1

        button_save_model = QPushButton()
        button_save_model.setText('Save Model')
        button_save_model.clicked.connect(self.button_save_model_press)
        controls_layout_grid.addWidget(button_save_model, row, col)
        button_save_model.setEnabled(self.cfg['button_saveModel'])
        controls['buttons']['save_model'] = button_save_model
        col = col + 1

        button_save_labels = QPushButton()
        button_save_labels.setText('Save Labels (S)')
        button_save_labels.clicked.connect(self.button_save_labels_press)
        controls_layout_grid.addWidget(button_save_labels, row, col)
        button_save_labels.setEnabled(self.cfg['button_saveLabels'])
        controls['buttons']['save_labels'] = button_save_labels
        row = row + 1
        col = 0

        button_load_origin = QPushButton()
        if self.originIsLoaded:
            button_load_origin.setStyleSheet("background-color: green;")
        else:
            button_load_origin.setStyleSheet("background-color: darkred;")
        button_load_origin.setText('Load Origin')
        controls_layout_grid.addWidget(button_load_origin, row, col)
        button_load_origin.clicked.connect(self.button_load_origin_press)
        button_load_origin.setEnabled(self.cfg['button_loadOrigin'])
        controls['buttons']['load_origin'] = button_load_origin
        col = col + 1

        button_load_sketch = QPushButton()
        if self.sketchIsLoaded:
            button_load_sketch.setStyleSheet("background-color: green;")
        else:
            button_load_sketch.setStyleSheet("background-color: darkred;")
        button_load_sketch.setText('Load Sketch')
        controls_layout_grid.addWidget(button_load_sketch, row, col)
        button_load_sketch.clicked.connect(self.button_load_sketch_press)
        button_load_sketch.setEnabled(self.cfg['button_loadSketch'])
        controls['buttons']['load_sketch'] = button_load_sketch
        col = col + 1

        button_sketch_mode = QPushButton()
        button_sketch_mode.setStyleSheet("background-color: darkred;")
        button_sketch_mode.setText('Sketch Mode')
        controls_layout_grid.addWidget(button_sketch_mode, row, col)
        button_sketch_mode.clicked.connect(self.button_sketch_mode_press)
        button_sketch_mode.setEnabled(self.cfg['button_sketchMode'])
        controls['buttons']['sketch_mode'] = button_sketch_mode
        controls['status']['button_sketchMode'] = False
        row = row + 1
        col = 0

        button_fast_labeling_mode = QPushButton()
        button_fast_labeling_mode.setStyleSheet("background-color: darkred;")
        button_fast_labeling_mode.setText('Fast Labeling Mode')
        button_fast_labeling_mode.clicked.connect(self.button_fast_labeling_mode_press)
        button_fast_labeling_mode.setSizePolicy(QSizePolicy.Expanding,
                                                QSizePolicy.Preferred)
        controls_layout_grid.addWidget(button_fast_labeling_mode, row, col, 2, 1)
        button_fast_labeling_mode.setEnabled(self.cfg['button_fastLabelingMode'])
        controls['buttons']['fast_labeling_mode'] = button_fast_labeling_mode
        controls['status']['button_fastLabelingMode'] = False
        col = col + 1

        list_fast_labeling_mode = QComboBox()
        list_fast_labeling_mode.addItems([str(i) for i in range(len(self.cameras))])
        list_fast_labeling_mode.setSizePolicy(QSizePolicy.Expanding,
                                              QSizePolicy.Preferred)
        controls_layout_grid.addWidget(list_fast_labeling_mode, row, col, 2, 2)
        # list_fast_labeling_mode.currentIndexChanged.connect(self.list_fast_labeling_mode_change)
        # list_fast_labeling_mode.setEnabled(self.cfg['list_fastLabelingMode'])
        controls['lists']['fast_labeling_mode'] = list_fast_labeling_mode
        row = row + 2
        col = 0

        button_centric_view_mode = QPushButton()
        button_centric_view_mode.setStyleSheet("background-color: darkred;")
        button_centric_view_mode.setText('Centric View Mode (C)')
        button_centric_view_mode.setSizePolicy(QSizePolicy.Expanding,
                                               QSizePolicy.Preferred)
        controls_layout_grid.addWidget(button_centric_view_mode, row, col, 2, 1)
        button_centric_view_mode.clicked.connect(self.button_centric_view_mode_press)
        button_centric_view_mode.setEnabled(self.cfg['button_centricViewMode'])
        controls['buttons']['centric_view_mode'] = button_centric_view_mode
        controls['status']['button_centricViewMode'] = False
        col = col + 1

        label_dx = QLabel()
        label_dx.setText('dx:')
        controls_layout_grid.addWidget(label_dx, row, col)
        controls['labels']['dx'] = label_dx
        col = col + 1

        label_dy = QLabel()
        label_dy.setText('dy:')
        controls_layout_grid.addWidget(label_dy, row, col)
        controls['labels']['dy'] = label_dy
        row = row + 1
        col = 1

        field_dx = QLineEdit()
        field_dx.setValidator(QIntValidator())
        field_dx.setText(str(self.dx))
        controls_layout_grid.addWidget(field_dx, row, col)
        field_dx.returnPressed.connect(self.field_dx_change)
        field_dx.setEnabled(self.cfg['field_dx'])
        controls['fields']['dx'] = field_dx
        col = col + 1

        field_dy = QLineEdit()
        field_dy.setValidator(QIntValidator())
        field_dy.setText(str(self.dy))
        controls_layout_grid.addWidget(field_dy, row, col)
        field_dy.returnPressed.connect(self.field_dy_change)
        field_dy.setEnabled(self.cfg['field_dy'])
        controls['fields']['dy'] = field_dy
        row = row + 1
        col = 0

        button_reprojection_mode = QPushButton()
        button_reprojection_mode.setStyleSheet("background-color: darkred;")
        button_reprojection_mode.setText('Reprojection Mode')
        button_reprojection_mode.setSizePolicy(QSizePolicy.Expanding,
                                               QSizePolicy.Preferred)
        controls_layout_grid.addWidget(button_reprojection_mode, row, col, 2, 1)
        button_reprojection_mode.clicked.connect(self.button_reprojection_mode_press)
        button_reprojection_mode.setEnabled(self.cfg['button_reprojectionMode'])
        controls['buttons']['reprojection_mode'] = button_reprojection_mode
        controls['status']['button_reprojectionMode'] = False
        col = col + 1

        label_vmin = QLabel()
        label_vmin.setText('vmin:')
        controls_layout_grid.addWidget(label_vmin, row, col)
        controls['labels']['vmin'] = label_vmin
        col = col + 1

        label_vmax = QLabel()
        label_vmax.setText('vmax:')
        controls_layout_grid.addWidget(label_vmax, row, col)
        controls['labels']['vmax'] = label_vmax
        row = row + 1
        col = 0

        col = col + 1

        field_vmin = QLineEdit()
        field_vmin.setValidator(QIntValidator())
        field_vmin.setText(str(self.vmin))
        controls_layout_grid.addWidget(field_vmin, row, col)
        field_vmin.returnPressed.connect(self.field_vmin_change)
        field_vmin.setEnabled(self.cfg['field_vmin'])
        controls['fields']['vmin'] = field_vmin
        col = col + 1

        field_vmax = QLineEdit()
        field_vmax.setValidator(QIntValidator())
        field_vmax.setText(str(self.vmax))
        controls_layout_grid.addWidget(field_vmax, row, col)
        field_vmax.returnPressed.connect(self.field_vmax_change)
        field_vmax.setEnabled(self.cfg['field_vmax'])
        controls['fields']['vmax'] = field_vmax
        row = row + 1
        col = 0

        field_labels3d = QLineEdit()
        field_labels3d.returnPressed.connect(self.button_insert_press)
        field_labels3d.setSizePolicy(QSizePolicy.Expanding,
                                     QSizePolicy.Preferred)
        controls_layout_grid.addWidget(field_labels3d, row, col, 1, 1)
        field_labels3d.setEnabled(self.cfg['field_labels3d'])
        controls['fields']['labels3d'] = field_labels3d
        col = col + 1

        list_labels = QListWidget()
        # list_labels.setSortingEnabled(True)
        list_labels.addItems(list(self.labels['labels'].keys()))
        list_labels.setSelectionMode(QAbstractItemView.SingleSelection)
        list_labels.itemClicked.connect(self.list_labels_select)
        field_labels3d.setSizePolicy(QSizePolicy.Expanding,
                                     QSizePolicy.Preferred)
        controls_layout_grid.addWidget(list_labels, row, col, 3, 2)
        controls['lists']['labels'] = list_labels
        row = row + 1
        col = 0

        button_insert = QPushButton()
        button_insert.setText('Insert')
        button_insert.clicked.connect(self.button_insert_press)
        controls_layout_grid.addWidget(button_insert, row, col)
        button_insert.setEnabled(self.cfg['button_insert'])
        controls['buttons']['insert'] = button_insert
        row = row + 1

        button_remove = QPushButton()
        button_remove.setText('Remove')
        button_remove.clicked.connect(self.button_remove_press)
        controls_layout_grid.addWidget(button_remove, row, col)
        button_remove.setEnabled(self.cfg['button_remove'])
        controls['buttons']['remove'] = button_remove
        row = row + 1

        button_label3d = QPushButton()
        controls['status']['button_label3d'] = False
        button_label3d.setStyleSheet("background-color: darkred;")
        button_label3d.setText('Label 3D')
        # button_label3d.clicked.connect(self.button_label3d_press)
        controls_layout_grid.addWidget(button_label3d, row, col)
        button_label3d.setEnabled(self.cfg['button_label3d'])
        controls['buttons']['label3d'] = button_label3d
        controls['status']['button_label3d'] = False
        col = col + 1

        button_previous_label = QPushButton()
        button_previous_label.setText('Previous Label (P)')
        button_previous_label.clicked.connect(self.button_previous_label_press)
        controls_layout_grid.addWidget(button_previous_label, row, col)
        button_previous_label.setEnabled(self.cfg['button_previousLabel'])
        controls['buttons']['previous_label'] = button_previous_label
        col = col + 1

        button_next_label = QPushButton()
        button_next_label.setText('Next Label (N)')
        button_next_label.clicked.connect(self.button_next_label_press)
        controls_layout_grid.addWidget(button_next_label, row, col)
        button_next_label.setEnabled(self.cfg['button_nextLabel'])
        controls['buttons']['next_label'] = button_next_label
        row = row + 1
        col = 0

        button_up = QPushButton()
        button_up.setText('Up (\u2191)')
        button_up.clicked.connect(self.button_up_press)
        controls_layout_grid.addWidget(button_up, row, col)
        button_up.setEnabled(self.cfg['button_up'])
        controls['buttons']['up'] = button_up
        col = col + 1

        button_right = QPushButton()
        button_right.setText('Right (\u2192)')
        button_right.clicked.connect(self.button_right_press)
        controls_layout_grid.addWidget(button_right, row, col)
        button_right.setEnabled(self.cfg['button_right'])
        controls['buttons']['right'] = button_right
        col = col + 1

        label_dxyz = QLabel()
        label_dxyz.setText('dxyz:')
        controls_layout_grid.addWidget(label_dxyz, row, col)
        controls['labels']['dxyz'] = label_dxyz
        row = row + 1
        col = 0

        button_down = QPushButton()
        button_down.setText('Down (\u2193)')
        button_down.clicked.connect(self.button_down_press)
        controls_layout_grid.addWidget(button_down, row, col)
        button_down.setEnabled(self.cfg['button_down'])
        controls['buttons']['down'] = button_down
        col = col + 1

        button_left = QPushButton()
        button_left.setText('Left (\u2190)')
        button_left.clicked.connect(self.button_left_press)
        controls_layout_grid.addWidget(button_left, row, col)
        button_left.setEnabled(self.cfg['button_left'])
        controls['buttons']['left'] = button_left
        col = col + 1

        field_dxyz = QLineEdit()
        field_dxyz.setText(str(self.dxyz))
        field_dxyz.returnPressed.connect(self.field_dxyz_change)
        controls_layout_grid.addWidget(field_dxyz, row, col)
        field_dxyz.setEnabled(self.cfg['field_dxyz'])
        controls['fields']['dxyz'] = field_dxyz
        row = row + 1
        col = 0

        button_previous = QPushButton()
        button_previous.setText('Previous Frame (A)')
        button_previous.clicked.connect(self.button_previous_press)
        controls_layout_grid.addWidget(button_previous, row, col)
        button_previous.setEnabled(self.cfg['button_previous'])
        controls['buttons']['previous'] = button_previous
        col = col + 1

        button_next = QPushButton()
        button_next.setText('Next Frame (D)')
        button_next.clicked.connect(self.button_next_press)
        controls_layout_grid.addWidget(button_next, row, col)
        button_next.setEnabled(self.cfg['button_next'])
        controls['buttons']['next'] = button_next
        col = col + 1

        button_home = QPushButton('Home (H)')
        button_home.clicked.connect(self.button_home_press)
        controls_layout_grid.addWidget(button_home, row, col)
        button_home.setEnabled(self.cfg['button_home'])
        controls['buttons']['home'] = button_home
        row = row + 1
        col = 0

        label_current_pose = QLabel()
        label_current_pose.setText('current frame:')
        controls_layout_grid.addWidget(label_current_pose, row, col)
        controls['labels']['current_pose'] = label_current_pose
        col = col + 1

        label_d_frame = QLabel()
        label_d_frame.setText('dFrame:')
        controls_layout_grid.addWidget(label_d_frame, row, col)
        controls['labels']['d_frame'] = label_d_frame
        row = row + 1
        col = 0

        field_current_pose = QLineEdit()
        field_current_pose.setValidator(QIntValidator())
        field_current_pose.setText(str(self.get_pose_idx()))
        field_current_pose.returnPressed.connect(self.field_current_pose_change)
        controls_layout_grid.addWidget(field_current_pose, row, col)
        field_current_pose.setEnabled(self.cfg['field_currentPose'])
        controls['fields']['current_pose'] = field_current_pose
        col = col + 1

        field_d_frame = QLineEdit()
        field_d_frame.setValidator(QIntValidator())
        field_d_frame.setText(str(self.dFrame))
        field_d_frame.returnPressed.connect(self.field_d_frame_change)
        controls_layout_grid.addWidget(field_d_frame, row, col)
        field_d_frame.setEnabled(self.cfg['field_dFrame'])
        controls['fields']['d_frame'] = field_d_frame
        row = row + 1
        col = 0

        button_zoom = QPushButton('Zoom (Z)')
        button_zoom.setStyleSheet("background-color: darkred;")
        button_zoom.clicked.connect(self.button_zoom_press)
        controls_layout_grid.addWidget(button_zoom, row, col)
        button_zoom.setEnabled(self.cfg['button_zoom'])
        controls['buttons']['zoom'] = button_zoom
        col = col + 1

        button_pan = QPushButton('Pan (W)')
        button_pan.setStyleSheet("background-color: darkred;")
        button_pan.clicked.connect(self.button_pan_press)
        controls_layout_grid.addWidget(button_pan, row, col)
        controls['buttons']['pan'] = button_pan
        col = col + 1

        button_rotate = QPushButton('Rotate (R)')
        button_rotate.clicked.connect(self.button_rotate_press)
        controls_layout_grid.addWidget(button_rotate, row, col)
        controls['buttons']['rotate'] = button_rotate

        controls['frames']['controls'].setLayout(controls_layout_grid)

        self.controls = controls

    def button_load_recording_press(self):
        dialog = QFileDialog()
        dialog.setStyleSheet("background-color: white;")
        dialog_options = dialog.Options()
        dialog_options |= dialog.DontUseNativeDialog
        rec_file_names_unsorted, _ = QFileDialog.getOpenFileNames(dialog,
                                                                  "Choose recording files",
                                                                  "",
                                                                  "video files (*.ccv, *.mp4, *.mkv)",
                                                                  options=dialog_options)
        if len(rec_file_names_unsorted) > 0:
            rec_file_names = sorted(rec_file_names_unsorted)
            self.load_recordings_from_names(rec_file_names)

            self.set_pose_idx(0)

            self.plot2d_ini()

            self.controls['buttons']['load_recording'].setStyleSheet("background-color: green;")
            print('Loaded recording:')
            for i_rec in rec_file_names:
                print(i_rec)

        self.controls['buttons']['load_recording'].clearFocus()

    def button_load_calibration_press(self):
        dialog = QFileDialog()
        dialog.setStyleSheet("background-color: white;")
        dialog_options = dialog.Options()
        dialog_options |= dialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(dialog,
                                                   "Choose calibration file",
                                                   ""
                                                   "npy files (*.npy)",
                                                   options=dialog_options)
        if file_name:
            self.load_calibrations(Path(file_name))
            self.controls['buttons']['load_calibration'].setStyleSheet("background-color: green;")
            print('Loaded calibration ({:s})'.format(file_name))
        self.controls['buttons']['load_calibration'].clearFocus()

    def button_load_origin_press(self):
        dialog = QFileDialog()
        dialog.setStyleSheet("background-color: white;")
        dialog_options = dialog.Options()
        dialog_options |= dialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(dialog,
                                                   "Choose origin file",
                                                   ""
                                                   "npy files (*.npy)",
                                                   options=dialog_options)
        if file_name:
            self.load_origin(Path(file_name))
        self.controls['buttons']['load_origin'].clearFocus()

    def button_load_model_press(self):
        return

    def button_save_model_press(self):
        if self.modelIsLoaded:
            dialog = QFileDialog()
            dialog.setStyleSheet("background-color: white;")
            dialog_options = dialog.Options()
            dialog_options |= dialog.DontUseNativeDialog
            file_name, _ = QFileDialog.getSaveFileName(dialog,
                                                       "Save model file",
                                                       ""
                                                       "npy files (*.npy)",
                                                       options=dialog_options)
            if file_name:
                self.model['labels3d'] = copy.deepcopy(self.labels3d)
                np.save(file_name, self.model)
                print('Saved model ({:s})'.format(file_name))
        else:
            print('WARNING: Model needs to be loaded first')
        self.controls['buttons']['save_model'].clearFocus()

    def button_load_labels_press(self):
        dialog = QFileDialog()
        dialog.setStyleSheet("background-color: white;")
        dialog_options = dialog.Options()
        dialog_options |= dialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(dialog,
                                                   "Choose labels file",
                                                   ""
                                                   "npz files (*.npz)",
                                                   options=dialog_options)
        if file_name:
            # self.labels2d_all = np.load(fileName, allow_pickle=True)[()]
            self.labels2d_all = np.load(file_name, allow_pickle=True)['arr_0'].item()
            self.standardLabelsFile = file_name
            if self.get_pose_idx() in self.labels2d_all.keys():
                self.labels2d = copy.deepcopy(self.labels2d_all[self.get_pose_idx()])
            if self.controls['status']['label3d_select']:
                if self.controls['lists']['labels'].currentItem().text() in self.labels2d.keys():
                    self.selectedLabel2d = np.copy(
                        self.labels2d[self.controls['lists']['labels'].currentItem().text()])

            self.labelsAreLoaded = True
            self.controls['buttons']['load_labels'].setStyleSheet("background-color: green;")
            print('Loaded labels ({:s})'.format(file_name))
        self.controls['buttons']['load_labels'].clearFocus()

    def button_save_labels_press(self):
        if self.master:
            dialog = QFileDialog()
            dialog.setStyleSheet("background-color: white;")
            dialog_options = dialog.Options()
            dialog_options |= dialog.DontUseNativeDialog
            print(self.standardLabelsFile)
            file_name, _ = QFileDialog.getSaveFileName(dialog,
                                                       "Save labels file",
                                                       os.path.dirname(self.standardLabelsFile),
                                                       "npz files (*.npz)",
                                                       options=dialog_options)
            if file_name:
                if bool(self.labels2d):
                    self.labels2d_all[self.get_pose_idx()] = copy.deepcopy(self.labels2d)
                # np.save(fileName, self.labels2d_all)
                np.savez(file_name, self.labels)
                print('Saved labels ({:s})'.format(file_name))
        else:
            if bool(self.labels2d):
                self.labels2d_all[self.get_pose_idx()] = copy.deepcopy(self.labels2d)
            # np.save(self.standardLabelsFile, self.labels2d_all)
            np.savez(self.standardLabelsFile, self.labels)
            print(f'Saved labels ({self.standardLabelsFile})')
        self.controls['buttons']['save_labels'].clearFocus()

    def button_load_sketch_press(self):
        dialog = QFileDialog()
        dialog.setStyleSheet("background-color: white;")
        dialog_options = dialog.Options()
        dialog_options |= dialog.DontUseNativeDialog
        file_name_sketch, _ = QFileDialog.getOpenFileName(dialog,
                                                          "Choose sketch file",
                                                          ""
                                                          "npy files (*.npy)",
                                                          options=dialog_options)

        self.load_sketch(Path(file_name_sketch))
        if self.sketchIsLoaded:
            self.controls['buttons']['load_sketch'].setStyleSheet("background-color: green;")
            print('Loaded sketch ({:s})'.format(file_name_sketch))

        self.controls['buttons']['load_sketch'].clearFocus()

    def button_sketch_mode_press(self):
        if self.controls['buttons']['fast_labeling_mode']:
            if self.modelIsLoaded:
                if self.controls['status']['button_sketchMode']:
                    self.controls['buttons']['sketch_mode'].setStyleSheet("background-color: darkred;")
                    self.controls['status']['button_sketchMode'] = not self.controls['status']['button_sketchMode']
                    # self.plot3d_draw()
                    self.controls['figs']['sketch'].canvas.mpl_disconnect(self.cidSketch)
                else:
                    print('WARNING: Model needs to be loaded first')
            elif self.sketchIsLoaded:
                if True:
                    self.controls['buttons']['sketch_mode'].setStyleSheet("background-color: green;")
                    self.controls['status']['button_sketchMode'] = not self.controls['status']['button_sketchMode']
                    self.sketch_init()
                    self.cidSketch = self.controls['canvases']['sketch'].mpl_connect('button_press_event',
                                                                                     lambda event: self.sketch_click(
                                                                                         event))
            else:
                print('WARNING: Sketch or model needs to be loaded first')
        else:
            print('WARNING: "Fast Labeling Mode" needs to be enabled to activate "Sketch Mode"')
        self.controls['buttons']['sketch_mode'].clearFocus()

    def button_insert_press(self):
        if ((not (np.any(np.isnan(self.clickedLabel3d)))) &
                (self.controls['fields']['labels3d'].text() != '')):
            if self.controls['fields']['labels3d'].text() in self.labels3d:
                print('WARNING: Label already exists')
            else:
                self.labels3d[self.controls['fields']['labels3d'].text()] = copy.deepcopy(self.clickedLabel3d)
                self.controls['lists']['labels'].addItems([self.controls['fields']['labels3d'].text()])
                self.clickedLabel3d = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
                self.controls['fields']['labels3d'].setText(str(''))
        self.controls['buttons']['insert'].clearFocus()
        self.controls['fields']['labels3d'].clearFocus()

    def button_remove_press(self):
        if self.controls['lists']['labels'].currentItem() is not None:
            self.controls['status']['label3d_select'] = False
            self.selectedLabel3d = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            self.selectedLabel2d = np.full((len(self.cameras), 2), np.nan, dtype=np.float64)
            self.clickedLabel2d = np.array([np.nan, np.nan], dtype=np.float64)
            del (self.labels3d[self.controls['lists']['labels'].currentItem().text()])
            self.controls['lists']['labels'].takeItem(self.controls['lists']['labels'].currentRow())
            self.plot2d_update()
        self.controls['buttons']['remove'].clearFocus()

    def list_labels_select(self):
        self.trigger_autosave_event()
        self.plot2d_draw_fast()
        self.plot2d_update()
        self.sketch_update()
        # self.plot3d_update()
        self.controls['lists']['labels'].clearFocus()

    def trigger_autosave_event(self):
        if self.cfg['autoSave'] and not self.master:
            self.autoSaveCounter = self.autoSaveCounter + 1
            if np.mod(self.autoSaveCounter, self.cfg['autoSaveN0']) == 0:
                if bool(self.labels2d):
                    self.labels2d_all[self.get_pose_idx()] = copy.deepcopy(self.labels2d)
                # file = self.standardLabelsFolder / 'labels.npy' # this is equal to self.standardLabelsFile
                # np.save(file, self.labels2d_all)
                # print('Automatically saved labels ({:s})'.format(file.as_posix()))
                file = self.standardLabelsFolder / 'labels.npz'  # this is equal to self.standardLabelsFile
                np.savez(file, self.labels)
                print('Automatically saved labels ({:s})'.format(file.as_posix()))
            if np.mod(self.autoSaveCounter, self.cfg['autoSaveN1']) == 0:
                if bool(self.labels2d):
                    self.labels2d_all[self.get_pose_idx()] = copy.deepcopy(self.labels2d)
                # file = self.standardLabelsFolder / 'autosave' / 'labels.npy'
                # np.save(file, self.labels2d_all)
                # print('Automatically saved labels ({:s})'.format(file.as_posix()))
                file = self.standardLabelsFolder / 'autosave' / 'labels.npz'
                np.savez(file, self.labels)
                print('Automatically saved labels ({:s})'.format(file.as_posix()))
                #
                self.autoSaveCounter = 0

    def button_fast_labeling_mode_press(self):
        return

    def button_centric_view_mode_press(self):
        if self.controls['status']['button_fastLabelingMode']:
            if not (self.controls['lists']['labels'].currentItem() is None):
                self.controls['status']['button_centricViewMode'] = not self.controls['status'][
                    'button_centricViewMode']
                #                 self.plot2d_drawFast_ini()
                self.plot2d_draw_fast()
                if not self.controls['status']['button_centricViewMode']:
                    self.controls['buttons']['centric_view_mode'].setStyleSheet("background-color: darkred;")
                else:
                    self.controls['buttons']['centric_view_mode'].setStyleSheet("background-color: green;")
            else:
                print('WARNING: A label needs to be selected to activate "Centric View Mode"')
        else:
            print('WARNING: "Fast Labeling Mode" needs to be enabled to activate "Centric View Mode"')
        self.controls['buttons']['centric_view_mode'].clearFocus()

    def button_reprojection_mode_press(self):
        if self.calibrationIsLoaded:
            self.controls['status']['button_reprojectionMode'] = not self.controls['status']['button_reprojectionMode']
            if self.controls['status']['button_reprojectionMode']:
                self.controls['buttons']['reprojection_mode'].setStyleSheet("background-color: green;")
            else:
                self.controls['buttons']['reprojection_mode'].setStyleSheet("background-color: darkred;")
            self.plot2d_update()
        else:
            print('WARNING: Calibration needs to be loaded first')
        self.controls['buttons']['reprojection_mode'].clearFocus()

    def field_dx_change(self):
        try:
            int(self.controls['fields']['dx'].text())
            field_input_is_correct = True
        except ValueError:
            field_input_is_correct = False
        if field_input_is_correct:
            self.dx = int(np.max([8, int(self.controls['fields']['dx'].text())]))
        self.controls['fields']['dx'].setText(str(self.dx))
        if self.controls['status']['button_fastLabelingMode']:
            self.plot2d_draw_fast()
        self.controls['fields']['dx'].clearFocus()

    def field_dy_change(self):
        try:
            int(self.controls['fields']['dy'].text())
            field_input_is_correct = True
        except ValueError:
            field_input_is_correct = False
        if field_input_is_correct:
            self.dy = int(np.max([8, int(self.controls['fields']['dy'].text())]))
        self.controls['fields']['dy'].setText(str(self.dy))
        if self.controls['status']['button_fastLabelingMode']:
            self.plot2d_draw_fast()
        self.controls['fields']['dy'].clearFocus()

    def field_vmin_change(self):
        try:
            int(self.controls['fields']['vmin'].text())
            field_input_is_correct = True
        except ValueError:
            field_input_is_correct = False
        if field_input_is_correct:
            self.vmin = int(self.controls['fields']['vmin'].text())
            self.vmin = int(np.max([0, self.vmin]))
            self.vmin = int(np.min([self.vmin, 254]))
            self.vmin = int(np.min([self.vmin, self.vmax - 1]))
        else:
            for i_cam in range(len(self.cameras)):
                self.controls['plots']['image2d'].set_clim(self.vmin, self.vmax)
                self.controls['figs']['2d'].canvas.draw()
        self.controls['fields']['vmin'].setText(str(self.vmin))
        self.controls['fields']['vmin'].clearFocus()

    def field_vmax_change(self):
        try:
            int(self.controls['fields']['vmax'].text())
            field_input_is_correct = True
        except ValueError:
            field_input_is_correct = False
        if field_input_is_correct:
            self.vmax = int(self.controls['fields']['vmax'].text())
            self.vmax = int(np.max([1, self.vmax]))
            self.vmax = int(np.min([self.vmax, 255]))
            self.vmax = int(np.max([self.vmin + 1, self.vmax]))
        if self.controls['status']['button_fastLabelingMode']:
            self.controls['plots']['image2d'].set_clim(self.vmin, self.vmax)
            self.controls['figs']['2d'].canvas.draw()
        else:
            for i_cam in range(len(self.cameras)):
                self.controls['plots']['image2d'].set_clim(self.vmin, self.vmax)
                self.controls['figs']['2d'].canvas.draw()
        self.controls['fields']['vmax'].setText(str(self.vmax))
        self.controls['fields']['vmax'].clearFocus()

    def button_home_press(self):
        if self.recordingIsLoaded:
            x_res = self.get_x_res()
            y_res = self.get_y_res()

            for i_cam in range(len(self.cameras)):
                self.cameras[i_cam]['x_lim_prev'] = np.array([0.0, x_res[i_cam] - 1], dtype=np.float64)
                self.cameras[i_cam]['y_lim_prev'] = np.array([0.0, y_res[i_cam] - 1], dtype=np.float64)

            self.plot2d_draw_fast()

            for i in self.toolbars:
                i.home()
        # if self.modelIsLoaded:
        #     v, f, v_center = self.get_model_v_f_vc()
        #
        #     self.controls['axes']['3d'].mouse_init()
        #     self.controls['axes']['3d'].view_init(elev=None, azim=None)
        #     self.controls['axes']['3d'].set_xlim([v_center[0] - self.dxyz_lim,
        #                                           v_center[0] + self.dxyz_lim])
        #     self.controls['axes']['3d'].set_ylim([v_center[1] - self.dxyz_lim,
        #                                           v_center[1] + self.dxyz_lim])
        #     self.controls['axes']['3d'].set_zlim([v_center[2] - self.dxyz_lim,
        #                                           v_center[2] + self.dxyz_lim])
        #     self.controls['canvases']['3d'].draw()
        self.controls['buttons']['home'].clearFocus()

    def button_zoom_press(self, tostate=None):
        if tostate is None or tostate is False:
            tostate = ["on", "off"]

        if self.controls['status']['toolbars_zoom'] and "off" not in tostate:
            return
        if not self.controls['status']['toolbars_zoom'] and "on" not in tostate:
            return

        if self.recordingIsLoaded:
            if not self.controls['status']['toolbars_zoom']:
                self.controls['buttons']['zoom'].setStyleSheet("background-color: green;")
            else:
                self.controls['buttons']['zoom'].setStyleSheet("background-color: darkred;")
            if self.controls['status']['toolbars_pan']:
                self.button_pan_press()
            for i in self.toolbars:
                i.zoom()
            self.controls['status']['toolbars_zoom'] = not self.controls['status']['toolbars_zoom']
        else:
            print('WARNING: Recording needs to be loaded first')
        self.controls['buttons']['zoom'].clearFocus()

    def button_rotate_press(self):
        if self.recordingIsLoaded:
            self.cameras[self.i_cam]["rotate"] = not self.cameras[self.i_cam]["rotate"]

            self.plot2d_draw_fast()
        else:
            print('WARNING: Recording needs to be loaded first')
        self.controls['buttons']['pan'].clearFocus()

    def button_pan_press(self):
        if self.recordingIsLoaded:
            if not self.controls['status']['toolbars_pan']:
                self.controls['buttons']['pan'].setStyleSheet("background-color: green;")
            else:
                self.controls['buttons']['pan'].setStyleSheet("background-color: darkred;")
            if self.controls['status']['toolbars_zoom']:
                self.button_zoom_press()
            for i in self.toolbars:
                i.pan()
            self.selectedLabel2d = np.full((len(self.cameras), 2), np.nan, dtype=np.float64)
            self.controls['status']['toolbars_pan'] = not self.controls['status']['toolbars_pan']
        else:
            print('WARNING: Recording needs to be loaded first')
        self.controls['buttons']['pan'].clearFocus()

    def sketch_click(self, event):
        if event.button == 1:
            x = event.xdata
            y = event.ydata
            if (x is not None) & (y is not None):
                label_coordinates = self.get_sketch_label_coordinates()
                dists = ((x - label_coordinates[:, 0]) ** 2 + (y - label_coordinates[:, 1]) ** 2) ** 0.5
                label_index = np.argmin(dists)
                self.controls['lists']['labels'].setCurrentRow(label_index)
                self.list_labels_select()

    # this works correctly but implementation is somewhat messed up
    # (rows are dimensions and not the columns, c.f. commented plotting command)
    def plot3d_move_center(self, move_direc):
        return
        # x_lim = self.controls['axes']['3d'].get_xlim()
        # y_lim = self.controls['axes']['3d'].get_ylim()
        # z_lim = self.controls['axes']['3d'].get_zlim()
        # center = np.array([np.mean(x_lim),
        #                    np.mean(y_lim),
        #                    np.mean(z_lim)], dtype=np.float64)
        # dxzy_lim = np.mean([np.abs(center[0] - x_lim),
        #                     np.abs(center[1] - y_lim),
        #                     np.abs(center[2] - z_lim)])
        #
        # azim = self.controls['axes']['3d'].azim / 180 * np.pi + np.pi
        # elev = self.controls['axes']['3d'].elev / 180 * np.pi
        #
        # r_azim = np.array([0.0, 0.0, -azim], dtype=np.float64)
        # R_azim = rodrigues2rotmat_single(r_azim)
        #
        # r_elev = np.array([0.0, -elev, 0.0], dtype=np.float64)
        # R_elev = rodrigues2rotmat_single(r_elev)
        #
        # R_azim_elev = np.dot(R_elev, R_azim)
        #
        # #        coord = center + R_azim_elev * 0.1
        # #        label = ['x', 'y', 'z']
        # #        for i in range(3):
        # #            self.controls['axes']['3d'].plot([center[0], coord[i, 0]],
        # #                           [center[1], coord[i, 1]],
        # #                           [center[2], coord[i, 2]],
        # #                           linestyle='-',
        # #                           color='green')
        # #            self.controls['axes']['3d'].text(coord[i, 0],
        # #                           coord[i, 1],
        # #                           coord[i, 2],
        # #                           label[i],
        # #                           color='red')
        #
        # center_new = center + np.sign(move_direc) * R_azim_elev[np.abs(move_direc), :] * self.dxyz
        # self.controls['axes']['3d'].set_xlim([center_new[0] - dxzy_lim,
        #                                       center_new[0] + dxzy_lim])
        # self.controls['axes']['3d'].set_ylim([center_new[1] - dxzy_lim,
        #                                       center_new[1] + dxzy_lim])
        # self.controls['axes']['3d'].set_zlim([center_new[2] - dxzy_lim,
        #                                       center_new[2] + dxzy_lim])
        # self.controls['canvases']['3d'].draw()

    def button_up_press(self):
        move_direc = +2
        self.plot3d_move_center(move_direc)
        self.controls['buttons']['up'].clearFocus()

    def button_down_press(self):
        move_direc = -2
        self.plot3d_move_center(move_direc)
        self.controls['buttons']['down'].clearFocus()

    def button_left_press(self):
        move_direc = +1
        self.plot3d_move_center(move_direc)
        self.controls['buttons']['left'].clearFocus()

    def button_right_press(self):
        move_direc = -1
        self.plot3d_move_center(move_direc)
        self.controls['buttons']['right'].clearFocus()

    def field_dxyz_change(self):
        try:
            float(self.controls['fields']['dxyz'].text())
            field_input_is_correct = True
        except ValueError:
            field_input_is_correct = False
        if field_input_is_correct:
            self.dxyz = float(self.controls['fields']['dxyz'].text())
        self.controls['fields']['dxyz'].setText(str(self.dxyz))
        self.controls['fields']['dxyz'].clearFocus()

    def button_next_label_press(self):
        self.change_label(1)

    def button_previous_label_press(self):
        self.change_label(-1)

    def change_label(self, d_label_idx):
        selected_label_name = self.get_current_label()
        selected_label_index = list(self.labels['labels'].keys()).index(selected_label_name)
        next_label_index = selected_label_index + d_label_idx
        if next_label_index >= np.size(list(self.labels['labels'].keys())):
            next_label_index = 0
        elif next_label_index < 0:
            next_label_index = np.size(list(self.labels['labels'].keys())) - 1
        self.controls['lists']['labels'].setCurrentRow(next_label_index)
        self.list_labels_select()
        self.controls['buttons']['next_label'].clearFocus()

    def button_next_press(self):
        self.set_pose_idx(self.get_pose_idx() + self.dFrame)
        self.plot2d_change_frame()
        self.controls['buttons']['next'].clearFocus()

    def button_previous_press(self):
        self.set_pose_idx(self.get_pose_idx() - self.dFrame)
        self.plot2d_change_frame()
        self.controls['buttons']['previous'].clearFocus()

    def field_current_pose_change(self):
        try:
            int(self.controls['fields']['current_pose'].text())
            field_input_is_correct = True
        except ValueError:
            field_input_is_correct = False
        if field_input_is_correct:
            if bool(self.labels2d):
                self.labels2d_all[self.get_pose_idx()] = copy.deepcopy(self.labels2d)

            self.set_pose_idx(self.controls['fields']['current_pose'].text())
            self.plot2d_change_frame()
        self.controls['fields']['current_pose'].setText(str(self.get_pose_idx()))

        first_label_name = list(self.labels['labels'].keys())[0]
        sorted_index = sorted(list(self.labels3d.keys())).index(first_label_name)
        self.controls['lists']['labels'].setCurrentRow(sorted_index)
        self.list_labels_select()
        self.sketch_update()
        self.controls['fields']['current_pose'].clearFocus()

    def field_d_frame_change(self):
        try:
            int(self.controls['fields']['d_frame'].text())
            field_input_is_correct = True
        except ValueError:
            field_input_is_correct = False
        if field_input_is_correct:
            self.dFrame = int(np.max([1, int(self.controls['fields']['d_frame'].text())]))
            x_res = self.get_x_res()
            y_res = self.get_y_res()

            for i_cam in range(len(self.cameras)):
                self.cameras[i_cam]['x_lim_prev'] = np.array([0.0, x_res[i_cam] - 1], dtype=np.float64)
                self.cameras[i_cam]['y_lim_prev'] = np.array([0.0, y_res[i_cam] - 1], dtype=np.float64)
        self.controls['fields']['d_frame'].setText(str(self.dFrame))
        self.controls['fields']['d_frame'].clearFocus()

    def plot2d_change_frame(self):
        for i_cam in range(len(self.cameras)):
            self.cameras[i_cam]['x_lim_prev'] = self.controls['axes']['2d'].get_xlim()
            self.cameras[i_cam]['y_lim_prev'] = self.controls['axes']['2d'].get_ylim()  # [::-1]

        self.controls['fields']['current_pose'].setText(str(self.get_pose_idx()))
        self.list_labels_select()
        self.sketch_update()

        self.plot2d_draw_fast()

    def closeEvent(self, event):
        if self.cfg['exitSaveModel']:
            self.button_save_model_press()
        if self.cfg['exitSaveLabels']:
            self.button_save_labels_press()

        if not self.master:
            exit_status = dict()
            exit_status['i_pose'] = self.get_pose_idx()
            np.save(self.standardLabelsFolder / 'exit_status.npy', exit_status)

    # shortkeys
    def keyPressEvent(self, event):
        if not (event.isAutoRepeat()):
            if self.cfg['button_next'] and event.key() == Qt.Key_D:
                self.button_next_press()
            elif self.cfg['button_previous'] and event.key() == Qt.Key_A:
                self.button_previous_press()
            elif self.cfg['button_nextLabel'] and event.key() == Qt.Key_N:
                self.button_next_label_press()
            elif self.cfg['button_previousLabel'] and event.key() == Qt.Key_P:
                self.button_previous_label_press()
            elif self.cfg['button_home'] and event.key() == Qt.Key_H:
                self.button_home_press()
            elif self.cfg['button_zoom'] and event.key() == Qt.Key_Z:
                self.button_zoom_press()
            elif self.cfg['button_pan'] and event.key() == Qt.Key_W:
                self.button_pan_press()
            elif self.cfg['button_pan'] and event.key() == Qt.Key_R:
                self.button_rotate_press()
            elif self.cfg['button_label3d'] and event.key() == Qt.Key_L:
                pass
                # self.button_label3d_press()
            elif self.cfg['button_up'] and event.key() == Qt.Key_Up:
                self.button_up_press()
            elif self.cfg['button_down'] and event.key() == Qt.Key_Down:
                self.button_down_press()
            elif self.cfg['button_left'] and event.key() == Qt.Key_Left:
                self.button_left_press()
            elif self.cfg['button_right'] and event.key() == Qt.Key_Right:
                self.button_right_press()
            elif self.cfg['button_centricViewMode'] and event.key() == Qt.Key_C:
                self.button_centric_view_mode_press()
            elif self.cfg['button_saveLabels'] and event.key() == Qt.Key_S:
                self.button_save_labels_press()
            elif self.cfg['field_vmax'] and event.key() == Qt.Key_Plus:
                self.controls['fields']['vmax'].setText(str(int(self.vmax * 0.8)))
                self.field_vmax_change()
            elif self.cfg['field_vmax'] and event.key() == Qt.Key_Minus:
                self.controls['fields']['vmax'].setText(str(int(self.vmax / 0.8)))
                self.field_vmax_change()
            elif event.key() == Qt.Key_1:
                if len(self.cameras) > 0:
                    self.controls['lists']['fast_labeling_mode'].setCurrentIndex(0)
            elif event.key() == Qt.Key_2:
                if len(self.cameras) > 1:
                    self.controls['lists']['fast_labeling_mode'].setCurrentIndex(1)
            elif event.key() == Qt.Key_3:
                if len(self.cameras) > 2:
                    self.controls['lists']['fast_labeling_mode'].setCurrentIndex(2)
            elif event.key() == Qt.Key_4:
                if len(self.cameras) > 3:
                    self.controls['lists']['fast_labeling_mode'].setCurrentIndex(3)
            elif event.key() == Qt.Key_5:
                if len(self.cameras) > 4:
                    self.controls['lists']['fast_labeling_mode'].setCurrentIndex(4)
            elif event.key() == Qt.Key_6:
                if len(self.cameras) > 5:
                    self.controls['lists']['fast_labeling_mode'].setCurrentIndex(5)

        else:
            print('WARNING: Auto-repeat is not supported')


# noinspection PyUnusedLocal
def main(drive: Path, config_file=None, master=True):
    app = QApplication(sys.argv)
    window = MainWindow(drive=drive, file_config=config_file, master=master)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main(Path('o:/analysis/'))


class UnsupportedFormatException(Exception):
    pass
