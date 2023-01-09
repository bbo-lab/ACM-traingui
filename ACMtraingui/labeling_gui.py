#!/usr/bin/env python3

import copy
import numpy as np
import os
import sys
import calibcamlib

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
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import imageio
from ccvtools import rawio

from .config import load_cfg,save_cfg

import warnings

def rodrigues2rotMat_single(r):
    theta = np.power(r[0]**2 + r[1]**2 + r[2]**2, 0.5)
    u = r / (theta + np.abs(np.sign(theta)) - 1.0)
    # row 1
    rotMat_00 = np.cos(theta) + u[0]**2 * (1.0 - np.cos(theta))
    rotMat_01 = u[0] * u[1] * (1.0 - np.cos(theta)) - u[2] * np.sin(theta)
    rotMat_02 = u[0] * u[2] * (1.0 - np.cos(theta)) + u[1] * np.sin(theta)

    # row 2
    rotMat_10 = u[0] * u[1] * (1.0 - np.cos(theta)) + u[2] * np.sin(theta)
    rotMat_11 = np.cos(theta) + u[1]**2 * (1.0 - np.cos(theta))
    rotMat_12 = u[1] * u[2] * (1.0 - np.cos(theta)) - u[0] * np.sin(theta)

    # row 3
    rotMat_20 = u[0] * u[2] * (1.0 - np.cos(theta)) - u[1] * np.sin(theta)
    rotMat_21 = u[1] * u[2] * (1.0 - np.cos(theta)) + u[0] * np.sin(theta)
    rotMat_22 = np.cos(theta) + u[2]**2 * (1.0 - np.cos(theta))

    rotMat = np.array([[rotMat_00, rotMat_01, rotMat_02],
                       [rotMat_10, rotMat_11, rotMat_12],
                       [rotMat_20, rotMat_21, rotMat_22]], dtype=np.float64)
    
    return rotMat

def calc_dst(m_udst, k):
    x_1 = m_udst[:, 0] / m_udst[:, 2]
    y_1 = m_udst[:, 1] / m_udst[:, 2]
    
    r2 = x_1**2 + y_1**2
    
    x_2 = x_1 * (1.0 + k[0] * r2 + k[1] * r2**2 + k[4] * r2**3) + \
          2.0 * k[2] * x_1 * y_1 + \
          k[3] * (r2 + 2.0 * x_1**2)
          
    y_2 = y_1 * (1.0 + k[0] * r2 + k[1] * r2**2 + k[4] * r2**3) + \
          k[2] * (r2 + 2.0 * y_1**2) + \
          2.0 * k[3] * x_1 * y_1
    
    nPoints = np.size(m_udst, 0)
    ones = np.ones(nPoints, dtype=np.float64)
    m_dst = np.concatenate([[x_2], [y_2], [ones]], 0).T
    return m_dst

def read_video_meta(reader):
    header = reader.get_meta_data()
    header['nFrames'] = len(reader) # len() may be Inf for formats where counting frames can be expensive
    if 1000000000000000<header['nFrames']:
        try:
            header['nFrames'] = reader.count_frames()
        except:
            print('Could not determine number of frames')
            raise UnsupportedFormatException

    # Add required headers that are not normally part of standard video formats but are required information
    if "sensor" in header:
        header['offset'] = tuple(header['sensor']['offset'])
        header['sensorsize'] = tuple(header['sensor']['size'])
    else:
        print("Infering sensor size from image and setting offset to 0!")
        header['sensorsize'] = tuple(reader.get_data(0).shape[0:2])
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
    nPoints = np.size(m_dst, 0)
    p = np.zeros(6, dtype=np.float64)
    p[4] = 1.0
    sol = np.zeros(3, dtype=np.float64)
    x_1 = np.zeros(nPoints, dtype=np.float64)
    y_1 = np.zeros(nPoints, dtype=np.float64)
    for i_point in range(nPoints):
        cond = (np.abs(x_2[i_point]) > np.abs(y_2[i_point]))
        if (cond):
            c = y_2[i_point] / x_2[i_point]
            p[5] = -x_2[i_point]
        else:
            c = x_2[i_point] / y_2[i_point]
            p[5] = -y_2[i_point]
#        p[4] = 1
        p[2] = k[0] * (1.0 + c**2)
        p[0] = k[1] * (1.0 + c**2)**2
        sol = np.real(np.roots(p))
        # use min(abs(x)) to make your model as accurate as possible
        sol_abs = np.abs(sol)
        if (cond):
            x_1[i_point] = sol[sol_abs == np.min(sol_abs)][0]
            y_1[i_point] = c * x_1[i_point]
        else:
            y_1[i_point] = sol[sol_abs == np.min(sol_abs)][0]
            x_1[i_point] = c * y_1[i_point]
    m_udst = np.concatenate([[x_1], [y_1], [m_dst[:, 2]]], 0).T
    return m_udst



# ATTENTION: hard coded
def sort_label_sequence(seq):
    try:
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
        labels_num_sorted = list([[] for i in num_order])
        labels_left_sorted = list([[] for i in left_right_order])
        labels_right_sorted = list([[] for i in left_right_order])
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
    except:
        return seq

    
class SelectUserWindow(QDialog):
    def __init__(self, parent=None, drive=[]):
        super(SelectUserWindow, self).__init__(parent)
        self.setGeometry(0, 0, 256, 128)
        self.center()
        self.setWindowTitle('Select User')
        
        self.drive = drive
        
        self.user_list = self.get_user_list()
        
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
    
    def get_user_list(self):
        user_list = sorted(os.listdir(self.drive + 'pose/data/user'))
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

    def start(parent=None,drive=[]):
        selecting = SelectUserWindow(parent,drive)
        exit = selecting.exec_()
        user = selecting.get_user()
        return (user, exit == QDialog.Accepted)

    
    
class MainWindow(QMainWindow):   
    def __init__(self, parent=None, fileConfig=None, master=True, drive=None):
        self.master = master
        self.cameraSystem = None
        self.label2d_max_err = []
        
        if drive is None:
            self.drive = 'O:/analysis/'
        else:
            self.drive = drive
        
        if self.master:
            if fileConfig is None:
                fileConfig = 'labeling_gui_cfg.py' # use hard coded path here
            self.cfg = load_cfg(fileConfig)
        else:
            if os.path.isdir(self.drive):
                self.user, correct_exit = SelectUserWindow.start(drive=self.drive)
                if correct_exit:
                    fileConfig = self.drive + 'pose/data/user/' + self.user + '/' + 'labeling_gui_cfg.py' # use hard coded path here
                    self.cfg = load_cfg(fileConfig)
                else:
                    sys.exit()
            else:
                print('ERROR: Server is not mounted')
                sys.exit()
                
        super(MainWindow, self).__init__(parent)    
        self.setGeometry(0, 0, 1024, 768)
        self.showMaximized()

        self.general_definitions()

        self.set_layout()
        self.set_controls()
        self.plot2d_drawNormal_ini()
        self.plot3d_draw()

        self.setFocus()
        self.setWindowTitle('Labeling GUI')
        self.show()

        if not(self.cfg['list_fastLabelingMode']):
            self.list_fastLabelingMode.setCurrentIndex(self.cfg['cam'])
        if (self.cfg['button_fastLabelingMode_activate']):
            self.button_fastLabelingMode_press()
        if (self.cfg['button_centricViewMode_activate']):
            self.button_centricViewMode_press()
        if (self.cfg['button_reprojectionMode_activate']):
            self.button_reprojectionMode_press()
        if (self.cfg['button_sketchMode_activate']):
            self.button_sketchMode_press()

            

    def general_definitions(self):          
        # general
        self.nCameras = int(0)
        
        self.dx = int(128)
        self.dy = int(128)
        self.vmin = int(0)
        self.vmax = int(127)
        self.dFrame = self.cfg['dFrame']

        self.minPose = self.cfg['minPose']
        self.maxPose = self.cfg['maxPose']
        self.i_pose = self.minPose
        
        self.i_cam = self.cfg['cam']
        
        self.toolbars_zoom_status = False
        self.toolbars_pan_status = False

        self.labels2d_all = dict()
        self.labels2d = dict()
        self.clickedLabel2d = np.array([np.nan, np.nan], dtype=np.float64)
        self.clickedLabel2d_pose = np.copy(self.i_pose)
        self.selectedLabel2d = np.full((self.nCameras, 2), np.nan, dtype=np.float64)
        

        
        self.clickedLabel3d = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        self.label3d_select_status = False
        self.selectedLabel3d = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        
        self.dxyz_lim = float(0.4)
        self.dxyz = float(0.01)
        self.labels3d = dict()
        
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        # Sort colors by hue, saturation, value and name.
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                        for name, color in colors.items())
        sorted_names = [name for hsv, name in by_hsv]    
        self.colors = list()
        for i in range(24, -1, -1):
            self.colors = self.colors + sorted_names[i::24]
        
        self.autoSaveCounter = int(0)
        
        # create folder structure / save backup / load last pose 
        if not(self.master):
            # folder structure
            standardRecordingFolder_split = self.cfg['standardRecordingFolder'].split('/')
            userfolder = self.drive + 'pose/user' + \
                     '/' + self.user
            if not(os.path.isdir(userfolder)):
                os.makedirs(userfolder)
                
            resultsfolder = userfolder + \
                     '/' + standardRecordingFolder_split[-1]
            if not(os.path.isdir(resultsfolder)):
                os.makedirs(resultsfolder)

            self.standardLabelsFolder = resultsfolder
            # backup
            backupfolder = self.standardLabelsFolder + '/' + 'backup'
            if not(os.path.isdir(backupfolder)):
                os.mkdir(backupfolder)
            file = self.standardLabelsFolder + '/' + 'labeling_gui_cfg.py'
            if os.path.isfile(file):
                cfg_old = load_cfg(file)
                save_cfg(backupfolder + '/' + 'labeling_gui_cfg.py', cfg_old)
            #file = self.standardLabelsFolder + '/' + 'labels.npy'
            file = self.standardLabelsFolder + '/' + 'labels.npz'
            if os.path.isfile(file):
                #labels_old = np.load(file, allow_pickle=True).item()
                labels_old = np.load(file, allow_pickle=True)['arr_0'].item()
                #np.save(backupfolder + '/' + 'labels.npy', labels_old)
                np.savez(backupfolder + '/' + 'labels.npz', labels_old)
            # autosave
            autosavefolder = self.standardLabelsFolder + '/' + 'autosave'
            if not(os.path.isdir(autosavefolder)):
                os.makedirs(autosavefolder)
            save_cfg(autosavefolder + '/' + 'labeling_gui_cfg.py', self.cfg)
            #file = self.standardLabelsFolder + '/' + 'labels.npy'
            file = self.standardLabelsFolder + '/' + 'labels.npz'
            if os.path.isfile(file):
                #labels_save = np.load(file, allow_pickle=True).item()
                labels_save = np.load(file, allow_pickle=True)['arr_0'].item()
                #np.save(autosavefolder + '/' + 'labels.npy', labels_save)
                ##np.savez(autosavefolder + '/' + 'labels.npz', labels_save)
            # save cfg-file
            save_cfg(self.standardLabelsFolder + '/' + 'labeling_gui_cfg.py', self.cfg)
            # last pose            
            if os.path.isfile(self.standardLabelsFolder + '/' + 'exit_status.npy'):
                exit_status = np.load(self.standardLabelsFolder + '/' + 'exit_status.npy', allow_pickle=True).item()
                self.i_pose = exit_status['i_pose']
                if ((self.i_pose < self.minPose) or (self.i_pose > self.maxPose)):
                    self.i_pose = self.minPose
        
        
        # the following defines default recording/calibration/origin/model/labels to be loaded at start up
        self.recordingIsLoaded = False
        self.calibrationIsLoaded = False
        self.originIsLoaded = False
        self.modelIsLoaded = False
        self.labelsAreLoaded = False
        self.sketchIsLoaded = False
        if (self.cfg['autoLoad']):            
            self.recFileNames = sorted(list([self.cfg['standardRecordingFolder'] + '/' + i for i in self.cfg['standardRecordingFileNames']]))
            self.standardCalibrationFile = self.cfg['standardCalibrationFile']
            self.standardOriginCoordFile = self.cfg['standardOriginCoordFile']
            self.standardModelFile = self.cfg['standardModelFile']
            self.standardLabelsFile = self.cfg['standardLabelsFile']
            self.standardSketchFile = self.cfg['standardSketchFile']
            
            # load recording
            if np.all([os.path.isfile(i_file) for i_file in self.recFileNames]):
                self.recordingIsLoaded = True
                self.nCameras = np.size(self.recFileNames, 0)
                self.xRes = np.zeros(self.nCameras, dtype=np.int64)
                self.yRes = np.zeros(self.nCameras, dtype=np.int64)
                self.nPoses = np.zeros(self.nCameras, dtype=np.int64)
                index = 0
                for fileName in self.recFileNames:
                    if fileName:    
                        reader = imageio.get_reader(fileName)
                        header = read_video_meta(reader)
                        self.nPoses[index] = header['nFrames']
                        self.xRes[index] = header['sensorsize'][0]
                        self.yRes[index] = header['sensorsize'][1]
                        index = index + 1
                    else:
                        print('WARNING: Invalid recording file')
                self.imgs = list()
                for i_cam in range(self.nCameras):
                    self.imgs.append(np.zeros((self.xRes[i_cam], self.yRes[i_cam]), dtype=np.int64))
                self.xLim_prev = np.zeros((self.nCameras, 2), dtype=np.float64)
                self.yLim_prev = np.zeros((self.nCameras, 2), dtype=np.float64)
                for i_cam in range(self.nCameras):
                    self.xLim_prev[i_cam] = np.array([0.0, self.xRes[i_cam] - 1], dtype=np.float64)
                    self.yLim_prev[i_cam] = np.array([0.0, self.yRes[i_cam] - 1], dtype=np.float64)
                self.maxPose = int(np.min([np.min(self.nPoses) - 1, self.cfg['maxPose']]))
            else:
                print(f'WARNING: Autoloading failed. Recording files do not exist: {self.recFileNames}')

            # load calibration
            if os.path.isfile(self.standardCalibrationFile):
                self.cameraSystem = calibcamlib.Camerasystem.from_calibcam_file(self.standardCalibrationFile)
                self.calibrationIsLoaded = True
                self.origin = np.zeros(3, dtype=np.float64)
                self.coord = np.identity(3, dtype=np.float64)
                self.result = np.load(self.standardCalibrationFile, allow_pickle=True).item()        
                A_val = self.result['A_fit']
                self.A = np.zeros((self.nCameras, 3, 3), dtype=np.float64)
                for i in range(self.nCameras):
                    self.A[i][0, 0] = A_val[i, 0]
                    self.A[i][0, 2] = A_val[i, 1]
                    self.A[i][1, 1] = A_val[i, 2]
                    self.A[i][1, 2] = A_val[i, 3]
                    self.A[i][2, 2] = 1.0
                self.k = self.result['k_fit']
                self.rX1 = self.result['rX1_fit']
                self.RX1 = self.result['RX1_fit']
                self.tX1 = self.result['tX1_fit']
            else:
                print(f'WARNING: Autoloading failed. Calibration file {self.standardCalibrationFile} does not exist.')

            # origin / coord in coordinate system of reference camera (only needed to plot reprojection lines until ground of arena floor)
            if os.path.isfile(self.standardOriginCoordFile):
                self.originIsLoaded = True
                self.origin_coord = np.load(self.standardOriginCoordFile, allow_pickle=True).item() 
                self.origin = self.origin_coord['origin']
                self.coord = self.origin_coord['coord']
            else:
                print(f'WARNING: Autoloading failed. Origin/Coord file {self.standardOriginCoordFile} does not exist.')

            # load model
            if os.path.isfile(self.standardModelFile):
                self.modelIsLoaded = True
                self.model = np.load(self.standardModelFile, allow_pickle=True).item()
                self.v = self.model['v'] # surface
                self.f = self.model['f'] # surface
        #        verts = model['verts'] # skeleton
        #        edges = model['edges'] # skeleton
        #        v_link = model['v_link'] # rigid connection v to edges
        #        verts_link = model['verts_link'] # rigid connection verts to edges
        #        nEdges = np.size(edges, 0)
                self.v_center = np.mean(self.v, 0)
                if 'labels3d' in self.model:
                    self.labels3d = copy.deepcopy(self.model['labels3d'])
                    self.labels3d_sequence = sorted(list(self.labels3d.keys()))
                    self.labels3d_sequence = sort_label_sequence(self.labels3d_sequence) # FIXME: comment when you want to label origin/coord, uncomment if you want to actually label something
                else:
                    self.labels3d = dict()
                    self.labels3d_sequence = list([])
                    print('WARNING: Model does not contain 3D Labels! This might lead to incorrect behavior of the GUI.')
            else:
                print(f'WARNING: Autoloading failed. 3D model file {self.standardModelFile} does not exist.')
                
            # load sketch
            if os.path.isfile(self.standardSketchFile):
                self.sketchIsLoaded = True
                self.sketchFile = np.load(self.cfg['standardSketchFile'], allow_pickle=True).item()
                self.sketch = self.sketchFile['sketch']
                self.sketch_annotation = self.sketchFile['sketch_label_locations']
                self.sketch_locations = list()
                for i_label in self.sketch_annotation:
                    self.sketch_locations.append(self.sketch_annotation[i_label])
                self.sketch_locations = np.array(self.sketch_locations, dtype=np.float64)
                
                if not hasattr(self, 'labels3d_sequence'):
                    self.labels3d = copy.deepcopy(self.sketchFile['sketch_label_locations'])
                    self.labels3d_sequence = sorted(list(self.labels3d.keys()))
                    self.labels3d_sequence = sort_label_sequence(self.labels3d_sequence) # FIXME: comment when you want to label origin/coord, uncomment if you want to actually label something
            else:
                print(f'WARNING: Autoloading failed. Sketch file {self.standardSketchFile} does not exist.')

            # load labels
            if (self.master):
                if os.path.isfile(self.standardLabelsFile):
                    self.labelsAreLoaded = True
                    #self.labels2d_all = np.load(self.standardLabelsFile, allow_pickle=True).item()
                    self.labels2d_all = np.load(self.standardLabelsFile, allow_pickle=True)['arr_0'].item()
                    if self.i_pose in self.labels2d_all.keys():
                        self.labels2d = copy.deepcopy(self.labels2d_all[self.i_pose])
                    if (self.label3d_select_status):
                        if (self.list_labels3d.currentItem().text() in self.labels2d.keys()):
                            self.selectedLabel2d = np.copy(self.labels2d[self.list_labels3d.currentItem().text()])
                else:
                    print(f'WARNING: Autoloading failed. Labels file {self.standardLabelsFile} does not exist.')
            else:
                #self.standardLabelsFile = self.standardLabelsFolder + '/' + 'labels.npy'
                self.standardLabelsFile = self.standardLabelsFolder + '/' + 'labels.npz'
                if os.path.isfile(self.standardLabelsFile):
                    self.labelsAreLoaded = True
                    #self.labels2d_all = np.load(self.standardLabelsFile, self.labels2d_all, allow_pickle=True).item()
                    self.labels2d_all = np.load(self.standardLabelsFile, self.labels2d_all, allow_pickle=True)['arr_0'].item()
                    if self.i_pose in self.labels2d_all.keys():
                        self.labels2d = copy.deepcopy(self.labels2d_all[self.i_pose])
                        if (self.label3d_select_status):
                            if (self.list_labels3d.currentItem().text() in self.labels2d.keys()):
                                self.selectedLabel2d = np.copy(self.labels2d[self.list_labels3d.currentItem().text()])

            
    def set_layout(self):
        # frame main
        self.frame_main = QFrame()
        self.setStyleSheet("background-color: black;")
        self.layoutGrid = QGridLayout()
        self.layoutGrid.setSpacing(10)
        self.frame_main.setMinimumSize(512, 512)
        # frame for 2d views
        self.frame_views2d = QFrame()
        self.frame_views2d.setStyleSheet("background-color: white;")
        self.layoutGrid.setRowStretch(0, 2)
        self.layoutGrid.setColumnStretch(0, 2)
        self.layoutGrid.addWidget(self.frame_views2d, 0, 0, 2, 4)
        
        self.views2d_layoutGrid = QGridLayout()
        self.views2d_layoutGrid.setSpacing(0)
        self.frame_views2d.setLayout(self.views2d_layoutGrid)
                
        # frame for 3d model
        self.frame_views3d = QFrame()
        self.frame_views3d.setStyleSheet("background-color:  white;")
        self.layoutGrid.setRowStretch(0, 1)
        self.layoutGrid.setColumnStretch(4, 1)
        self.layoutGrid.addWidget(self.frame_views3d, 0, 4)
        
        self.views3d_layoutGrid = QGridLayout()
        self.views3d_layoutGrid.setSpacing(0)
        self.frame_views3d.setLayout(self.views3d_layoutGrid)
        
        # frame for controls
        self.frame_controls = QFrame()
        self.frame_controls.setStyleSheet("background-color: white;")
        self.layoutGrid.setRowStretch(1, 1)
        self.layoutGrid.setColumnStretch(4, 1)
        self.layoutGrid.addWidget(self.frame_controls, 1, 4)
        # add to grid
        self.frame_main.setLayout(self.layoutGrid)
        self.setCentralWidget(self.frame_main)

    # 2d plots
    def plot2d_update(self):
        if (self.button_fastLabelingMode_status):
            self.plot2d_drawLabels(self.ax2d[self.i_cam], self.i_cam)
            self.fig2d[self.i_cam].canvas.draw()
        else:
            for i_cam in range(self.nCameras):
                self.plot2d_drawLabels(self.ax2d[i_cam], i_cam)
                self.fig2d[i_cam].canvas.draw()
    
    def plot2d_drawLabels(self, ax, i_cam):
        ax.lines = list()
        # reprojection lines
        if (self.button_reprojectionMode_status &
            self.label3d_select_status):
            for i in range(self.nCameras):
                if ((i != i_cam) &
                    (not(np.any(np.isnan(self.selectedLabel2d[i]))))):
                    
                    # 2d point to 3d line
                    nLineElements = np.int64(1e3)

                    point = self.selectedLabel2d[i]
                    # lazy implementation of A**-1 * point
#                    point = np.array([[point[0], point[1], 1.0]], dtype=np.float64)
#                    A_1 = np.linalg.lstsq(self.A[i], np.identity(3), rcond=None)[0]
#                    point = np.dot(A_1, point.T).T
                    # fast implementation of A**-1 * point (assumes no skew!)
                    point = np.array([[(point[0] - self.A[i][0, 2]) / self.A[i][0, 0],
                                       (point[1] - self.A[i][1, 2]) / self.A[i][1, 1],
                                       1.0]], dtype=np.float64)
                    point = calc_udst(point, self.k[i]).T
                    
                    # beginning and end of linspace-function are arbitary
                    # might need to increase the range here when the lines are not visible
                    point = point * np.linspace(0, 1e3, nLineElements)
                    line = np.dot(self.RX1[i].T,
                                  point - self.tX1[i].reshape(3, 1))
                    
                    if (self.originIsLoaded):
                        # transform into world coordinate system
                        line = line - self.origin.reshape(3, 1)
                        line = np.dot(self.coord.T, line)
                        # only use line until intersection with the x-y-plane
                        n = line[:, 0]
                        m = line[:, -1] - line[:, 0]
                        lambda_val = -n[2] / m[2]
                        line = np.linspace(0.0, 1.0, nLineElements).reshape(1, nLineElements).T * \
                               m * lambda_val + n
                        # transform back into coordinate system of camera i
                        line = np.dot(self.coord, line.T)
                        line = line + self.origin.reshape(3, 1)
        
                    # 3d line to 2d point
                    line_proj = np.dot(self.RX1[i_cam], line) + \
                                       self.tX1[i_cam].reshape(3, 1)
                    line_proj = calc_dst(line_proj.T, self.k[i_cam]).T
                    line_proj = np.dot(self.A[i_cam], line_proj).T
                    

                    ax.plot(line_proj[:, 0], line_proj[:, 1],
                            linestyle='-',
                            color=self.colors[i % np.size(self.colors)],
                            alpha=0.5,
                            zorder=1,
                            linewidth=1.0)
        # labels
#        if (self.labels2d_exists):
        for i_label in self.labels2d.keys():
            point = self.labels2d[i_label][i_cam]
            ax.plot([point[0]], [point[1]],
                     marker='o',
                     color='cyan',
                     markersize=3,
                     zorder=2)
        if (self.label3d_select_status):
            if not(np.any(np.isnan(self.selectedLabel2d[i_cam]))):
                ax.plot([self.selectedLabel2d[i_cam, 0]],
                        [self.selectedLabel2d[i_cam, 1]],
                        marker='o',
                        color='darkgreen',
                        markersize=4,
                        zorder=3)
                                       
    def plot2d_plotSingleImage_ini(self, ax, i_cam):
        reader = imageio.get_reader(self.recFileNames[i_cam])
        self.imgs[i_cam] = reader.get_data(self.i_pose)
        self.h_imgs[i_cam] = ax.imshow(self.imgs[i_cam],
                                       aspect=1,
                                       cmap='gray',
                                       vmin=self.vmin,
                                       vmax=self.vmax)
        ax.legend('',
                  facecolor=self.colors[i_cam % np.size(self.colors)],
                  loc='upper left',
                  bbox_to_anchor=(0, 1))
#         self.h_titles[i_cam] = ax.set_title('camera: {:01d}, frame: {:06d}'.format(i_cam, self.i_pose))
#         ax.set_xticklabels('')
#         ax.set_yticklabels('')
        ax.axis('off')
        if (self.button_fastLabelingMode_status &
            self.button_centricViewMode_status):
            ax.set_xlim(self.xLim_prev[i_cam])
            ax.set_ylim(self.yLim_prev[i_cam])
            if not(np.any(np.isnan(self.selectedLabel2d[i_cam]))):
                ax.set_xlim(self.selectedLabel2d[i_cam, 0] - self.dx,
                            self.selectedLabel2d[i_cam, 0] + self.dx)
                ax.set_ylim(self.selectedLabel2d[i_cam, 1] - self.dy,
                            self.selectedLabel2d[i_cam, 1] + self.dy)
            if not(np.any(np.isnan(self.clickedLabel2d))):
                ax.set_xlim(self.clickedLabel2d[0] - self.dx,
                            self.clickedLabel2d[0] + self.dx)
                ax.set_ylim(self.clickedLabel2d[1] - self.dy,
                            self.clickedLabel2d[1] + self.dy)
        else:
            ax.set_xlim(0.0, self.xRes[i_cam] - 1)
            ax.set_ylim(0.0, self.yRes[i_cam] - 1)
        if (self.cfg['invert_xaxis']):
            ax.invert_xaxis()
        if (self.cfg['invert_yaxis']):
            ax.invert_yaxis()
        #
        self.plot2d_drawLabels(ax, i_cam)

    def plot2d_plotSingleImage(self, ax, i_cam):
        reader = imageio.get_reader(self.recFileNames[i_cam])
        self.imgs[i_cam] = reader.get_data(self.i_pose)
        self.h_imgs[i_cam].set_array(self.imgs[i_cam])
        self.h_imgs[i_cam].set_clim(self.vmin, self.vmax)
#         self.h_titles[i_cam].set_text('camera: {:01d}, frame: {:06d}'.format(i_cam, self.i_pose))
        self.plot2d_drawLabels(ax, i_cam)
        if (self.button_fastLabelingMode_status &
            self.button_centricViewMode_status):
#             ax.set_xlim(self.xLim_prev[i_cam])
#             ax.set_ylim(self.yLim_prev[i_cam])
            ax.set_xlim(self.xLim_prev[i_cam][0], self.xLim_prev[i_cam][1])
            ax.set_ylim(self.yLim_prev[i_cam][0], self.yLim_prev[i_cam][1])
            if not(np.any(np.isnan(self.selectedLabel2d[i_cam]))):
                ax.set_xlim(self.selectedLabel2d[i_cam, 0] - self.dx,
                            self.selectedLabel2d[i_cam, 0] + self.dx)
                ax.set_ylim(self.selectedLabel2d[i_cam, 1] - self.dy,
                            self.selectedLabel2d[i_cam, 1] + self.dy)
            if not(np.any(np.isnan(self.clickedLabel2d))):
                ax.set_xlim(self.clickedLabel2d[0] - self.dx,
                            self.clickedLabel2d[0] + self.dx)
                ax.set_ylim(self.clickedLabel2d[1] - self.dy,
                            self.clickedLabel2d[1] + self.dy)
        else:
            ax.set_xlim(0.0, self.xRes[i_cam] - 1)
            ax.set_ylim(0.0, self.yRes[i_cam] - 1)
        if (self.cfg['invert_xaxis']):
            ax.invert_xaxis()
        if (self.cfg['invert_yaxis']):
            ax.invert_yaxis()
        self.fig2d[i_cam].canvas.draw()
        
    def plot2d_drawNormal_ini(self):
        for i in reversed(range(self.views2d_layoutGrid.count())):
            widgetToRemove = self.views2d_layoutGrid.itemAt(i).widget()
            self.views2d_layoutGrid.removeWidget(widgetToRemove)
            widgetToRemove.setParent(None)
            
        if (self.toolbars_zoom_status):
            self.button_zoom_press()
        if (self.toolbars_pan_status):
            self.button_pan_press()
            
        self.fig2d = list()
        self.ax2d = list()
        self.h_imgs = list()
#         self.h_titles = list()
        self.toolbars = list()
        self.frame_views2d.setCursor(QCursor(QtCore.Qt.CrossCursor))
        for i_cam in range(self.nCameras):   
                frame = QFrame()
                frame.setParent(self.frame_views2d)
                frame.setStyleSheet("background-color: gray;")
                fig = Figure(tight_layout=True)
                fig.clear()
                self.fig2d.append(fig)
                canvas = FigureCanvasQTAgg(fig)
                canvas.setParent(frame)   
                ax = fig.add_subplot('111')
                ax.clear()
                self.ax2d.append(ax)
                self.h_imgs.append(list())
#                 self.h_titles.append(list())
                self.plot2d_plotSingleImage_ini(self.ax2d[-1], i_cam)
                
                layout = QGridLayout()
                layout.addWidget(canvas)
                frame.setLayout(layout)
                
                self.views2d_layoutGrid.addWidget(frame,
                                                  int(np.floor(i_cam / 2)),
                                                  i_cam % 2)
                
                toolbar = NavigationToolbar2QT(canvas, self)
                toolbar.hide()
                self.toolbars.append(toolbar)
                
                fig.canvas.mpl_connect('button_press_event',
                                       lambda event: self.plot2d_click(event))
                
    def plot2d_drawNormal(self):            
        if (self.toolbars_zoom_status):
            self.button_zoom_press()
        if (self.toolbars_pan_status):
            self.button_pan_press()
            
        for i_cam in range(self.nCameras):
            self.plot2d_plotSingleImage(self.ax2d[i_cam], i_cam)

    def plot2d_drawFast_ini(self):
        for i in reversed(range(self.views2d_layoutGrid.count())):
            widgetToRemove = self.views2d_layoutGrid.itemAt(i).widget()
            self.views2d_layoutGrid.removeWidget(widgetToRemove)
            widgetToRemove.setParent(None)
            
        if (self.toolbars_zoom_status):
            self.button_zoom_press()
        if (self.toolbars_pan_status):
            self.button_pan_press()
                
        fig = Figure(tight_layout=True)
        fig.clear()
        self.fig2d[self.i_cam] = fig
        canvas = FigureCanvasQTAgg(fig)
        canvas.setParent(self.frame_views2d)
        ax = fig.add_subplot('111')
        ax.clear()
        self.ax2d[self.i_cam] = ax
        self.plot2d_plotSingleImage_ini(self.ax2d[self.i_cam], self.i_cam)
                
        self.views2d_layoutGrid.addWidget(canvas, 0, 0)
                        
        toolbar = NavigationToolbar2QT(canvas, self)
        toolbar.hide()
        self.toolbars = list([toolbar])
        
        fig.canvas.mpl_connect('button_press_event',
                               lambda event: self.plot2d_click(event))
        
    def plot2d_drawFast(self):
        if (self.toolbars_zoom_status):
            self.button_zoom_press()
        if (self.toolbars_pan_status):
            self.button_pan_press()
            
        self.plot2d_plotSingleImage(self.ax2d[self.i_cam], self.i_cam)
        
    def plot2d_click(self, event):
        if (self.label3d_select_status & (not(self.toolbars_zoom_status)) & (not(self.toolbars_pan_status))):
            ax = event.inaxes
            if (ax != None):
                i_cam = self.ax2d.index(ax)
                if (event.button == 1):
                    x = event.xdata
                    y = event.ydata
                    if ((x != None) & (y != None)):
                        self.clickedLabel2d = np.array([x, y], dtype=np.float64)
                        self.clickedLabel2d_pose = np.copy(self.i_pose)
                        self.selectedLabel2d[i_cam] = np.array([x, y], dtype=np.float64)

                        if not(self.list_labels3d.currentItem().text() in self.labels2d.keys()):
                            self.labels2d[self.list_labels3d.currentItem().text()] = np.full((self.nCameras, 2), np.nan, dtype=np.float64)
                        self.labels2d[self.list_labels3d.currentItem().text()][i_cam] = np.array([x, y], dtype=np.float64)
                        
                        self.plot2d_update()
                        if (self.button_sketchMode_status):
                            self.sketch_update()
                elif (event.button == 3):
                    self.clickedLabel2d = np.array([np.nan, np.nan], dtype=np.float64)
                    self.selectedLabel2d[i_cam] = np.array([np.nan, np.nan], dtype=np.float64)
                    if (self.list_labels3d.currentItem().text() in self.labels2d.keys()):
                        self.labels2d[self.list_labels3d.currentItem().text()][i_cam] = np.array([np.nan, np.nan], dtype=np.float64)
                        if np.all(np.isnan(self.labels2d[self.list_labels3d.currentItem().text()])):
                            del(self.labels2d[self.list_labels3d.currentItem().text()])
                    if (self.i_pose in self.labels2d_all.keys()):
                        if (self.list_labels3d.currentItem().text() in self.labels2d_all[self.i_pose].keys()):
                            self.labels2d_all[self.i_pose][self.list_labels3d.currentItem().text()][i_cam] = np.array([np.nan, np.nan], dtype=np.float64)
                            if np.all(np.isnan(self.labels2d_all[self.i_pose][self.list_labels3d.currentItem().text()])):
                                del(self.labels2d_all[self.i_pose][self.list_labels3d.currentItem().text()])
                        if not(bool(self.labels2d_all[self.i_pose])):
                            del(self.labels2d_all[self.i_pose])
                    self.plot2d_update()
                    if (self.button_sketchMode_status):
                        self.sketch_update()
        
    # 3d plot
    def plot3d_draw(self):
        for i in reversed(range(self.views3d_layoutGrid.count())):
            widgetToRemove = self.views3d_layoutGrid.itemAt(i).widget()
            self.views3d_layoutGrid.removeWidget(widgetToRemove)
            widgetToRemove.setParent(None)
        
        self.fig3d = Figure(tight_layout=True)
        self.fig3d.clear()
        self.canvas3d = FigureCanvasQTAgg(self.fig3d)
        self.ax3d = self.fig3d.add_subplot(111, projection='3d')
        self.ax3d.clear()
        self.ax3d.grid(False)

        if (self.modelIsLoaded):
            try:
                # FIXME: xyz should be already in the dictonary
                nPoly = np.size(self.f, 0)
                xyz = np.zeros((3, 3), dtype=np.float64)
                xyz_all = np.zeros((nPoly, 3, 3), dtype=np.float64)
                for i in range(nPoly):
                    f_use = self.f[i, :, 0]
                    xyz[0, :] = self.v[f_use[0] - 1]
                    xyz[1, :] = self.v[f_use[1] - 1]
                    xyz[2, :] = self.v[f_use[2] - 1]
                    xyz_all[i] = xyz

                self.surfModel3d = Poly3DCollection(xyz_all)
                self.surfModel3d.set_alpha(0.1)
                self.surfModel3d.set_edgecolor('black')
                self.surfModel3d.set_facecolor('gray')

                self.ax3d.add_collection3d(self.surfModel3d)

                #self.ax3d.set_aspect('equal')

                self.ax3d.set_xlim([self.v_center[0] - self.dxyz_lim,
                                    self.v_center[0] + self.dxyz_lim])
                self.ax3d.set_ylim([self.v_center[1] - self.dxyz_lim,
                                    self.v_center[1] + self.dxyz_lim])
                self.ax3d.set_zlim([self.v_center[2] - self.dxyz_lim,
                                    self.v_center[2] + self.dxyz_lim])

        #        self.ax3d.dist = 9.25

        #            self.ax3d.set_xlabel('x')
        #            self.ax3d.set_ylabel('y')
        #            self.ax3d.set_zlabel('z')
        #            # it is important to hide the ticks separately like this
        #            # if done otherwise, self.ax3d.format_coord() does not work
        #            for x_label in self.ax3d.xaxis.get_ticklabels():
        #                x_label.set_visible(False)
        #            for y_label in self.ax3d.yaxis.get_ticklabels():
        #                y_label.set_visible(False)
        #            for z_label in self.ax3d.zaxis.get_ticklabels():
        #                z_label.set_visible(False)
                self.ax3d.set_axis_off()
            except:
                pass
    
        self.plot3d_update()
#        self.fig3d.tight_layout()
        
        self.views3d_layoutGrid.addWidget(self.canvas3d)
    
    def plot3d_update(self):
        try:
            self.ax3d.lines = list()
            for label3d_name in self.labels3d.keys():
                color = 'orange'
                if label3d_name in self.label2d_max_err:
                    color = 'red'
                elif label3d_name in self.labels2d.keys():
                    color = 'cyan'

                self.ax3d.plot([self.labels3d[label3d_name][0]],
                            [self.labels3d[label3d_name][1]],
                            [self.labels3d[label3d_name][2]],
                                marker='o',
                                color=color,
                                markersize=4,
                                zorder=2)
    #        if (self.label3d_select_status):
            self.ax3d.plot([self.selectedLabel3d[0]],
                        [self.selectedLabel3d[1]],
                        [self.selectedLabel3d[2]],
                            marker='o',
                            color='darkgreen',
                            markersize=6,
                            zorder=3)
            self.canvas3d.draw()
        except:
            pass
        
    # sketch
    def sketch_draw(self):
        for i in reversed(range(self.views3d_layoutGrid.count())):
            widgetToRemove = self.views3d_layoutGrid.itemAt(i).widget()
            self.views3d_layoutGrid.removeWidget(widgetToRemove)
            widgetToRemove.setParent(None)
            
        self.figSketch = Figure()
        self.figSketch.clear()
        self.canvasSketch = FigureCanvasQTAgg(self.figSketch)
        
        self.sketchZoom_scale = 0.1
        self.sketchZoom_dx = np.max(np.shape(self.sketch)) * self.sketchZoom_scale
        self.sketchZoom_dy = np.max(np.shape(self.sketch)) * self.sketchZoom_scale
        
        # full
        self.axSketch_left = 0/3
        self.axSketch_bottom = 1/18
        self.axSketch_width = 1/3
        self.axSketch_height = 16/18
        self.axSketch = self.figSketch.add_axes([self.axSketch_left,
                                                 self.axSketch_bottom,
                                                 self.axSketch_width,
                                                 self.axSketch_height])
        self.axSketch.clear()
        self.axSketch.grid(False)
        self.axSketch.imshow(self.sketch)
        self.axSketch.axis('off')
        self.sketchTitle = self.axSketch.set_title('Full:',
                                                   ha='center', va='center',
                                                   zorder=0) 
        self.sketch_point = self.axSketch.plot([np.nan], [np.nan],
                                               color='darkgreen',
                                               marker='.',
                                               markersize=2,
                                               alpha=1.0,
                                               zorder=2)
        self.sketch_circle = self.axSketch.plot([np.nan], [np.nan],
                                                color='darkgreen',
                                                marker='o',
                                                markersize=20,
                                                markeredgewidth=2,
                                                fillstyle='none',
                                                alpha=2/3,
                                                zorder=2)        
        self.axSketch.set_xlim([-self.sketchZoom_dx/2, np.shape(self.sketch)[1]+self.sketchZoom_dx/2])
        self.axSketch.set_ylim([-self.sketchZoom_dy/2, np.shape(self.sketch)[0]+self.sketchZoom_dy/2])
        self.axSketch.invert_yaxis()
        # zoom
        self.axSketchZoom_left = 1/3
        self.axSketchZoom_bottom = 5/18
        self.axSketchZoom_width = 2/3
        self.axSketchZoom_height = 12/18
        self.axSketchZoom = self.figSketch.add_axes([self.axSketchZoom_left,
                                                     self.axSketchZoom_bottom,
                                                     self.axSketchZoom_width,
                                                     self.axSketchZoom_height])
        self.axSketchZoom.imshow(self.sketch)
        self.axSketchZoom.set_xlabel('')
        self.axSketchZoom.set_ylabel('')
        self.axSketchZoom.set_xticks(list())
        self.axSketchZoom.set_yticks(list())
        self.axSketchZoom.set_xticklabels(list())
        self.axSketchZoom.set_yticklabels(list())
        self.sketchZoomTitle = self.axSketchZoom.set_title('Zoom:',
                                                           ha='center', va='center',
                                                           zorder=0) 
        self.axSketchZoom.grid(False)
        self.sketchZoom_point = self.axSketchZoom.plot([np.nan], [np.nan],
                                               color='darkgreen',
                                               marker='.',
                                               markersize=4,
                                               alpha=1.0,
                                               zorder=2)
        self.sketchZoom_circle = self.axSketchZoom.plot([np.nan], [np.nan],
                                                color='darkgreen',
                                                marker='o',
                                                markersize=40,
                                                markeredgewidth=4,
                                                fillstyle='none',
                                                alpha=2/3,
                                                zorder=2)
        self.axSketchZoom.set_xlim([np.shape(self.sketch)[1]/2-self.sketchZoom_dx, np.shape(self.sketch)[1]/2+self.sketchZoom_dx])
        self.axSketchZoom.set_ylim([np.shape(self.sketch)[0]/2-self.sketchZoom_dy, np.shape(self.sketch)[0]/2+self.sketchZoom_dy])
        self.axSketchZoom.invert_yaxis()
        # text
        self.textSketch = self.figSketch.text(self.axSketchZoom_left + self.axSketchZoom_width/2,
                                              self.axSketchZoom_bottom/2,
                                              'Label {:02d}:\n{:s}'.format(0, ''),
                                              ha='center', va='center',
                                              fontsize=18,
                                              zorder=2)
        # overview of marked labels
        self.sketch_labels = list()
        self.sketchZoom_labels = list()
        for label_index in range(np.size(self.labels3d_sequence)):
            color = 'orange'
            label_name = self.labels3d_sequence[label_index]
            if label_name in self.labels2d:
                if label_name in self.label2d_max_err:
                    color = 'red'
                elif (self.button_fastLabelingMode_status):
                    if np.all(np.logical_not(np.isnan(self.labels2d[label_name][self.i_cam]))):
                        color = 'cyan'
                else:
                    if np.any(np.logical_not(np.isnan(self.labels2d[label_name]))):
                        color = 'cyan'

            label_location = self.sketch_locations[label_index]
            sketch_labels = self.axSketch.plot([label_location[0]],
                                               [label_location[1]],
                                               marker='o',
                                               color=color,
                                               markersize=3,
                                               zorder=1)
            self.sketch_labels.append(sketch_labels[0])
            sketchZoom_labels = self.axSketchZoom.plot([label_location[0]],
                                                       [label_location[1]],
                                                       marker='o',
                                                       color=color,
                                                       markersize=5,
                                                       zorder=1)
            self.sketchZoom_labels.append(sketchZoom_labels[0])
        # set selected label to first in sequence if none is selected
        if not(self.label3d_select_status):
            first_label_name = self.labels3d_sequence[0]
            sorted_index = sorted(list(self.labels3d.keys())).index(first_label_name) 
            self.list_labels3d.setCurrentRow(sorted_index)
            self.list_labels3d_select() 
    
        self.sketch_update()
        self.views3d_layoutGrid.addWidget(self.canvasSketch)
        
        
    def sketch_update(self):
        if (self.label3d_select_status):
            selected_label_name = self.list_labels3d.currentItem().text()
            label_index = self.labels3d_sequence.index(selected_label_name)
            x = self.sketch_locations[label_index, 0]
            y = self.sketch_locations[label_index, 1]

            sellabelerr = np.asarray([])
            labelerr = np.asarray([])
            if self.cameraSystem is not None and len(self.labels2d.keys())>0:
                #self.labels2d[i_label][i_cam]
                #self.selectedLabel2d[i_cam, 0]

                labels2d = np.zeros(shape=(self.selectedLabel2d.shape[0],len(self.labels2d.keys()),self.selectedLabel2d.shape[1]))
                labels2d[:] = np.NaN
                for i,m in enumerate(self.labels2d.keys()):
                    labels2d[:,i,:] = self.labels2d[m][:,:]

                (X,P,V) = self.cameraSystem.triangulate_3derr(labels2d)
                sel_x = self.cameraSystem.project(X)
                labelerr = np.sum((labels2d-sel_x)**2,axis=2)

                (X,P,V) = self.cameraSystem.triangulate_3derr(self.selectedLabel2d[:,np.newaxis,:])
                sel_x = self.cameraSystem.project(X)
                sellabelerr = np.sum((self.selectedLabel2d[:,np.newaxis,:]-sel_x)**2,axis=2)

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

                if not sellabelerr.size == 0 and not labelerr.size == 0:
                    self.textSketch.set(text=f'Label {(label_index+1):02d}:\n{selected_label_name}\nLabel error: {np.nanmax(sellabelerr):6.1f}\nFrame error: {np.nanmax(labelerr):6.1f}')
                    self.label2d_max_err = np.asarray(list(self.labels2d.keys()))[np.nanmax(labelerr)==np.nanmax(labelerr,axis=0)]
                else:
                    self.textSketch.set(text=f'Label {(label_index+1):02d}:\n{selected_label_name}')

            # labels
            for label_index in range(np.size(self.labels3d_sequence)):
                color = 'orange'
                label_name = self.labels3d_sequence[label_index]
                if label_name in self.labels2d:
                    if label_name in self.label2d_max_err:
                        color = 'red'
                    elif (self.button_fastLabelingMode_status):
                        if np.all(np.logical_not(np.isnan(self.labels2d[label_name][self.i_cam]))):
                            color = 'cyan'
                    else:
                        if np.any(np.logical_not(np.isnan(self.labels2d[label_name]))):
                            color = 'cyan'
                self.sketch_labels[label_index].set(color=color)
                self.sketchZoom_labels[label_index].set(color=color)
            # full
            self.sketch_point[0].set_data([x], [y])
            self.sketch_circle[0].set_data([x], [y])
            # zoom
            self.sketchZoom_point[0].set_data([x], [y])
            self.sketchZoom_circle[0].set_data([x], [y])
            self.axSketchZoom.set_xlim([x - self.sketchZoom_dx, x + self.sketchZoom_dx])
            self.axSketchZoom.set_ylim([y - self.sketchZoom_dy, y + self.sketchZoom_dy])
            self.axSketchZoom.invert_yaxis()

            self.canvasSketch.draw()
            self.button_zoom_press(tostate="off")
        
    # controls
    def set_controls(self):
        self.controls_layoutGrid = QGridLayout()
        row = 0
        col = 0
        
        # general
        self.button_loadRecording = QPushButton()
        if (self.recordingIsLoaded):
            self.button_loadRecording.setStyleSheet("background-color: green;")
        else:
            self.button_loadRecording.setStyleSheet("background-color: darkred;")
        self.button_loadRecording.setText('Load Recording')
        self.button_loadRecording.clicked.connect(self.button_loadRecording_press)
        self.controls_layoutGrid.addWidget(self.button_loadRecording, row, col)
        self.button_loadRecording.setEnabled(self.cfg['button_loadRecording'])
        col = col + 1
        
        self.button_loadModel = QPushButton()
        if (self.modelIsLoaded):
            self.button_loadModel.setStyleSheet("background-color: green;")
        else:
            self.button_loadModel.setStyleSheet("background-color: darkred;")
        self.button_loadModel.setText('Load Model')
        self.button_loadModel.clicked.connect(self.button_loadModel_press)
        self.controls_layoutGrid.addWidget(self.button_loadModel, row, col)
        self.button_loadModel.setEnabled(self.cfg['button_loadModel'])
        col = col + 1
        
        self.button_loadLabels = QPushButton()
        if (self.labelsAreLoaded):
            self.button_loadLabels.setStyleSheet("background-color: green;")
        else:
            self.button_loadLabels.setStyleSheet("background-color: darkred;")
        self.button_loadLabels.setText('Load Labels')
        self.button_loadLabels.clicked.connect(self.button_loadLabels_press)
        self.controls_layoutGrid.addWidget(self.button_loadLabels, row, col)
        self.button_loadLabels.setEnabled(self.cfg['button_loadLabels'])
        row = row + 1
        col = 0
        
        self.button_loadCalibration = QPushButton()
        if (self.calibrationIsLoaded):
            self.button_loadCalibration.setStyleSheet("background-color: green;")
        else:
            self.button_loadCalibration.setStyleSheet("background-color: darkred;")
        self.button_loadCalibration.setText('Load Calibration')
        self.button_loadCalibration.clicked.connect(self.button_loadCalibration_press)
        self.controls_layoutGrid.addWidget(self.button_loadCalibration, row, col)
        self.button_loadCalibration.setEnabled(self.cfg['button_loadCalibration'])
        col = col + 1
        
        self.button_saveModel = QPushButton()
        self.button_saveModel.setText('Save Model')
        self.button_saveModel.clicked.connect(self.button_saveModel_press)
        self.controls_layoutGrid.addWidget(self.button_saveModel, row, col)
        self.button_saveModel.setEnabled(self.cfg['button_saveModel'])
        col = col + 1
        
        self.button_saveLabels = QPushButton()
        self.button_saveLabels.setText('Save Labels (S)')
        self.button_saveLabels.clicked.connect(self.button_saveLabels_press)
        self.controls_layoutGrid.addWidget(self.button_saveLabels, row, col)
        self.button_saveLabels.setEnabled(self.cfg['button_saveLabels'])
        row = row + 1
        col = 0
       
    
    
        self.button_loadOrigin = QPushButton()
        self.button_loadOrigin_status = False
        if (self.originIsLoaded):
            self.button_loadOrigin.setStyleSheet("background-color: green;")
        else:
            self.button_loadOrigin.setStyleSheet("background-color: darkred;")
        self.button_loadOrigin.setText('Load Origin')
        self.controls_layoutGrid.addWidget(self.button_loadOrigin, row, col)
        self.button_loadOrigin.clicked.connect(self.button_loadOrigin_press)
        self.button_loadOrigin.setEnabled(self.cfg['button_loadOrigin'])
        col = col + 1
        
        self.button_loadSketch = QPushButton()
        self.button_loadSketch_status = False
        if (self.sketchIsLoaded):
            self.button_loadSketch.setStyleSheet("background-color: green;")
        else:
            self.button_loadSketch.setStyleSheet("background-color: darkred;")
        self.button_loadSketch.setText('Load Sketch')
        self.controls_layoutGrid.addWidget(self.button_loadSketch, row, col)
        self.button_loadSketch.clicked.connect(self.button_loadSketch_press)
        self.button_loadSketch.setEnabled(self.cfg['button_loadSketch'])
        col = col + 1
        
        self.button_sketchMode = QPushButton()
        self.button_sketchMode_status = False
        self.button_sketchMode.setStyleSheet("background-color: darkred;")
        self.button_sketchMode.setText('Sketch Mode')
        self.controls_layoutGrid.addWidget(self.button_sketchMode, row, col)
        self.button_sketchMode.clicked.connect(self.button_sketchMode_press)
        self.button_sketchMode.setEnabled(self.cfg['button_sketchMode'])
        row = row + 1
        col = 0
    
    
    
    
        self.button_fastLabelingMode = QPushButton()
        self.button_fastLabelingMode_status = False
        self.button_fastLabelingMode.setStyleSheet("background-color: darkred;")
        self.button_fastLabelingMode.setText('Fast Labeling Mode')
        self.button_fastLabelingMode.clicked.connect(self.button_fastLabelingMode_press)
        self.button_fastLabelingMode.setSizePolicy(QSizePolicy.Expanding,
                                                   QSizePolicy.Preferred)
        self.controls_layoutGrid.addWidget(self.button_fastLabelingMode, row, col, 2, 1)
        self.button_fastLabelingMode.setEnabled(self.cfg['button_fastLabelingMode'])
        col = col + 1
        
        self.list_fastLabelingMode = QComboBox()
        self.list_fastLabelingMode.addItems([str(i) for i in range(self.nCameras)])
        self.list_fastLabelingMode.setSizePolicy(QSizePolicy.Expanding,
                                                 QSizePolicy.Preferred)
        self.controls_layoutGrid.addWidget(self.list_fastLabelingMode, row, col, 2, 2)
        self.list_fastLabelingMode.currentIndexChanged.connect(self.list_fastLabelingMode_change)
        self.list_fastLabelingMode.setEnabled(self.cfg['list_fastLabelingMode'])
        row = row + 2
        col = 0
        
        self.button_centricViewMode = QPushButton()
        self.button_centricViewMode_status = False
        self.button_centricViewMode.setStyleSheet("background-color: darkred;")
        self.button_centricViewMode.setText('Centric View Mode (C)')
        self.button_centricViewMode.setSizePolicy(QSizePolicy.Expanding,
                                                  QSizePolicy.Preferred)
        self.controls_layoutGrid.addWidget(self.button_centricViewMode, row, col, 2, 1)
        self.button_centricViewMode.clicked.connect(self.button_centricViewMode_press)
        self.button_centricViewMode.setEnabled(self.cfg['button_centricViewMode'])
        col = col + 1
        
        self.label_dx = QLabel()
        self.label_dx.setText('dx:')
        self.controls_layoutGrid.addWidget(self.label_dx, row, col)
        col = col + 1
        
        self.label_dy = QLabel()
        self.label_dy.setText('dy:')
        self.controls_layoutGrid.addWidget(self.label_dy, row, col)
        row = row + 1
        col = 1
        
        self.field_dx = QLineEdit()
        self.field_dx.setValidator(QIntValidator())
        self.field_dx.setText(str(self.dx))
        self.controls_layoutGrid.addWidget(self.field_dx, row, col)
        self.field_dx.returnPressed.connect(self.field_dx_change)
        self.field_dx.setEnabled(self.cfg['field_dx'])
        col = col + 1
        
        self.field_dy = QLineEdit()
        self.field_dy.setValidator(QIntValidator())
        self.field_dy.setText(str(self.dy))
        self.controls_layoutGrid.addWidget(self.field_dy, row, col)
        self.field_dy.returnPressed.connect(self.field_dy_change)
        self.field_dy.setEnabled(self.cfg['field_dy'])
        row = row + 1
        col = 0
        
        
        self.button_reprojectionMode = QPushButton()
        self.button_reprojectionMode_status = False
        self.button_reprojectionMode.setStyleSheet("background-color: darkred;")
        self.button_reprojectionMode.setText('Reprojection Mode')
        self.button_reprojectionMode.setSizePolicy(QSizePolicy.Expanding,
                                                  QSizePolicy.Preferred)
        self.controls_layoutGrid.addWidget(self.button_reprojectionMode, row, col, 2, 1)
        self.button_reprojectionMode.clicked.connect(self.button_reprojectionMode_press)
        self.button_reprojectionMode.setEnabled(self.cfg['button_reprojectionMode'])
        col = col + 1
        
        self.label_vmin = QLabel()
        self.label_vmin.setText('vmin:')
        self.controls_layoutGrid.addWidget(self.label_vmin, row, col)
        col = col + 1
        
        self.label_vmax = QLabel()
        self.label_vmax.setText('vmax:')
        self.controls_layoutGrid.addWidget(self.label_vmax, row, col)
        row = row + 1
        col = 0
        
        col = col + 1
        
        self.field_vmin = QLineEdit()
        self.field_vmin.setValidator(QIntValidator())
        self.field_vmin.setText(str(self.vmin))
        self.controls_layoutGrid.addWidget(self.field_vmin, row, col)
        self.field_vmin.returnPressed.connect(self.field_vmin_change)
        self.field_vmin.setEnabled(self.cfg['field_vmin'])
        col = col + 1
        
        self.field_vmax = QLineEdit()
        self.field_vmax.setValidator(QIntValidator())
        self.field_vmax.setText(str(self.vmax))
        self.controls_layoutGrid.addWidget(self.field_vmax, row, col)
        self.field_vmax.returnPressed.connect(self.field_vmax_change)
        self.field_vmax.setEnabled(self.cfg['field_vmax'])
        row = row + 1
        col = 0



        # 3d
        self.field_labels3d = QLineEdit()
        self.field_labels3d.returnPressed.connect(self.button_insert_press)
        self.field_labels3d.setSizePolicy(QSizePolicy.Expanding,
                                          QSizePolicy.Preferred)
        self.controls_layoutGrid.addWidget(self.field_labels3d, row, col, 1, 1)
        self.field_labels3d.setEnabled(self.cfg['field_labels3d'])
        col = col + 1
        
        self.list_labels3d = QListWidget()
        self.list_labels3d.setSortingEnabled(True)
        self.list_labels3d.addItems(sorted(list(self.labels3d.keys())))
        self.list_labels3d.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_labels3d.itemClicked.connect(self.list_labels3d_select)
        self.list_labels3d.setSizePolicy(QSizePolicy.Expanding,
                                         QSizePolicy.Preferred)
        self.controls_layoutGrid.addWidget(self.list_labels3d, row, col, 3, 2)
        row = row + 1
        col = 0
        
        self.button_insert = QPushButton()
        self.button_insert.setText('Insert')
        self.button_insert.clicked.connect(self.button_insert_press)
        self.controls_layoutGrid.addWidget(self.button_insert, row, col)
        self.button_insert.setEnabled(self.cfg['button_insert'])
        row = row + 1
        
        self.button_remove = QPushButton()
        self.button_remove.setText('Remove')
        self.button_remove.clicked.connect(self.button_remove_press)
        self.controls_layoutGrid.addWidget(self.button_remove, row, col)
        self.button_remove.setEnabled(self.cfg['button_remove'])
        row = row + 1    
                
        self.button_label3d = QPushButton()
        self.button_label3d_status = False
        self.button_label3d.setStyleSheet("background-color: darkred;")
        self.button_label3d.setText('Label 3D')
        self.button_label3d.clicked.connect(self.button_label3d_press)
        self.controls_layoutGrid.addWidget(self.button_label3d, row, col)
        self.button_label3d.setEnabled(self.cfg['button_label3d'])
        col = col + 1
        
        self.button_previousLabel = QPushButton()
        self.button_previousLabel.setText('Previous Label (P)')
        self.button_previousLabel.clicked.connect(self.button_previousLabel_press)
        self.controls_layoutGrid.addWidget(self.button_previousLabel, row, col)
        self.button_previousLabel.setEnabled(self.cfg['button_previousLabel'])
        col = col + 1
        
        self.button_nextLabel = QPushButton()
        self.button_nextLabel.setText('Next Label (N)')
        self.button_nextLabel.clicked.connect(self.button_nextLabel_press)
        self.controls_layoutGrid.addWidget(self.button_nextLabel, row, col)
        self.button_nextLabel.setEnabled(self.cfg['button_nextLabel'])
        row = row + 1
        col = 0
        
        self.button_up = QPushButton()
        self.button_up.setText('Up (\u2191)')
        self.button_up.clicked.connect(self.button_up_press)
        self.controls_layoutGrid.addWidget(self.button_up, row, col)
        self.button_up.setEnabled(self.cfg['button_up'])
        col = col + 1
    
        self.button_right = QPushButton()
        self.button_right.setText('Right (\u2192)')
        self.button_right.clicked.connect(self.button_right_press)
        self.controls_layoutGrid.addWidget(self.button_right, row, col)
        self.button_right.setEnabled(self.cfg['button_right'])
        col = col + 1
        
        self.label_dxyz = QLabel()
        self.label_dxyz.setText('dxyz:')
        self.controls_layoutGrid.addWidget(self.label_dxyz, row, col)
        row = row + 1
        col = 0
        
        self.button_down = QPushButton()
        self.button_down.setText('Down (\u2193)')
        self.button_down.clicked.connect(self.button_down_press)
        self.controls_layoutGrid.addWidget(self.button_down, row, col)
        self.button_down.setEnabled(self.cfg['button_down'])
        col = col + 1
        
        self.button_left = QPushButton()
        self.button_left.setText('Left (\u2190)')
        self.button_left.clicked.connect(self.button_left_press)
        self.controls_layoutGrid.addWidget(self.button_left, row, col)
        self.button_left.setEnabled(self.cfg['button_left'])
        col = col + 1

        self.field_dxyz = QLineEdit()
        self.field_dxyz.setText(str(self.dxyz))
        self.field_dxyz.returnPressed.connect(self.field_dxyz_change)
        self.controls_layoutGrid.addWidget(self.field_dxyz, row, col)
        self.field_dxyz.setEnabled(self.cfg['field_dxyz'])
        row = row + 1
        col = 0
        

        # 2d
        self.button_previous = QPushButton()
        self.button_previous.setText('Previous Frame (A)')
        self.button_previous.clicked.connect(self.button_previous_press)
        self.controls_layoutGrid.addWidget(self.button_previous, row, col)
        self.button_previous.setEnabled(self.cfg['button_previous'])
        col = col + 1
        
        self.button_next = QPushButton()
        self.button_next.setText('Next Frame (D)')
        self.button_next.clicked.connect(self.button_next_press)
        self.controls_layoutGrid.addWidget(self.button_next, row, col)
        self.button_next.setEnabled(self.cfg['button_next'])
        col = col + 1
        
        self.button_home = QPushButton('Home (H)')
        self.button_home.clicked.connect(self.button_home_press)
        self.controls_layoutGrid.addWidget(self.button_home, row, col)
        self.button_home.setEnabled(self.cfg['button_home'])
        row = row + 1
        col = 0
        
        self.label_currentPose = QLabel()
        self.label_currentPose.setText('current frame:')
        self.controls_layoutGrid.addWidget(self.label_currentPose, row, col)
        col = col + 1
        
        self.label_dFrame = QLabel()
        self.label_dFrame.setText('dFrame:')
        self.controls_layoutGrid.addWidget(self.label_dFrame, row, col)
        col = col + 1

        self.button_zoom = QPushButton('Zoom (Z)')
        self.button_zoom.setStyleSheet("background-color: darkred;")
        self.button_zoom.clicked.connect(self.button_zoom_press)
        self.controls_layoutGrid.addWidget(self.button_zoom, row, col)
        self.button_zoom.setEnabled(self.cfg['button_zoom'])
        row = row + 1
        col = 0
        
        self.field_currentPose = QLineEdit()
        self.field_currentPose.setValidator(QIntValidator())
        self.field_currentPose.setText(str(self.i_pose))
        self.field_currentPose.returnPressed.connect(self.field_currentPose_change)
        self.controls_layoutGrid.addWidget(self.field_currentPose, row, col)
        self.field_currentPose.setEnabled(self.cfg['field_currentPose'])
        col = col + 1
        
        self.field_dFrame = QLineEdit()
        self.field_dFrame.setValidator(QIntValidator())
        self.field_dFrame.setText(str(self.dFrame))
        self.field_dFrame.returnPressed.connect(self.field_dFrame_change)
        self.controls_layoutGrid.addWidget(self.field_dFrame, row, col)
        self.field_dFrame.setEnabled(self.cfg['field_dFrame'])
        col = col + 1

        
        self.button_pan = QPushButton('Pan (W)')
        self.button_pan.setStyleSheet("background-color: darkred;")
        self.button_pan.clicked.connect(self.button_pan_press)
        self.controls_layoutGrid.addWidget(self.button_pan, row, col)
        
    
    
        self.frame_controls.setLayout(self.controls_layoutGrid)
    
    def button_loadRecording_press(self):
        dialog = QFileDialog()
        dialog.setStyleSheet("background-color: white;")
        dialogOptions = dialog.Options()
        dialogOptions |= dialog.DontUseNativeDialog
        recFileNames_unsorted, _ = QFileDialog.getOpenFileNames(dialog,
                                                                "Choose recording files",
                                                                "",
                                                                "video files (*.ccv, *.mp4, *.mkv)",
                                                                options=dialogOptions)
        if (len(recFileNames_unsorted) > 0):
            self.recFileNames = sorted(recFileNames_unsorted)
        
            self.nCameras = np.size(self.recFileNames, 0)
            self.xRes = np.zeros(self.nCameras, dtype=np.int64)
            self.yRes = np.zeros(self.nCameras, dtype=np.int64)
            self.nPoses = np.zeros(self.nCameras, dtype=np.int64)
            index = 0
            for fileName in self.recFileNames:
                if fileName:  
                    reader = imageio.get_reader(fileName)
                    header = read_video_meta(reader)
                    self.nPoses[index] = header['nFrames']
                    self.xRes[index] = header['sensorsize'][0]
                    self.yRes[index] = header['sensorsize'][1]
                    index = index + 1
                else:
                    print('ERROR: Invalid recording file')
                    raise
            self.imgs = list()
            for i_cam in range(self.nCameras):
                self.imgs.append(np.zeros((self.xRes[i_cam], self.yRes[i_cam]), dtype=np.int64))
            
            self.xLim_prev = np.zeros((self.nCameras, 2), dtype=np.float64)
            self.yLim_prev = np.zeros((self.nCameras, 2), dtype=np.float64)
            for i_cam in range(self.nCameras):
                self.xLim_prev[i_cam] = np.array([0.0, self.xRes[i_cam] - 1], dtype=np.float64)
                self.yLim_prev[i_cam] = np.array([0.0, self.yRes[i_cam] - 1], dtype=np.float64)
            self.maxPose = int(np.min([np.min(self.nPoses) - 1, self.cfg['maxPose']]))
    
            self.i_pose = 0
            if (int(index) == int(self.nCameras)):
                if (self.button_fastLabelingMode_status):
                    self.plot2d_drawFast_ini()
                else:
                    self.plot2d_drawNormal_ini()
                
                self.recordingIsLoaded = True
                self.button_loadRecording.setStyleSheet("background-color: green;")
                print('Loaded recording:')
                for i_rec in self.recFileNames:
                    print(i_rec)
                
            self.list_fastLabelingMode.currentIndexChanged.disconnect()
            self.list_fastLabelingMode.clear()
            self.list_fastLabelingMode.addItems([str(i) for i in range(self.nCameras)])
            self.list_fastLabelingMode.currentIndexChanged.connect(self.list_fastLabelingMode_change)

        self.button_loadRecording.clearFocus()                                          
    
    def button_loadCalibration_press(self):
        dialog = QFileDialog()
        dialog.setStyleSheet("background-color: white;")
        dialogOptions = dialog.Options()
        dialogOptions |= dialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(dialog,
                                                  "Choose calibration file",
                                                  ""
                                                  "npy files (*.npy)",
                                                  options=dialogOptions)
        if fileName:
            self.cameraSystem = calibcamlib.Camerasystem.from_calibcam_file(fileName)
            self.result = np.load(fileName, allow_pickle=True).item()
            A_val = self.result['A_fit']
            self.A = np.zeros((self.nCameras, 3, 3), dtype=np.float64)
            for i in range(self.nCameras):
                self.A[i][0, 0] = A_val[i, 0]
                self.A[i][0, 2] = A_val[i, 1]
                self.A[i][1, 1] = A_val[i, 2]
                self.A[i][1, 2] = A_val[i, 3]
                self.A[i][2, 2] = 1.0
            self.k = self.result['k_fit']
            self.rX1 = self.result['rX1_fit']
            self.RX1 = self.result['RX1_fit']
            self.tX1 = self.result['tX1_fit']
            self.origin = np.zeros(3, dtype=np.float64)
            self.coord = np.identity(3, dtype=np.float64)
            
            self.calibrationIsLoaded = True
            self.button_loadCalibration.setStyleSheet("background-color: green;")
            print('Loaded calibration ({:s})'.format(fileName))
        self.button_loadCalibration.clearFocus()

    def button_loadOrigin_press(self):
        dialog = QFileDialog()
        dialog.setStyleSheet("background-color: white;")
        dialogOptions = dialog.Options()
        dialogOptions |= dialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(dialog,
                                                  "Choose origin file",
                                                  ""
                                                  "npy files (*.npy)",
                                                  options=dialogOptions)
        if fileName:
            self.result = np.load(fileName, allow_pickle=True).item()
            self.origin = self.result['origin']
            self.coord = self.result['coord']
            
            self.originIsLoaded = True
            self.button_loadOrigin.setStyleSheet("background-color: green;")
            print('Loaded origin ({:s})'.format(fileName))
        self.button_loadOrigin.clearFocus()
        
    def button_loadModel_press(self):
        dialog = QFileDialog()
        dialog.setStyleSheet("background-color: white;")
        dialogOptions = dialog.Options()
        dialogOptions |= dialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(dialog,
                                                  "Choose model file",
                                                  ""
                                                  "npy files (*.npy)",
                                                  options=dialogOptions)
        if fileName:
            self.model = np.load(fileName, allow_pickle=True).item()
            self.v = self.model['v'] # surface
            self.f = self.model['f'] # surface
    #        verts = model['verts'] # skeleton
    #        edges = model['edges'] # skeleton
    #        v_link = model['v_link'] # rigid connection v to edges
    #        verts_link = model['verts_link'] # rigid connection verts to edges
    #        nEdges = np.size(edges, 0)
            self.v_center = np.mean(self.v, 0)
    
            if 'labels3d' in self.model:
                self.labels3d = copy.deepcopy(self.model['labels3d'])
                self.labels3d_sequence = sorted(list(self.labels3d.keys()))
                self.labels3d_sequence = sort_label_sequence(self.labels3d_sequence) # FIXME: comment when you want to label origin/coord, uncomment if you want to actually label something
            else:
                self.labels3d = dict()
                self.labels3d_sequence = list([])
                print('WARNING: Model does not contain 3D Labels! This might lead to incorrect behavior of the GUI.')
            
            
            self.list_labels3d.clear()
            self.list_labels3d.addItems(sorted(list(self.labels3d.keys())))
            
            self.modelIsLoaded = True
            
            self.label3d_select_status = False
            self.selectedLabel3d = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            self.selectedLabel2d = np.full((self.nCameras, 2), np.nan, dtype=np.float64)
            self.clickedLabel2d = np.array([np.nan, np.nan], dtype=np.float64)
                        
            self.plot3d_draw()
            self.button_loadModel.setStyleSheet("background-color: green;")
            print('Loaded model ({:s})'.format(fileName))
        self.button_loadModel.clearFocus()

    def button_saveModel_press(self):
        if (self.modelIsLoaded):
            dialog = QFileDialog()
            dialog.setStyleSheet("background-color: white;")
            dialogOptions = dialog.Options()
            dialogOptions |= dialog.DontUseNativeDialog
            fileName, _ = QFileDialog.getSaveFileName(dialog,
                                                      "Save model file",
                                                      ""
                                                      "npy files (*.npy)",
                                                      options=dialogOptions)
            if fileName:
                self.model['labels3d'] = copy.deepcopy(self.labels3d)
                np.save(fileName, self.model)
                print('Saved model ({:s})'.format(fileName))
        else:
            print('WARNING: Model needs to be loaded first')
        self.button_saveModel.clearFocus()

    def button_loadLabels_press(self):
        dialog = QFileDialog()
        dialog.setStyleSheet("background-color: white;")
        dialogOptions = dialog.Options()
        dialogOptions |= dialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(dialog,
                                                  "Choose labels file",
                                                  ""
                                                  "npz files (*.npz)",
                                                  options=dialogOptions)
        if fileName:
            #self.labels2d_all = np.load(fileName, allow_pickle=True).item()
            self.labels2d_all = np.load(fileName, allow_pickle=True)['arr_0'].item()
            self.standardLabelsFile = fileName;
            if self.i_pose in self.labels2d_all.keys():
                self.labels2d = copy.deepcopy(self.labels2d_all[self.i_pose])
            if (self.label3d_select_status):
                if (self.list_labels3d.currentItem().text() in self.labels2d.keys()):
                    self.selectedLabel2d = np.copy(self.labels2d[self.list_labels3d.currentItem().text()])
            
            self.labelsAreLoaded = True
            self.button_loadLabels.setStyleSheet("background-color: green;")
            print('Loaded labels ({:s})'.format(fileName))
        self.button_loadLabels.clearFocus()
    
    def button_saveLabels_press(self):
        if (self.master):
            dialog = QFileDialog()
            dialog.setStyleSheet("background-color: white;")
            dialogOptions = dialog.Options()
            dialogOptions |= dialog.DontUseNativeDialog
            print(self.standardLabelsFile)
            fileName, _ = QFileDialog.getSaveFileName(dialog,
                                                      "Save labels file",
                                                      os.path.dirname(self.standardLabelsFile),
                                                      "npz files (*.npz)",
                                                      options=dialogOptions)
            if fileName:
                if bool(self.labels2d):
                    self.labels2d_all[self.i_pose] = copy.deepcopy(self.labels2d)
                #np.save(fileName, self.labels2d_all)
                np.savez(fileName, self.labels2d_all)
                print('Saved labels ({:s})'.format(fileName))
        else:
            if bool(self.labels2d):
                self.labels2d_all[self.i_pose] = copy.deepcopy(self.labels2d)
            #np.save(self.standardLabelsFile, self.labels2d_all)
            np.savez(self.standardLabelsFile, self.labels2d_all)
            print('Saved labels ({:s})'.format(self.standardLabelsFile))
        self.button_saveLabels.clearFocus()

    def button_loadSketch_press(self):
        dialog = QFileDialog()
        dialog.setStyleSheet("background-color: white;")
        dialogOptions = dialog.Options()
        dialogOptions |= dialog.DontUseNativeDialog
        fileName_sketch, _ = QFileDialog.getOpenFileName(dialog,
                                                         "Choose sketch file",
                                                         ""
                                                         "npy files (*.npy)",
                                                         options=dialogOptions)
        if fileName_sketch:
            self.sketchFile = np.load(fileName_sketch, allow_pickle=True).item()
            self.sketch = self.sketchFile['sketch']
            self.sketch_annotation = self.sketchFile['sketch_label_locations']
            self.sketch_locations = list()
            for i_label in self.labels3d_sequence:
                self.sketch_locations.append(self.sketch_annotation[i_label])
            self.sketch_locations = np.array(self.sketch_locations, dtype=np.float64)
            self.sketchIsLoaded = True
            self.button_loadSketch.setStyleSheet("background-color: green;")
            print('Loaded sketch ({:s})'.format(fileName_sketch))
        self.button_loadSketch.clearFocus()
        
    def button_sketchMode_press(self):
        if (self.button_fastLabelingMode):
            if (self.modelIsLoaded):
                if (self.button_sketchMode_status):
                    self.button_sketchMode.setStyleSheet("background-color: darkred;")
                    self.button_sketchMode_status = not(self.button_sketchMode_status)
                    self.plot3d_draw()
                    self.figSketch.canvas.mpl_disconnect(self.cidSketch)
                else:
                    print('WARNING: Model needs to be loaded first')
            elif (self.sketchIsLoaded):
                if True:

                    self.button_sketchMode.setStyleSheet("background-color: green;")
                    self.button_sketchMode_status = not(self.button_sketchMode_status)
                    self.sketch_draw()
                    self.cidSketch = self.canvasSketch.mpl_connect('button_press_event',
                                                                    lambda event: self.sketch_click(event))
            else:
                print('WARNING: Sketch or model needs to be loaded first')
        else:
            print('WARNING: "Fast Labeling Mode" needs to be enabled to activate "Sketch Mode"')
        self.button_sketchMode.clearFocus()
    
    
    
    
    def button_insert_press(self):
        if ((not(np.any(np.isnan(self.clickedLabel3d)))) &
            (self.field_labels3d.text() != '')):
            if self.field_labels3d.text() in self.labels3d:
                print('WARNING: Label already exists')
            else:
                self.labels3d[self.field_labels3d.text()] = copy.deepcopy(self.clickedLabel3d)
                self.list_labels3d.addItems([self.field_labels3d.text()])
                self.clickedLabel3d = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
                self.field_labels3d.setText(str(''))
                self.plot3d_update()
        self.button_insert.clearFocus()
        self.field_labels3d.clearFocus()
        
    def button_remove_press(self):
        if (self.list_labels3d.currentItem() != None):
            self.label3d_select_status = False
            self.selectedLabel3d = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            self.selectedLabel2d = np.full((self.nCameras, 2), np.nan, dtype=np.float64)
            self.clickedLabel2d = np.array([np.nan, np.nan], dtype=np.float64)
            del(self.labels3d[self.list_labels3d.currentItem().text()])
            self.list_labels3d.takeItem(self.list_labels3d.currentRow())
            self.plot2d_update()
            self.plot3d_update()
        self.button_remove.clearFocus()
        
    def list_labels3d_select(self):
        if (self.cfg['autoSave'] and not(self.master)):
            self.autoSaveCounter = self.autoSaveCounter + 1
            if ((np.mod(self.autoSaveCounter, self.cfg['autoSaveN0']) == 0)):
                if bool(self.labels2d):
                    self.labels2d_all[self.i_pose] = copy.deepcopy(self.labels2d)
                #file = self.standardLabelsFolder + '/' + 'labels.npy' # this is equal to self.standardLabelsFile                
                #np.save(file, self.labels2d_all)
                #print('Automatically saved labels ({:s})'.format(file))
                file = self.standardLabelsFolder + '/' + 'labels.npz' # this is equal to self.standardLabelsFile                
                np.savez(file, self.labels2d_all)
                print('Automatically saved labels ({:s})'.format(file))
            if ((np.mod(self.autoSaveCounter, self.cfg['autoSaveN1']) == 0)):
                if bool(self.labels2d):
                    self.labels2d_all[self.i_pose] = copy.deepcopy(self.labels2d)
                #file = self.standardLabelsFolder + '/' + 'autosave' + '/' + 'labels.npy'
                #np.save(file, self.labels2d_all)
                #print('Automatically saved labels ({:s})'.format(file))
                file = self.standardLabelsFolder + '/' + 'autosave' + '/' + 'labels.npz'
                np.savez(file, self.labels2d_all)
                print('Automatically saved labels ({:s})'.format(file))
                #
                self.autoSaveCounter = 0

        self.label3d_select_status = True
        if (self.button_label3d_status):
            self.button_label3d_press()
        self.clickedLabel2d = np.array([np.nan, np.nan], dtype=np.float64)
        self.clickedLabel3d = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        self.selectedLabel3d = copy.deepcopy(self.labels3d[self.list_labels3d.currentItem().text()])
        if (self.list_labels3d.currentItem().text() in self.labels2d.keys()):
            self.selectedLabel2d = np.copy(self.labels2d[self.list_labels3d.currentItem().text()])
        else:
            self.selectedLabel2d = np.full((self.nCameras, 2), np.nan, dtype=np.float64)
        if (self.button_centricViewMode_status):
            self.plot2d_drawFast()
        self.plot2d_update()
        if (self.button_sketchMode_status):
            self.sketch_update()
        else:
            self.plot3d_update()
        self.list_labels3d.clearFocus()
        
    def button_fastLabelingMode_press(self):
        if (self.recordingIsLoaded):
            if (self.button_fastLabelingMode_status):
                if (self.button_centricViewMode_status):
                    self.button_centricViewMode_press()
                self.button_fastLabelingMode.setStyleSheet("background-color: darkred;")
                self.button_fastLabelingMode_status = not(self.button_fastLabelingMode_status)
                self.plot2d_drawNormal_ini()
            else:
                self.button_fastLabelingMode.setStyleSheet("background-color: green;")
                self.button_fastLabelingMode_status = not(self.button_fastLabelingMode_status)
                self.plot2d_drawFast_ini()
            if (self.button_sketchMode_status):
                self.sketch_update()
        else:
            print('WARNING: Recording needs to be loaded first')
        self.button_fastLabelingMode.clearFocus()
    
    # getting camera selection by directly accesing list_fastLabelingMode
    def list_fastLabelingMode_change(self):
        self.i_cam = self.list_fastLabelingMode.currentIndex()
        self.xLim_prev[self.i_cam] = np.array([0.0, self.xRes[self.i_cam] - 1], dtype=np.float64)
        self.yLim_prev[self.i_cam] = np.array([0.0, self.yRes[self.i_cam] - 1], dtype=np.float64)
        self.clickedLabel2d = np.array([np.nan, np.nan], dtype=np.float64)
        if (self.button_fastLabelingMode_status):
            self.plot2d_drawFast_ini()
        if (self.button_sketchMode_status):
            self.sketch_update()
        self.list_fastLabelingMode.clearFocus()
    
    def button_centricViewMode_press(self):
        if (self.button_fastLabelingMode_status):
            if not(self.list_labels3d.currentItem() == None):
                self.button_centricViewMode_status = not(self.button_centricViewMode_status)
#                 self.plot2d_drawFast_ini()
                self.plot2d_drawFast()
                if not(self.button_centricViewMode_status):
                    self.button_centricViewMode.setStyleSheet("background-color: darkred;")
                else:
                    self.button_centricViewMode.setStyleSheet("background-color: green;")
            else:
                print('WARNING: A label needs to be selected to activate "Centric View Mode"') 
        else:
            print('WARNING: "Fast Labeling Mode" needs to be enabled to activate "Centric View Mode"')
        self.button_centricViewMode.clearFocus()
    
    def button_reprojectionMode_press(self):
        if (self.calibrationIsLoaded):
            self.button_reprojectionMode_status = not(self.button_reprojectionMode_status)
            if (self.button_reprojectionMode_status):
                self.button_reprojectionMode.setStyleSheet("background-color: green;")
            else:
                self.button_reprojectionMode.setStyleSheet("background-color: darkred;")
            self.plot2d_update()
        else:
            print('WARNING: Calibration needs to be loaded first')
        self.button_reprojectionMode.clearFocus()
    
    def field_dx_change(self):
        try: 
            int(self.field_dx.text())
            fieldInputIsCorrect = True
        except ValueError:
            fieldInputIsCorrect = False  
        if (fieldInputIsCorrect):
            self.dx = int(np.max([8, int(self.field_dx.text())]))
        self.field_dx.setText(str(self.dx))
        if (self.button_fastLabelingMode_status):
            self.plot2d_drawFast()
        self.field_dx.clearFocus()
        
    def field_dy_change(self):
        try: 
            int(self.field_dy.text())
            fieldInputIsCorrect = True
        except ValueError:
            fieldInputIsCorrect = False  
        if (fieldInputIsCorrect):
            self.dy = int(np.max([8, int(self.field_dy.text())]))
        self.field_dy.setText(str(self.dy))
        if (self.button_fastLabelingMode_status):
            self.plot2d_drawFast()
        self.field_dy.clearFocus()
        
    def field_vmin_change(self):
        try: 
            int(self.field_vmin.text())
            fieldInputIsCorrect = True
        except ValueError:
            fieldInputIsCorrect = False  
        if (fieldInputIsCorrect):
            self.vmin = int(self.field_vmin.text())
            self.vmin = int(np.max([0, self.vmin]))
            self.vmin = int(np.min([self.vmin, 254]))
            self.vmin = int(np.min([self.vmin, self.vmax - 1]))
        if (self.button_fastLabelingMode_status):
            self.h_imgs[self.i_cam].set_clim(self.vmin, self.vmax)
            self.fig2d[self.i_cam].canvas.draw()
        else:
            for i_cam in range(self.nCameras):
                self.h_imgs[i_cam].set_clim(self.vmin, self.vmax)
                self.fig2d[i_cam].canvas.draw()
        self.field_vmin.setText(str(self.vmin))
        self.field_vmin.clearFocus()
        
    def field_vmax_change(self):
        try: 
            int(self.field_vmax.text())
            fieldInputIsCorrect = True
        except ValueError:
            fieldInputIsCorrect = False  
        if (fieldInputIsCorrect):
            self.vmax = int(self.field_vmax.text())
            self.vmax = int(np.max([1, self.vmax]))
            self.vmax = int(np.min([self.vmax, 255]))
            self.vmax = int(np.max([self.vmin + 1, self.vmax]))
        if (self.button_fastLabelingMode_status):
            self.h_imgs[self.i_cam].set_clim(self.vmin, self.vmax)
            self.fig2d[self.i_cam].canvas.draw()
        else:
            for i_cam in range(self.nCameras):
                self.h_imgs[i_cam].set_clim(self.vmin, self.vmax)
                self.fig2d[i_cam].canvas.draw()
        self.field_vmax.setText(str(self.vmax))
        self.field_vmax.clearFocus()
    
    def button_home_press(self):
        if (self.recordingIsLoaded):
            for i_cam in range(self.nCameras):
                self.xLim_prev[i_cam] = np.array([0.0, self.xRes[i_cam] - 1], dtype=np.float64)
                self.yLim_prev[i_cam] = np.array([0.0, self.yRes[i_cam] - 1], dtype=np.float64)
            if (self.button_fastLabelingMode_status):
                self.plot2d_drawFast()
            else:
                self.plot2d_drawNormal()
            for i in self.toolbars:
                i.home()
        if (self.modelIsLoaded):
            self.ax3d.mouse_init()
            self.ax3d.view_init(elev=None, azim=None)
            self.ax3d.set_xlim([self.v_center[0] - self.dxyz_lim,
                                self.v_center[0] + self.dxyz_lim])
            self.ax3d.set_ylim([self.v_center[1] - self.dxyz_lim,
                                self.v_center[1] + self.dxyz_lim])
            self.ax3d.set_zlim([self.v_center[2] - self.dxyz_lim,
                                self.v_center[2] + self.dxyz_lim])
            self.canvas3d.draw()
        self.button_home.clearFocus()
            
    def button_zoom_press(self,status=None,tostate=["on","off"]):
        if self.toolbars_zoom_status and "off" not in tostate:
            return
        if not self.toolbars_zoom_status and "on" not in tostate:
            return

        if (self.recordingIsLoaded):
            if (not(self.toolbars_zoom_status)):
                self.button_zoom.setStyleSheet("background-color: green;")
            else:
                self.button_zoom.setStyleSheet("background-color: darkred;")
            if (self.toolbars_pan_status):
                self.button_pan_press()
            for i in self.toolbars:
                i.zoom()
            self.toolbars_zoom_status = not(self.toolbars_zoom_status)
        else:
            print('WARNING: Recording needs to be loaded first')
        self.button_zoom.clearFocus()
            
    def button_pan_press(self):
        if (self.recordingIsLoaded):
            if (not(self.toolbars_pan_status)):
                self.button_pan.setStyleSheet("background-color: green;")
            else:
                self.button_pan.setStyleSheet("background-color: darkred;")
            if (self.toolbars_zoom_status):
                self.button_zoom_press()
            for i in self.toolbars:
                i.pan()
            self.selectedLabel2d = np.full((self.nCameras, 2), np.nan, dtype=np.float64)
            self.toolbars_pan_status = not(self.toolbars_pan_status)
        else:
            print('WARNING: Recording needs to be loaded first')
        self.button_pan.clearFocus()
        
    def button_label3d_press(self):
        if (self.modelIsLoaded):
            if not(self.button_sketchMode_status):
                if not(self.button_label3d_status):
                    self.label3d_select_status = False
                    self.selectedLabel2d = np.full((self.nCameras, 2), np.nan, dtype=np.float64)
                    self.selectedLabel3d = np.array([np.nan, np.nan, np.nan], dtype=np.float64)            
                    self.list_labels3d.setCurrentItem(None)
                    self.button_label3d.setStyleSheet("background-color: green;")
                    self.ax3d.disable_mouse_rotation()
                    self.cid = self.canvas3d.mpl_connect('button_press_event',
                                                         lambda event: self.plot3d_click(event))
                else:
                    self.button_label3d.setStyleSheet("background-color: darkred;")
                    self.ax3d.mouse_init()
                    self.fig3d.canvas.mpl_disconnect(self.cid)
        #            self.clickedLabel3d = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
                    self.field_labels3d.setText(str(''))
                self.button_label3d_status = not(self.button_label3d_status)
            else:
                print('WARNING: Sketch mode needs to be deactivated first') 
        else:
            print('WARNING: Model needs to be loaded first')
        self.button_label3d.clearFocus()
        
    def plot3d_click(self, event):
        if (event.button == 1):
            x = event.xdata
            y = event.ydata
            if ((x != None) & (y != None)):                
                s = self.ax3d.format_coord(event.xdata, event.ydata)
                s = s.split('=')
                s = [i.split(',') for i in s[1:]]
                xyz = [float(i[0]) for i in s]
                xyz = np.array(xyz, dtype=np.float64)
                
                M_proj = self.ax3d.get_proj()
                x2, y2, _ = proj3d.proj_transform(xyz[0], xyz[1], xyz[2],
                                                  M_proj)
                xv2, yv2, _ = proj3d.proj_transform(self.v[:, 0], self.v[:, 1], self.v[:, 2],
                                                    M_proj)
                
                diff = np.array([xv2 - x2, yv2 - y2], dtype=np.float64).T
                dist = np.sqrt(np.sum(diff**2, 1))
                self.selectedLabel3d = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
                self.clickedLabel3d = self.v[dist == np.min(dist)].squeeze()
                self.plot3d_update()
                self.ax3d.plot([self.clickedLabel3d[0]],
                               [self.clickedLabel3d[1]],
                               [self.clickedLabel3d[2]],
                               marker='o',
                               color='darkgreen')
                self.canvas3d.draw()
    
    def sketch_click(self, event):
        if (event.button == 1):
            x = event.xdata
            y = event.ydata
            if ((x != None) & (y != None)): 
                dists = ((x - self.sketch_locations[:, 0])**2 + (y - self.sketch_locations[:, 1])**2)**0.5
                label_index = np.argmin(dists)
                label_name = self.labels3d_sequence[label_index]
                sorted_index = sorted(list(self.labels3d.keys())).index(label_name)
                self.list_labels3d.setCurrentRow(sorted_index)
                self.list_labels3d_select()
        
    # this works correctly but implementation is somewhat messed up
    # (rows are dimensions and not the columns, c.f. commented plotting command)
    def plot3d_moveCenter(self, move_direc):
        x_lim = self.ax3d.get_xlim()
        y_lim = self.ax3d.get_ylim()
        z_lim = self.ax3d.get_zlim()
        center = np.array([np.mean(x_lim),
                           np.mean(y_lim),
                           np.mean(z_lim)], dtype=np.float64)
        dxzy_lim = np.mean([np.abs(center[0] - x_lim),
                            np.abs(center[1] - y_lim),
                            np.abs(center[2] - z_lim)])

        azim = self.ax3d.azim / 180 * np.pi + np.pi
        elev = self.ax3d.elev / 180 * np.pi
        
        r_azim = np.array([0.0, 0.0, -azim], dtype=np.float64)
        R_azim = rodrigues2rotMat_single(r_azim)
                        
        r_elev = np.array([0.0, -elev, 0.0], dtype=np.float64)
        R_elev = rodrigues2rotMat_single(r_elev)
        
        R_azim_elev = np.dot(R_elev, R_azim)
                
#        coord = center + R_azim_elev * 0.1
#        label = ['x', 'y', 'z']
#        for i in range(3):
#            self.ax3d.plot([center[0], coord[i, 0]],
#                           [center[1], coord[i, 1]],
#                           [center[2], coord[i, 2]],
#                           linestyle='-',
#                           color='green')
#            self.ax3d.text(coord[i, 0],
#                           coord[i, 1],
#                           coord[i, 2],
#                           label[i],
#                           color='red')
        
        center_new = center + np.sign(move_direc) * R_azim_elev[np.abs(move_direc), :] * self.dxyz
        self.ax3d.set_xlim([center_new[0] - dxzy_lim,
                            center_new[0] + dxzy_lim])
        self.ax3d.set_ylim([center_new[1] - dxzy_lim,
                            center_new[1] + dxzy_lim])
        self.ax3d.set_zlim([center_new[2] - dxzy_lim,
                            center_new[2] + dxzy_lim])
        self.canvas3d.draw()
    
    def button_up_press(self):
        move_direc = +2
        self.plot3d_moveCenter(move_direc)
        self.button_up.clearFocus()
        
    def button_down_press(self):
        move_direc = -2
        self.plot3d_moveCenter(move_direc)
        self.button_down.clearFocus()
        
    def button_left_press(self):
        move_direc = +1
        self.plot3d_moveCenter(move_direc)
        self.button_left.clearFocus()
        
    def button_right_press(self):
        move_direc = -1
        self.plot3d_moveCenter(move_direc)
        self.button_right.clearFocus()
        
    def field_dxyz_change(self):
        try: 
            float(self.field_dxyz.text())
            fieldInputIsCorrect = True
        except ValueError:
            fieldInputIsCorrect = False  
        if (fieldInputIsCorrect):
            self.dxyz = float(self.field_dxyz.text())
        self.field_dxyz.setText(str(self.dxyz))
        self.field_dxyz.clearFocus()

    def button_nextLabel_press(self):
        if (self.label3d_select_status):
            selected_label_name = self.list_labels3d.currentItem().text()
            selected_label_index = self.labels3d_sequence.index(selected_label_name)
            next_label_index = selected_label_index + 1
            if (next_label_index >= np.size(self.labels3d_sequence)):
                next_label_index = 0
            next_label_name = self.labels3d_sequence[next_label_index]
            sorted_index = sorted(list(self.labels3d.keys())).index(next_label_name) 
            self.list_labels3d.setCurrentRow(sorted_index)
            self.list_labels3d_select() 
        self.button_nextLabel.clearFocus()
        
    def button_previousLabel_press(self):
        if (self.label3d_select_status):
            selected_label_name = self.list_labels3d.currentItem().text()
            selected_label_index = self.labels3d_sequence.index(selected_label_name)
            previous_label_index = selected_label_index - 1
            if (previous_label_index < 0):
                previous_label_index = np.size(self.labels3d_sequence) - 1
            previous_label_name = self.labels3d_sequence[previous_label_index]
            sorted_index = sorted(list(self.labels3d.keys())).index(previous_label_name)
            self.list_labels3d.setCurrentRow(sorted_index)
            self.list_labels3d_select() 
        self.button_previousLabel.clearFocus()
        
    def button_next_press(self):
        if bool(self.labels2d):
            self.labels2d_all[self.i_pose] = copy.deepcopy(self.labels2d)
        self.i_pose = int(np.min([self.maxPose, self.i_pose + self.dFrame]))
        self.plot2d_changeFrame()
        self.button_next.clearFocus()
        
    def button_previous_press(self):
        if bool(self.labels2d):
            self.labels2d_all[self.i_pose] = copy.deepcopy(self.labels2d)
        self.i_pose = int(np.max([self.minPose, self.i_pose - self.dFrame]))
        self.plot2d_changeFrame()
        self.button_previous.clearFocus()
        
    def field_currentPose_change(self):
        try: 
            int(self.field_currentPose.text())
            fieldInputIsCorrect = True
        except ValueError:
            fieldInputIsCorrect = False  
        if (fieldInputIsCorrect):
            if bool(self.labels2d):
                self.labels2d_all[self.i_pose] = copy.deepcopy(self.labels2d)
            self.i_pose = int(self.field_currentPose.text())
            self.i_pose = int(np.max([self.minPose, self.i_pose]))
            self.i_pose = int(np.min([self.maxPose, self.i_pose]))
            self.plot2d_changeFrame()
        self.field_currentPose.setText(str(self.i_pose))
        if (self.button_sketchMode_status):
            first_label_name = self.labels3d_sequence[0]
            sorted_index = sorted(list(self.labels3d.keys())).index(first_label_name)
            self.list_labels3d.setCurrentRow(sorted_index)
            self.list_labels3d_select()
            self.sketch_update()
        self.field_currentPose.clearFocus()
        
    def field_dFrame_change(self):
        try: 
            int(self.field_dFrame.text())
            fieldInputIsCorrect = True
        except ValueError:
            fieldInputIsCorrect = False  
        if (fieldInputIsCorrect):
            self.dFrame = int(np.max([1, int(self.field_dFrame.text())]))
            for i_cam in range(self.nCameras):
                self.xLim_prev[i_cam] = np.array([0.0, self.xRes[i_cam] - 1], dtype=np.float64)
                self.yLim_prev[i_cam] = np.array([0.0, self.yRes[i_cam] - 1], dtype=np.float64)
        self.field_dFrame.setText(str(self.dFrame))
        self.field_dFrame.clearFocus()
    
    def plot2d_changeFrame(self):
        if (self.button_fastLabelingMode_status):
            self.xLim_prev[self.i_cam] = self.ax2d[self.i_cam].get_xlim()
            self.yLim_prev[self.i_cam] = self.ax2d[self.i_cam].get_ylim()#[::-1]
        else:
            for i_cam in range(self.nCameras):
                self.xLim_prev[i_cam] = self.ax2d[i_cam].get_xlim()
                self.yLim_prev[i_cam] = self.ax2d[i_cam].get_ylim()#[::-1]
        if (np.abs(self.clickedLabel2d_pose - self.i_pose) > self.dFrame):
            self.clickedLabel2d = np.array([np.nan, np.nan], dtype=np.float64)
        self.selectedLabel2d = np.full((self.nCameras, 2), np.nan, dtype=np.float64)
        self.field_currentPose.setText(str(self.i_pose))
        if self.i_pose in self.labels2d_all.keys():
            self.labels2d = copy.deepcopy(self.labels2d_all[self.i_pose])
        else:
            self.labels2d = dict()
        if (self.label3d_select_status):
            label_name = self.list_labels3d.currentItem().text()
            if (label_name in self.labels2d.keys()):
                self.selectedLabel2d = np.copy(self.labels2d[label_name])
        if (self.button_sketchMode_status):
            first_label_name = self.labels3d_sequence[0]
            sorted_index = sorted(list(self.labels3d.keys())).index(first_label_name)
            self.list_labels3d.setCurrentRow(sorted_index)
            self.list_labels3d_select()
            self.sketch_update()
        else:
            self.plot3d_update()
        if (self.button_fastLabelingMode_status):
            self.plot2d_drawFast()
        else:
            self.plot2d_drawNormal()
    
    
    def closeEvent(self, event):
        if (self.cfg['exitSaveModel']):
            self.button_saveModel_press()
        if (self.cfg['exitSaveLabels']):
            self.button_saveLabels_press()
        
        if not(self.master):
            exit_status = dict()
            exit_status['i_pose'] = self.i_pose
            np.save(self.standardLabelsFolder + '/' + 'exit_status.npy', exit_status)
        
        
    # shortkeys
    def keyPressEvent(self, event):
        if not(event.isAutoRepeat()):
            if (self.cfg['button_next'] and event.key() == Qt.Key_D):
                self.button_next_press()
            elif (self.cfg['button_previous'] and event.key() == Qt.Key_A):
                self.button_previous_press()
            elif (self.cfg['button_nextLabel'] and event.key() == Qt.Key_N):
                self.button_nextLabel_press()
            elif (self.cfg['button_previousLabel'] and event.key() == Qt.Key_P):
                self.button_previousLabel_press()
            elif(self.cfg['button_home'] and event.key() == Qt.Key_H):
                self.button_home_press()
            elif(self.cfg['button_zoom'] and event.key() == Qt.Key_Z):
                self.button_zoom_press()
            elif(self.cfg['button_pan'] and event.key() == Qt.Key_W):
                self.button_pan_press()
            elif(self.cfg['button_label3d'] and event.key() == Qt.Key_L):
                self.button_label3d_press()
            elif (self.cfg['button_up'] and event.key() == Qt.Key_Up):
                self.button_up_press()
            elif (self.cfg['button_down'] and event.key() == Qt.Key_Down):
                self.button_down_press()
            elif (self.cfg['button_left'] and event.key() == Qt.Key_Left):
                self.button_left_press()
            elif (self.cfg['button_right'] and event.key() == Qt.Key_Right):
                self.button_right_press()
            elif (self.cfg['button_centricViewMode'] and event.key() == Qt.Key_C):
                self.button_centricViewMode_press()
            elif (self.cfg['button_saveLabels'] and event.key() == Qt.Key_S):
                self.button_saveLabels_press()
            elif (self.cfg['field_vmax'] and event.key() == Qt.Key_Plus):
                self.field_vmax.setText(str(int(self.vmax*0.8)))
                self.field_vmax_change()
            elif (self.cfg['field_vmax'] and event.key() == Qt.Key_Minus):
                self.field_vmax.setText(str(int(self.vmax/0.8)))
                self.field_vmax_change()
            elif event.key() == Qt.Key_1:
                if self.nCameras>0:
                    self.list_fastLabelingMode.setCurrentIndex(0)
            elif event.key() == Qt.Key_2:
                if self.nCameras>1:
                    self.list_fastLabelingMode.setCurrentIndex(1)
            elif event.key() == Qt.Key_3:
                if self.nCameras>2:
                    self.list_fastLabelingMode.setCurrentIndex(2)
            elif event.key() == Qt.Key_4:
                if self.nCameras>3:
                    self.list_fastLabelingMode.setCurrentIndex(3)
            elif event.key() == Qt.Key_5:
                if self.nCameras>4:
                    self.list_fastLabelingMode.setCurrentIndex(4)
            elif event.key() == Qt.Key_6:
                if self.nCameras>5:
                    self.list_fastLabelingMode.setCurrentIndex(5)
                
        else:
            print('WARNING: Auto-repeat is not supported')

def main(configFile=None,master=True,drive=None):
    app = QApplication(sys.argv)
    window = MainWindow(fileConfig=configFile,master=master,drive=drive)
    sys.exit(app.exec_())          
        
        
        
if __name__ == '__main__':
    main()

class UnsupportedFormatException(Exception):
    pass
