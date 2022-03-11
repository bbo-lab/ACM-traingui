{
# FILES
'autoLoad': True,
'standardModelFile':       os.path.dirname(os.path.abspath(path))+'/../../../data/model/model.npy',
'standardSketchFile':      os.path.dirname(os.path.abspath(path))+'/../../../data/sketch/rouse.npy',
'standardRecordingFolder': os.path.dirname(os.path.abspath(path))+'/../../../data/ccv/M220217_DW03',
'standardRecordingFileNames': list([
                                    'cam01VI_20220217_3.ccv',
                                    'cam02VI_20220217_3.ccv',
                                    'cam03VI_20220217_3.ccv',
                                    'cam04VI_20220217_3.ccv',
                                   ]),
#
'standardCalibrationFile': '',
'standardOriginCoordFile': '',
'standardLabelsFile': '',
'invert_xaxis': True,
'invert_yaxis': False,
# GENERAL
'cam': int(0),
'minPose': int(70200),
'maxPose': int(80000),
'dFrame': int(200),
# mode activation
'button_fastLabelingMode_activate': True,
'button_centricViewMode_activate': False,
'button_reprojectionMode_activate': False,
'button_sketchMode_activate': True,
# save settings
'exitSaveModel': False,
'exitSaveLabels': True,
'autoSave': True,
'autoSaveN0': int(10),
'autoSaveN1': int(100),
# BUTTONS
# general
'button_loadRecording': False,
'button_loadModel': False,
'button_loadLabels': False,
'button_loadCalibration': False,
'button_saveModel': False,
'button_saveLabels': True,
'button_loadOrigin': False,
'button_loadSketch': False,
'button_sketchMode': False,
'button_fastLabelingMode': False,
'button_centricViewMode': False,
'button_reprojectionMode': False,
# labels
'button_insert': False,
'button_remove': False,
'button_label3d': False,
'button_previousLabel': True,
'button_nextLabel': True,
# 3d
'button_up': False,
'button_right': False,
'button_down': False,
'button_left': False,
# 2d
'button_next': True,
'button_previous': True,
# 3d and 2d
'button_home': True,
'button_zoom': True,
'button_pan': True,
# FIELDS
'field_dx': False,
'field_dy': False,
'field_vmin': True,
'field_vmax': True,
'field_labels3d': False,
'field_dxyz': False,
'field_currentPose': False,
'field_dFrame': False,
# LISTS
'list_fastLabelingMode': True
}
