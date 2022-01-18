{
# FILES
'autoLoad': True,
# ARENA 05:
'standardModelFile': 'labeling_model.npy',
'standardSketchFile': '20200205_sketch.npy',
'standardRecordingFolder': '/media/smb/soma.ad01.caesar.de/bbo/analysis/pose/data/ccv/20200205_arena',
'standardRecordingFileNames': list(['cam1_arena.ccv',
                                    'cam2_arena.ccv',
                                    'cam3_arena.ccv',
                                    'cam4_arena.ccv']),
'standardCalibrationFile': 'multicalibration.npy',
'standardOriginCoordFile': 'origin_coord.npy',
'standardLabelsFile': 'labels.npy',
'invert_xaxis': False,
'invert_yaxis': True,
# GENERAL
'cam': int(0),
'minPose': int(0),
'maxPose': int(1e9),
'dFrame': int(50),
# mode activation
'button_fastLabelingMode_activate': False,
'button_centricViewMode_activate': False,
'button_reprojectionMode_activate': False,
'button_sketchMode_activate': True,
# save settings
'exitSaveModel': False,
'exitSaveLabels': False,
'autoSave': False,
'autoSaveN0': int(10),
'autoSaveN1': int(100),
# BUTTONS
# general
'button_loadRecording': True,
'button_loadModel': True,
'button_loadLabels': True,
'button_loadCalibration': True,
'button_saveModel': True,
'button_saveLabels': True,
'button_loadOrigin': True,
'button_loadSketch': True,
'button_sketchMode': True,
'button_fastLabelingMode': True,
'button_centricViewMode': True,
'button_reprojectionMode': True,
# labels
'button_insert': True,
'button_remove': True,
'button_label3d': True,
'button_previousLabel': True,
'button_nextLabel': True,
# 3d
'button_up': True,
'button_right': True,
'button_down': True,
'button_left': True,
# 2d
'button_next': True,
'button_previous': True,
# 3d and 2d
'button_home': True,
'button_zoom': True,
'button_pan': True,
# FIELDS
'field_dx': True,
'field_dy': True,
'field_vmin': True,
'field_vmax': True,
'field_labels3d': True,
'field_dxyz': True,
'field_currentPose': True,
'field_dFrame': True,
# LISTS
'list_fastLabelingMode': True
}
