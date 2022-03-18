{
# FILES - WARNING: Filenames must NOT contain \ (not even from Python functions like os.dirname(__file__) on Windows)!
'autoLoad': True,
'standardModelFile': datadir+'/model/model.npy', # datadir translates to [currentfile]/../../../data
'standardSketchFile': datadir+'/sketch/20210511_sketch_1_ds.npy',
'standardRecordingFolder': datadir+'/ccv/M220217_DW01',
'standardRecordingFileNames': list([
                                    'cam01VI_20220217_1.ccv',
                                    'cam02VI_20220217_1.ccv',
                                    'cam03VI_20220217_1.ccv',
                                    'cam04VI_20220217_1.ccv',
                                   ]),
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
