import numpy as np


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
