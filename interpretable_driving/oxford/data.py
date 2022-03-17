import carla_utils as cu

import os
from os.path import join
import numpy as np
import pandas as pd



def find_nearest(timestamp, timestamp_array):
    idx = (np.abs(timestamp_array - timestamp)).argmin()
    return timestamp_array[idx]


class Data(object):
    def __init__(self, path):
        self.path = path
        self.dataset_name = os.path.split(path)[1]




class GlobalPositionSystem(Data):
    def __init__(self, path):
        super().__init__(path)

        self.data = pd.read_csv(join(path, 'gps', 'gps.csv'))



class InertialNavigationSystem(Data):
    def __init__(self, path):
        super().__init__(path)

        self.data = pd.read_csv(join(path, 'gps', 'ins.csv'))
        self.timestamps = self.data['timestamp'].values
        self.delta_t = np.average(np.diff(1e-6* self.timestamps))
        # print(cu.basic.prefix(self) + self.dataset_name + ' freq: ', 1/ self.delta_t)

    def __getitem__(self, key):
        return self.data[key]



class VisualOdometry(Data):
    def __init__(self, path):
        super().__init__(path)

        self.data = pd.read_csv(join(path, 'vo', 'vo.csv'))
        self.timestamps = self.data['destination_timestamp'].values
        self.delta_t = np.average(np.diff(1e-6* self.timestamps))
        # print(cu.basic.prefix(self) + self.dataset_name + ' freq: ', 1/ self.delta_t)



class RadarOdometry(Data):
    def __init__(self, path):
        super().__init__(path)

        self.data = pd.read_csv(join(path, 'gt', 'radar_odometry.csv'))
        self.timestamps = self.data['destination_timestamp'].values
        self.delta_t = np.average(np.diff(1e-6* self.timestamps))
        # print(cu.basic.prefix(self) + self.dataset_name + ' freq: ', 1/ self.delta_t)

    def __getitem__(self, key):
        return self.data[key]



class StereoCentre(Data):
    def __init__(self, path):
        super().__init__(path)

        self.data_path = join(path, 'stereo/centre')
        self.timestamps = np.loadtxt(join(path, 'stereo.timestamps'), delimiter=' ', usecols=[0], dtype=np.int64)
        self.delta_t = np.average(np.diff(1e-6* self.timestamps))
        # print(cu.basic.prefix(self) + self.dataset_name + ' freq: ', 1/ self.delta_t)






