import carla_utils as cu

import os, sys
from os.path import join
import time
import numpy as np

from . import data, data_augment


class DataMaster(object):
    imu_height = 0.45   ### imu height w.r.t. ground
    trajectory_length = 30.0
    type_threshold = 0.17  ### delta yaw

    def __init__(self, path, skim_time=cu.basic.Data(start=0.0, end=0.0), pc_type='png'):
        self.path = path
        self.name = os.path.split(path)[1]
        self.save_path = join(path, 'augment')
        cu.system.mkdir(self.save_path)


        '''data'''
        self.gps = data.GlobalPositionSystem(path)
        self.ins = data.InertialNavigationSystem(path)
        self.vo = data.VisualOdometry(path)
        self.ro = data.RadarOdometry(path)


        '''reference timestamps'''
        t1 = time.time()
        max_timestamp = min(
                            # self.velodyne_left_timestamp_array[-1],
                            # self.velodyne_right_timestamp_array[-1],
                            # self.stereo_centre_timestamp_array[-1],
                            self.ins.timestamps[-1],
                            self.vo.timestamps[-1],
                            self.ro.timestamps[-1],
                        ) -skim_time.end*1e6
        min_timestamp = max(
                            # self.velodyne_left_timestamp_array[0],
                            # self.velodyne_right_timestamp_array[0],
                            # self.stereo_centre_timestamp_array[0],
                            self.ins.timestamps[0],
                            self.vo.timestamps[0],
                            self.ro.timestamps[0],
                        ) +skim_time.start*1e6
        mask = np.argwhere((self.ro.timestamps >= min_timestamp)&(self.ro.timestamps <= max_timestamp))[:,0]
        self.ref_timestamp_array = self.ro.timestamps[mask]
        t2 = time.time()
        print('[{}] {} generate time (reference.timestamps): '.format(self.__class__.__name__, self.name), t2-t1)
        print()

        self.delta_t = np.average(np.diff(1e-6* self.ref_timestamp_array))
        return


    def __len__(self):
        return len(self.ref_timestamp_array)



    def init_augment_data(self):
        self.pose_velocity = data_augment.PoseVelocity(self.path, self.ref_timestamp_array, self.ro, self.ins, self.imu_height)

        print()
        return




