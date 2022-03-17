import carla_utils as cu

import os, sys
from os.path import join
import time
import numpy as np

from . import data, data_augment


class DataMaster(object):
    imu_height = 0.45   ### imu height w.r.t. ground

    ### trajectory
    trajectory_time = 5.0
    max_speed = 12.2
    max_length = max_speed * trajectory_time
    num_points = 20

    skip_time = cu.basic.Data(start=0.0, end=trajectory_time)


    def __init__(self, path, pc_type='png'):
        self.path = path
        self.name = os.path.split(path)[1]
        self.save_path = join(path, 'augment')
        cu.system.mkdir(self.save_path)


        '''data'''
        self.gps = data.GlobalPositionSystem(path)
        self.ins = data.InertialNavigationSystem(path)
        self.vo = data.VisualOdometry(path)
        self.ro = data.RadarOdometry(path)
        self.stereo_centre = data.StereoCentre(path)


        '''reference timestamps'''
        t1 = time.time()
        max_timestamp = min(
                            # self.velodyne_left_timestamp_array[-1],
                            # self.velodyne_right_timestamp_array[-1],
                            self.stereo_centre.timestamps[-1],
                            self.ins.timestamps[-1],
                            self.vo.timestamps[-1],
                            self.ro.timestamps[-1],
                        ) -self.skip_time.end*1e6
        min_timestamp = max(
                            # self.velodyne_left_timestamp_array[0],
                            # self.velodyne_right_timestamp_array[0],
                            self.stereo_centre.timestamps[0],
                            self.ins.timestamps[0],
                            self.vo.timestamps[0],
                            self.ro.timestamps[0],
                        ) +self.skip_time.start*1e6
        mask = np.argwhere((self.ro.timestamps >= min_timestamp)&(self.ro.timestamps <= max_timestamp))[:,0]
        self.timestamps = self.ro.timestamps
        self.ref_timestamp_array = self.ro.timestamps[mask]
        t2 = time.time()
        print(cu.basic.prefix(self) + self.name + ' generate time (reference.timestamps): ', t2-t1)
        print()

        self.delta_t = np.average(np.diff(1e-6* self.ref_timestamp_array))
        return


    def __len__(self):
        return len(self.ref_timestamp_array)



    def init_augment_data(self):
        self.pose_velocity = data_augment.PoseVelocity(self.path, self.timestamps, self.ro, self.ins, self.imu_height)
        self.stereo_centre_augment = data_augment.StereoCentreAugment(self.path, self.timestamps)

        print()


        ### trajectory
        self.num_trajectory_points = int(np.around(self.trajectory_time / self.delta_t))
        assert self.num_trajectory_points % self.num_points == 0
        self.sample_interval = int(self.num_trajectory_points / self.num_points)

        return


    def get_trajectory(self, index: int):
        # timestamps = self.ref_timestamp_array[index:index+self.num_trajectory_points]
        timestamps = self.timestamps[index:index+self.num_trajectory_points]

        x, y, yaw, vx, vy = self.pose_velocity.data[:,
            index:index+self.num_trajectory_points:self.sample_interval]  ##! warning check
        
        x, y = x - x[0], y - y[0]
        R = cu.basic.RotationMatrix2D(yaw[0]).astype(np.float32)
        x, y = np.dot(R.T, np.vstack((x, y)))
        vx, vy = np.dot(R.T, np.vstack((vx, vy)))
        yaw = cu.basic.pi2pi(yaw - yaw[0])

        times = timestamps - timestamps[0]
        times = times * 1e-6

        # import matplotlib.pyplot as plt
        # plt.figure(1)
        # plt.subplot(411)
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.plot(x, y, '-r')

        # plt.subplot(412)
        # plt.plot(times, vx, 'og')

        # plt.subplot(413)
        # plt.plot(times, vy, 'ob')

        # plt.subplot(414)
        # plt.plot(times, np.rad2deg(yaw), 'oc')

        # plt.show()

        return cu.basic.Data(times=times, x=x, y=y, vx=vx, vy=vy, yaw=yaw)


