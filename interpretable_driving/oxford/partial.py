import carla_utils as cu

import os, sys
from os.path import join


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import random
import cv2
import copy
import time

from ..basic import utils as basic_utils

# from .tools import load_velodyne_binary, load_velodyne_png


def find_nearest(timestamp, timestamp_array):
    idx = (np.abs(timestamp_array - timestamp)).argmin()
    return timestamp_array[idx]


class PartialDataset(object):
    imu_height = 0.45   ### imu height w.r.t. ground
    trajectory_length = 30.0
    type_threshold = 0.17  ### delta yaw

    def __init__(self, path, skim_time=cu.basic.Data(start=0.0, end=0.0), pc_type='png'):
        self.path = path
        self.name = os.path.split(path)[1]
        self.save_path = join(path, 'augment')
        cu.system.mkdir(self.save_path)

        '''lidar'''
        # self.velodyne_left_timestamp_array = np.loadtxt(join(path, 'velodyne_left.timestamps'), delimiter=' ', usecols=[0], dtype=np.int64)
        # self.velodyne_right_timestamp_array= np.loadtxt(join(path, 'velodyne_right.timestamps'),delimiter=' ', usecols=[0], dtype=np.int64)
        # (postfix, func) = ('.png', load_velodyne_png) if pc_type == 'png' else ('.bin', load_velodyne_binary)
        # self.velodyne_left_data = lambda timestamp: func(join(path, 'velodyne_left', str(timestamp)+postfix))
        # self.velodyne_right_data= lambda timestamp: func(join(path, 'velodyne_right',str(timestamp)+postfix))

        '''stereo centre'''
        # self.stereo_centre_timestamp_array = np.loadtxt(join(path, 'stereo.timestamps'), delimiter=' ', usecols=[0], dtype=np.int64)
        # self.stereo_centre_data = lambda timestamp: cv2.imread(join(path, 'stereo/centre', str(timestamp)+'.png'), cv2.IMREAD_GRAYSCALE)

        '''gps'''
        self.gps_df = pd.read_csv(join(path, 'gps', 'gps.csv'))
        self.ins_df = pd.read_csv(join(path, 'gps', 'ins.csv'))
        self.ins_timestamp_array = self.ins_df['timestamp'].values

        '''vo'''
        self.vo_df = pd.read_csv(join(path, 'vo', 'vo.csv'))

        '''ro'''
        self.ro_df = pd.read_csv(join(path, 'gt', 'radar_odometry.csv'))


        # import pdb; pdb.set_trace()


        '''reference timestamps'''
        t1 = time.time()
        max_timestamp = min(
                            # self.velodyne_left_timestamp_array[-1],
                            # self.velodyne_right_timestamp_array[-1],
                            # self.stereo_centre_timestamp_array[-1],
                            self.ins_df['timestamp'].values[-1],
                            self.vo_df['destination_timestamp'].values[-1],
                            self.ro_df['destination_timestamp'].values[-1],
                        ) -skim_time.end*1e6
        min_timestamp = max(
                            # self.velodyne_left_timestamp_array[0],
                            # self.velodyne_right_timestamp_array[0],
                            # self.stereo_centre_timestamp_array[0],
                            self.ins_df['timestamp'].values[0],
                            self.vo_df['destination_timestamp'].values[0],
                            self.ro_df['destination_timestamp'].values[0],
                        ) +skim_time.start*1e6
        # mask = np.argwhere((self.stereo_centre_timestamp_array >= min_timestamp)&(self.stereo_centre_timestamp_array <= max_timestamp))[:,0]
        # self.ref_timestamp_array = self.stereo_centre_timestamp_array[mask]
        mask = np.argwhere((self.ins_timestamp_array >= min_timestamp)&(self.ins_timestamp_array <= max_timestamp))[:,0]
        self.ref_timestamp_array = self.ins_timestamp_array[mask]
        # np.savetxt(join(self.save_path, 'reference.timestamps'), self.ref_timestamp_array, delimiter=' ', fmt='%d')
        t2 = time.time()
        print('[PartialDataset] {} generate time (reference.timestamps): '.format(self.name), t2-t1)


        self.delta_t = np.average(np.diff(1e-6* self.ref_timestamp_array))
        return


    def __len__(self):
        return len(self.ref_timestamp_array)
    

    def generate_augment_data(self):
        '''ins_2d'''
        try:
            self.ins_2d = np.loadtxt(join(self.save_path, 'ins_2d.txt'), delimiter=' ', usecols=[], dtype=np.float64).T
        except IOError:
            t1 = time.time()
            min_timestamp, max_timestamp = min(self.ref_timestamp_array), max(self.ref_timestamp_array)
            df = self.ins_df[(self.ins_df['timestamp'] >= min_timestamp) & (self.ins_df['timestamp'] <= max_timestamp)]
            x, y = df['northing'].values, df['easting'].values
            vx, vy = df['velocity_north'].values, df['velocity_east'].values
            yaw = cu.basic.pi2pi(df['yaw'].values)

            x, y = x - x[0], y - y[0]
            R = cu.basic.RotationMatrix2D(yaw[0])
            x, y = np.dot(R.T, np.vstack((x, y)))
            vx, vy = np.dot(R.T, np.vstack((vx, vy)))
            yaw = cu.basic.pi2pi(yaw - yaw[0])

            self.ins_2d = np.vstack([x, y, vx, vy, yaw])
            np.savetxt(join(self.save_path, 'ins_2d.txt'), self.ins_2d.T, delimiter=' ', fmt='%f')
            t2 = time.time()
            print('[PartialDataset] {} save time (ins_2d): '.format(self.name), t2-t1)


        '''gt_ro'''
        try:
            self.gt_ro = np.loadtxt(join(self.save_path, 'gt_ro.txt'), delimiter=' ', usecols=[], dtype=np.float64).T
        except IOError:
            t1 = time.time()
            min_timestamp, max_timestamp = min(self.ref_timestamp_array), max(self.ref_timestamp_array)
            df = self.ro_df[(self.ro_df['destination_timestamp'] >= min_timestamp) & (self.ro_df['destination_timestamp'] <= max_timestamp)]
            x, y = df['x'].values, df['y'].values
            x_array, y_array, z_array = df['x'].values, df['y'].values, df['z'].values

            delta_pose_array = np.vstack((x_array, y_array, z_array, df['roll'].values, df['pitch'].values, df['yaw'].values))
            pose_array = cum_vo(delta_pose_array, self.imu_height)
            x, y = pose_array[0], pose_array[1]
            yaw = cu.basic.pi2pi(pose_array[-1])

            self.gt_ro = np.vstack([x, y, yaw])
            np.savetxt(join(self.save_path, 'gt_ro.txt'), self.gt_ro.T, delimiter=' ', fmt='%f')
            t2 = time.time()
            print('[PartialDataset] {} save time (gt_ro): '.format(self.name), t2-t1)





        '''trajectory'''
        try:
            self.lengths = np.loadtxt(join(self.save_path, 'lengths.txt'), delimiter=' ', usecols=[], dtype=np.float64)
            self.delta_pose_array = np.loadtxt(join(self.save_path, 'delta_pose_array.txt'), delimiter=' ', usecols=[], dtype=np.float64).T
        except IOError:
            t1 = time.time()
            min_timestamp, max_timestamp = min(self.ref_timestamp_array), max(self.ref_timestamp_array)
            df = self.vo_df[(self.vo_df['destination_timestamp'] >= min_timestamp) & (self.vo_df['destination_timestamp'] <= max_timestamp)]
            x_array, y_array, z_array = df['x'].values, df['y'].values, df['z'].values
            self.lengths = np.sqrt(x_array**2 + y_array**2 + z_array**2)
            self.delta_pose_array = np.vstack((x_array, y_array, z_array, df['roll'].values, df['pitch'].values, df['yaw'].values))

            np.savetxt(join(self.save_path, 'lengths.txt'), self.lengths, delimiter=' ', fmt='%f')
            np.savetxt(join(self.save_path, 'delta_pose_array.txt'), self.delta_pose_array.T, delimiter=' ', fmt='%f')
            t2 = time.time()
            print('[PartialDataset] {} save time (lengths and delta_pose_array): '.format(self.name), t2-t1)

        '''trajectory type'''
        try:
            self.trajectory_types = np.loadtxt(join(self.save_path, 'trajectory_types.txt'), delimiter=' ', usecols=[], dtype=np.int64)
        except IOError:
            t1 = time.time()
            self.trajectory_types = self.generate_trajectory_types()
            np.savetxt(join(self.save_path, 'trajectory_types.txt'), self.trajectory_types, delimiter=' ', fmt='%d')
            t2 = time.time()
            print('[PartialDataset] {} save time (trajectory_types): '.format(self.name), t2-t1)


        return


    def generate_pose_array(self, ref_timestamp, horizon=None):
        index = int(np.argwhere((self.ref_timestamp_array == ref_timestamp)))
        # if horizon is None:
        if horizon == None:
            horizon = basic_utils.find_desired_length(self.lengths, index, self.trajectory_length)
        if horizon <= 0:
            print('no enough trajectory length')
            horizon = 0
        delta_pose_array = self.delta_pose_array[:,index:index+horizon]

        import pdb; pdb.set_trace()

        pose_array = cum_vo(delta_pose_array, self.imu_height)[:,:-1]
        return pose_array


    def generate_trajectory_types(self):
        types = []
        print('[PartialDataset]  generating trajectory_types...')
        # for ref_timestamp in self.ref_timestamp_array:
        for i in tqdm(range(len(self))):
        #     if i < 33000:
        #         continue
            ref_timestamp = self.ref_timestamp_array[i]

            pose_array = self.generate_pose_array(ref_timestamp)
            if pose_array.shape[1] == 0:
                break
            delta_yaw = pose_array[-1,:] - pose_array[-1,0]
            avg = np.average(delta_yaw)
            if abs(avg) <= self.type_threshold:
                type_id = 0
            elif avg > self.type_threshold:
                type_id = 1
            else:
                type_id = 2
            types.append([ref_timestamp, type_id])
        return np.array(types)



    def image(self, ref_timestamp):  ## TODO
        stereo_centre_timestamp = find_nearest(ref_timestamp, self.stereo_centre_timestamp_array)
        stereo_centre_data = self.stereo_centre_data(stereo_centre_timestamp)
        return self.process.image(stereo_centre_data)

    def pointcloud(self, ref_timestamp):  ## TODO
        velodyne_left_timestamp = find_nearest(ref_timestamp, self.velodyne_left_timestamp_array)
        velodyne_left_data = self.velodyne_left_data(velodyne_left_timestamp)
        velodyne_right_timestamp= find_nearest(ref_timestamp, self.velodyne_right_timestamp_array)
        velodyne_right_data= self.velodyne_right_data(velodyne_right_timestamp)

        velodyne_left = self.process.pointcloud(velodyne_left_data, 'left')
        velodyne_right = self.process.pointcloud(velodyne_right_data, 'right')
        return np.hstack((velodyne_left, velodyne_right))
    
    


    def get_velocity(self, ref_timestamp):
        timestamp = find_nearest(ref_timestamp, self.ins_timestamp_array)
        df = self.ins_df
        df = df[df['timestamp'] == timestamp]
        vx, vy = float(df['velocity_north'].values), float(df['velocity_east'].values)
        return vx, vy

    def get_yaw(self, ref_timestamp):
        timestamp = find_nearest(ref_timestamp, self.ins_timestamp_array)
        df = self.ins_df
        df = df[df['timestamp'] == timestamp]
        return float(df['yaw'].values)



    
    
def cum_vo(delta_pose_array, imu_height):
    num = delta_pose_array.shape[1]
    delta_point_array = np.vstack((delta_pose_array[:3,:], np.ones((1,num))))
    delta_euler_array = delta_pose_array[3:,:]

    pose_array = np.array([0.,0,0,0,0,0]).reshape(6,1)
    for i in range(num):
        p0 = pose_array[:,-1]
        T = cu.basic.HomogeneousMatrix.xyzrpy(p0)
        p = np.dot(T, delta_point_array[:,i])
        e = p0[3:] + delta_euler_array[:,i]
        pose_array = np.hstack((pose_array, np.vstack((p[:3], e)).reshape(6,1)))
    pose_array[2,:] += imu_height
    # pose_array[2,:] -= imu_height
    pose_array[3:,:] = cu.basic.pi2pi(pose_array[3:,:])
    return pose_array
