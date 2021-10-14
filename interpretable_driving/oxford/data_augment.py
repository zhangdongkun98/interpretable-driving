import carla_utils as cu

import os
from os.path import join
import numpy as np
import time

from . import data


class DataAugment(data.Data):
    def __init__(self, path):
        super().__init__(path)

        self.save_path = join(path, 'augment')



class PoseVelocity(DataAugment):
    def __init__(self, path, timestamps, ro: data.RadarOdometry, ins: data.InertialNavigationSystem, imu_height):
        super().__init__(path)

        self.file_name = 'pose_velocity.txt'

        data_path = join(self.save_path, self.file_name)
        if os.path.isfile(data_path):
            # self.data = np.loadtxt(data_path, delimiter=' ', usecols=[], dtype=np.float64).T
            self.data = np.loadtxt(data_path, delimiter=' ', usecols=[], dtype=np.float32).T
        else:
            t1 = time.time()
            min_timestamp, max_timestamp = min(timestamps), max(timestamps)

            ### ro
            df = ro[(ro['destination_timestamp'] >= min_timestamp) & (ro['destination_timestamp'] <= max_timestamp)]
            dx, dy, dz = df['x'].values, df['y'].values, df['z'].values
            droll, dpitch, dyaw = df['roll'].values, df['pitch'].values, df['yaw'].values

            delta_pose_array = np.vstack((dx, dy, dz, droll, dpitch, dyaw))
            pose_array = cum_odometry(delta_pose_array, imu_height)[:,:-1]
            x, y = pose_array[0], pose_array[1]
            yaw = cu.basic.pi2pi(pose_array[-1])
            self.data = np.vstack([x, y, yaw])

            ### ins
            vx, vy = [], []
            for timestamp in timestamps:
                ins_timestamp = data.find_nearest(timestamp, ins.timestamps)
                df = ins[ins['timestamp'] == ins_timestamp]
                vx.append(float(df['velocity_north'].values))
                vy.append(float(df['velocity_east'].values))
            vx, vy = np.asarray(vx), np.asarray(vy)
            df = ins[ins['timestamp'] == data.find_nearest(timestamps[0], ins.timestamps)]
            R = cu.basic.RotationMatrix2D(float(df['yaw'].values))
            vxy = np.dot(R.T, np.vstack((vx, vy)))

            self.data = np.vstack((self.data, vxy))

            np.savetxt(data_path, self.data.T, delimiter=' ', fmt='%f')
            t2 = time.time()
            print('[{}] {} save time: '.format(self.__class__.__name__, self.dataset_name), t2-t1)
        return





def cum_odometry(delta_pose_array, imu_height):
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
    pose_array[2,:] += imu_height   ### ! TODO remove
    # pose_array[2,:] -= imu_height
    pose_array[3:,:] = cu.basic.pi2pi(pose_array[3:,:])
    return pose_array


