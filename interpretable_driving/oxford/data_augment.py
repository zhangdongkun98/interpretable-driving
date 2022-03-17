import carla_utils as cu

import os
from os.path import join
from tqdm import tqdm
import numpy as np
import time
import cv2
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic

from . import data
from .utils import camera_model


class DataAugment(data.Data):
    def __init__(self, path, timestamps):
        super().__init__(path)

        self.save_path = join(path, 'augment')



class PoseVelocity(DataAugment):
    def __init__(self, path, timestamps, ro: data.RadarOdometry, ins: data.InertialNavigationSystem, imu_height):
        super().__init__(path, timestamps)

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



class StereoCentreAugment(DataAugment):
    def __init__(self, path, timestamps):
        super().__init__(path, timestamps)

        self.data = data.StereoCentre(path)
        self.camera_model = camera_model.CameraModel(
            join(os.path.split(os.path.abspath(__file__))[0], 'models'),
            'stereo/centre',
        )
        self.data_path = join(self.save_path, 'stereo_centre')
        if not cu.system.isdir(self.data_path):
            cu.system.mkdir(self.data_path)

            for timestamp in tqdm(timestamps):
                t = data.find_nearest(timestamp, self.data.timestamps)
                image = self.load_image(t)
                self.save_image(timestamp, image)
        return


    def load_image(self, timestamp):
        image = cv2.imread(join(self.data.data_path, str(timestamp)+'.png'), cv2.IMREAD_GRAYSCALE)

        image = demosaic(image, 'gbrg')
        image = self.camera_model.undistort(image)
        image = np.array(image).astype(np.uint8)
        image[:,:,[0,2]] = image[:,:,[2,0]]   # change BGR to RGB
        return image


    def save_image(self, timestamp, image):
        image[:,:,[0,2]] = image[:,:,[2,0]]
        image_path = join(self.data_path, str(timestamp)+'.png')
        image = Image.fromarray(image)
        # image = image.crop([0,200, 1280,960])
        # image = image.resize((self.param.image.image_width, self.param.image.image_height))
        image.save(image_path)




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

