import carla_utils as cu

import os
from os.path import join
import glob
import random
import numpy as np
from PIL import Image
import cv2
import time
import multiprocessing as mp

import torch
from torch.utils.data import Dataset
# import torchvision.transforms as transforms

# from .partial_master import PartialDatasetMaster
# from .partial_augment import PartialDatasetAugment
from .partial import PartialDataset

from .data_master import DataMaster



class TrajectoryDataset(Dataset):
    trajectory_time = 5.0
    max_length = 25.0
    max_speed = 10.0
    num_points = 50


    def __init__(self, config, mode):
        
        self.config = config
        self.mode = mode

        self.basedir = config.basedir if isinstance(config.basedir, list) else [config.basedir]

        data_master_names = []
        for basedir in self.basedir:
            data_master_names.extend([(basedir, name) for name in os.listdir(basedir)])

        assert len(data_master_names) > 3
        if mode == 'train':
            self.data_master_names = data_master_names[:-2]
        elif mode == 'evaluate':
            self.data_master_names = data_master_names[-2:-1]
        elif mode == 'test':
            self.data_master_names = data_master_names[-1:]
        else:
            raise NotImplementedError

        skip_time = cu.basic.Data(start=0.0, end=self.trajectory_time)
        self.data_masters = [DataMaster(join(basedir, name), skip_time) for (basedir, name) in self.data_master_names]
        # self.data_masters = [PartialDataset(join(basedir, name)) for (basedir, name) in self.data_master_names]

        process_list = []
        for partial_dataset in self.data_masters:
            process = mp.Process(target=partial_dataset.generate_augment_data)
            process.start()
            process_list.append(process)
        for process in process_list:
            process.join()

        for partial_dataset in self.data_masters:
            partial_dataset.generate_augment_data()


        self.num_trajectory_points = int(np.around(self.trajectory_time / self.data_masters[0].delta_t))
        assert self.num_trajectory_points % self.num_points == 0
        self.sample_interval = int(self.num_trajectory_points / self.num_points)

        # split_path = join(os.path.dirname(__file__), 'split')
        # cu.system.mkdir(split_path)
        # try:
        #     straight = np.loadtxt(join(split_path, 'train_s.txt'), delimiter=' ', usecols=[], dtype=np.int64)
        #     left     = np.loadtxt(join(split_path, 'train_l.txt'), delimiter=' ', usecols=[], dtype=np.int64)
        #     right    = np.loadtxt(join(split_path, 'train_r.txt'), delimiter=' ', usecols=[], dtype=np.int64)
        #     self.train_straight, self.train_left, self.train_right = list(straight), list(left), list(right)
        #     self.train_key_list = self.train_straight + self.train_left + self.train_right
        # except IOError:
        #     t1 = time.time()
        #     key_list = self.generate_valid_key(list(range(7)))
        #     straight, left, right = self.generate_type_split(key_list)
        #     self.train_straight, self.train_left, self.train_right = straight, left * (int((len(straight)+1)/(len(left)+1))+1), right * (int((len(straight)+1)/(len(right)+1))+1)
        #     self.train_key_list = self.train_straight + self.train_left + self.train_right

        #     np.savetxt(join(split_path, 'train_s.txt'), np.array(self.train_straight), delimiter=' ', fmt='%d')
        #     np.savetxt(join(split_path, 'train_l.txt'), np.array(self.train_left    ), delimiter=' ', fmt='%d')
        #     np.savetxt(join(split_path, 'train_r.txt'), np.array(self.train_right   ), delimiter=' ', fmt='%d')

        #     t2 = time.time()
        #     print('[{}] generate train time: '.format(self.__class__.__name__), t2-t1)



        # import pdb; pdb.set_trace()

        # self.input_type = None
        # self.dataset_master = PartialDatasetMaster(config)
        # self.data_masters = self.dataset_master.data_masters

        # self.train_key_list = self.dataset_master.train_key_list
        # self.eval_key_list = self.dataset_master.eval_key_list
        # self.test_key_list = self.dataset_master.test_key_list

        # # self.num_costmap = config.net.num_costmap
        # # self.num_trajectory = config.net.num_trajectory
        # self.normalize_length = config.net.normalize_length
        # self.normalize_speed = config.net.normalize_speed

        # self.random_index = None
    

    def set_tmp_index(self, index):
        self.tmp_index = index

    def set_mode(self, mode):
        self.mode = mode


    def __len__(self):
        return np.inf



    def get_trajectory(self, dataset: PartialDataset, index: int):
        timestamps = dataset.ref_timestamp_array[index:index+self.num_trajectory_points]

        x, y, yaw = dataset.gt_ro[:,index:index+self.num_trajectory_points]

        x, y = x - x[0], y - y[0]
        R = cu.basic.RotationMatrix2D(yaw[0])
        x, y = np.dot(R.T, np.vstack((x, y)))
        yaw = cu.basic.pi2pi(yaw - yaw[0])

        times = timestamps - timestamps[0]
        times = times * 1e-6

        return cu.basic.Data(times=times, x=x, y=y, yaw=yaw)





    def get_trajectory_ins(self, dataset: PartialDataset, index: int):
        timestamps = dataset.ref_timestamp_array[index:index+self.num_trajectory_points]

        x, y, vx, vy, yaw = dataset.ins_2d[:,index:index+self.num_trajectory_points]

        x, y = x - x[0], y - y[0]
        R = cu.basic.RotationMatrix2D(yaw[0])
        x, y = np.dot(R.T, np.vstack((x, y)))
        vx, vy = np.dot(R.T, np.vstack((vx, vy)))
        yaw = cu.basic.pi2pi(yaw - yaw[0])

        times = timestamps - timestamps[0]
        times = times * 1e-6

        return cu.basic.Data(times=times, x=x, y=y, vx=vx, vy=vy, yaw=yaw)

        import matplotlib.pyplot as plt
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.plot(x, y, '-r')
        # plt.show()

        plt.figure(1)
        plt.subplot(411)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(x, y, '-r')

        plt.subplot(412)
        plt.plot(times, vx, 'og')

        plt.subplot(413)
        plt.plot(times, vy, 'ob')

        plt.subplot(414)
        plt.plot(times, np.rad2deg(yaw), 'oc')

        plt.show()




    def get_trajectory_old(self, dataset: PartialDataset, index):
        timestamps = dataset.ref_timestamp_array[index:index+self.num_trajectory_points]
        poses = dataset.generate_pose_array(timestamps[0], self.num_trajectory_points)

        timestamps = timestamps[::self.sample_interval]
        poses = poses[:,::self.sample_interval]


        x, y = poses[0,:], poses[1,:]


        yaw = dataset.get_yaw(timestamps[0])
        vxs, vys = [], []
        for ref_timestamp in timestamps:
            vx, vy = dataset.get_velocity(ref_timestamp)
            vxs.append(vx)
            vys.append(vy)
        # vx_array, vy_array = np.array(vxs), np.array(vys)
        vx_array, vy_array = np.asarray(vxs), np.asarray(vys)

        R = cu.basic.RotationMatrix2D(-yaw)
        vxy = np.dot(R, np.vstack((vx_array, vy_array)))
        vx_array, vy_array = vxy[0,:], vxy[1,:]
        
        times = timestamps - timestamps[0]
        times = times * 1e-6

        import pdb; pdb.set_trace()

        return times, x, y
        return times, x, y, vx_array, vy_array




    # def __getitem__(self, pesudo_index):
    #     if self.mode == 'train':
    #         key, index = random.choice(self.train_key_list)
    #     elif self.mode == 'evaluate':
    #         key, index = random.choice(self.eval_key_list)
    #         # key, index = self.eval_key_list[self.tmp_index % len(self.eval_key_list)]
    #     else:
    #         key, index = random.choice(self.test_key_list)
            
    #     dataset: PartialDatasetAugment = self.data_masters[key]



    #     # images = self.dataset_master.get_images(dataset, index)
        
    #     times, x, y, vx, vy = self.dataset_master.get_trajectory(dataset, index)
    #     times = times[::3]
    #     x = x[::3]
    #     y = y[::3]
    #     vx = vx[::3]
    #     vy = vy[::3]
    #     x /= self.normalize_length
    #     y /= self.normalize_length
    #     vx /= self.normalize_speed
    #     vy /= self.normalize_speed

    #     xy = torch.FloatTensor([x, -y]).T
    #     vxy = torch.FloatTensor([vx, -vy]).T

    #     return {'t': times, 'x':x,'y':-y, 'vx':vx,'vy':-vy, 'xy':xy, 'vxy':vxy, 'key': key, 'index': index}
    
    
