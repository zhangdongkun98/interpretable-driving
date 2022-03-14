import carla_utils as cu

import os
from os.path import join
import random
import numpy as np
from PIL import Image
import cv2
import time
from tqdm import tqdm
import multiprocessing as mp

import torch
from torch.utils.data import Dataset


from .data_master import DataMaster


class DatasetTemplate(Dataset):
    DataMaster = DataMaster

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

        self.data_masters = [self.DataMaster(join(basedir, name)) for (basedir, name) in self.data_master_names]

        process_list = []
        for data_master in self.data_masters:
            process = mp.Process(target=data_master.init_augment_data)
            process.start()
            process_list.append(process)
        for process in process_list:
            process.join()

        for data_master in self.data_masters:
            data_master.init_augment_data()

        return


class TrajectoryDataset(DatasetTemplate):
    # trajectory_time = 5.0
    # max_speed = 12.2
    # max_length = max_speed * trajectory_time
    # num_points = 20

    data_balance_cls = None

    skip_time = cu.basic.Data(start=0.0, end=DataMaster.trajectory_time)

    def __init__(self, config, mode):
        super().__init__(config, mode)

        data_key_path = join(config.data_keys_dir, 'data_keys_{}.txt'.format(self.mode))
        if not os.path.isfile(data_key_path):
            print('\n\n' + cu.basic.prefix(self) + 'generating data keys...')
            data_balance_cls =  config.get('data_balance_cls', self.data_balance_cls)
            data_balance = data_balance_cls(config, mode)
            for i, data_master in enumerate(self.data_masters):
                for index in tqdm(range(len(data_master))):
                    trajectory = data_master.get_trajectory(index)
                    data_balance.classify((i, index), trajectory)

            data_balance.remove_redundancy()
            data_balance.save_to_disk(config.data_keys_dir)
            print(cu.basic.prefix(self) + 'generated data keys.\n')

        self.data_keys = np.loadtxt(data_key_path, delimiter=' ', usecols=[], dtype=np.int64)
        return


    def __len__(self):
        return len(self.data_keys)


    def __getitem__(self, index):
        data_master_index, data_index = self.data_keys[index]

        # data_master_index, data_index = 2, 5328
        # data_master_index, data_index = 0, 7463

        data_master = self.data_masters[data_master_index]

        trajectory = data_master.get_trajectory(data_index)
        times = torch.from_numpy(trajectory.times.astype(np.float32)) /data_master.trajectory_time
        x = torch.from_numpy(trajectory.x) /data_master.max_length
        y = torch.from_numpy(trajectory.y) /data_master.max_length
        vx = torch.from_numpy(trajectory.vx) /data_master.max_speed
        vy = torch.from_numpy(trajectory.vy) /data_master.max_speed

        # print(data_master_index, data_index, times.shape)

        xy = torch.FloatTensor([trajectory.x, trajectory.y]).T /data_master.max_length
        # print('xy: shape: ', xy.shape)

        return cu.basic.Data(times=times, x=x, y=y, vx=vx, vy=vy, xy=xy).to_dict()

