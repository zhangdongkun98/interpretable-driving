import carla_utils as cu
import interpretable_driving

import numpy as np

from .tools import generate_args



if __name__ == '__main__':
    config = cu.basic.YamlConfig()
    args = generate_args()
    config.update(args)

    partial_dataset = interpretable_driving.oxford.PartialDataset(config.partial_dir)
    partial_dataset.generate_augment_data()

    ro_df = partial_dataset.ro_df
    x_array, y_array, z_array = ro_df['x'].values, ro_df['y'].values, ro_df['z'].values
    delta_pose_array = np.vstack((x_array, y_array, z_array, ro_df['roll'].values, ro_df['pitch'].values, ro_df['yaw'].values))


    print(delta_pose_array.shape, partial_dataset.delta_pose_array.shape)

    print('doing')
    pose_array = interpretable_driving.oxford.partial.cum_vo(delta_pose_array, partial_dataset.imu_height)


    import matplotlib.pyplot as plt
    plt.gca().set_aspect('equal', adjustable='box')
    
    x, y = pose_array[0], pose_array[1]
    # plt.plot(x - x[0], y - y[0], '-r')
    plt.plot(x, y, '-r')
    plt.plot(x[-1], y[-1], 'ob')


    plt.show()


