import carla_utils as cu
import interpretable_driving


from .tools import generate_args



if __name__ == '__main__':
    config = cu.basic.YamlConfig()
    args = generate_args()
    config.update(args)

    partial_dataset = interpretable_driving.oxford.PartialDataset(config.partial_dir)


    import matplotlib.pyplot as plt
    plt.subplots(1)
    plt.gca().set_aspect('equal', adjustable='box')
    
    x, y = partial_dataset.gps_df['latitude'].values, partial_dataset.gps_df['longitude'].values
    plt.plot(x - x[0], y - y[0], '-r')

    plt.subplots(1)
    plt.gca().set_aspect('equal', adjustable='box')

    x, y = partial_dataset.gps_df['northing'].values, partial_dataset.gps_df['easting'].values
    plt.plot(x - x[0], y - y[0], '-r')


    plt.show()


