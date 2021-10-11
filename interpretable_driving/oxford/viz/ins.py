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
    
    x, y = partial_dataset.ins_df['latitude'].values, partial_dataset.ins_df['longitude'].values
    plt.plot(x, y, '-r')
    plt.plot(x[0], y[0], 'og')
    plt.plot(x[-1], y[-1], 'ob')

    plt.subplots(1)
    plt.gca().set_aspect('equal', adjustable='box')

    x, y = partial_dataset.ins_df['northing'].values, partial_dataset.ins_df['easting'].values
    plt.plot(x, y, '-r')
    plt.plot(x[0], y[0], 'og')
    plt.plot(x[-1], y[-1], 'ob')


    plt.show()


