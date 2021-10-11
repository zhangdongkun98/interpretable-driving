import carla_utils as cu
import interpretable_driving


from .tools import generate_args



if __name__ == '__main__':
    config = cu.basic.YamlConfig()
    args = generate_args()
    config.update(args)

    partial_dataset = interpretable_driving.oxford.PartialDataset(config.partial_dir)
    partial_dataset.generate_augment_data()


    import matplotlib.pyplot as plt
    plt.gca().set_aspect('equal', adjustable='box')
    
    x, y = partial_dataset.ins_2d[0], partial_dataset.ins_2d[1]
    plt.plot(x, y, '-r')
    plt.plot(x[-1], y[-1], 'ob')


    plt.show()


