import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
from projects.panosim import PanoSim

pano = PanoSim(data_root='/home/kinsd/test_bev/data/panosim', version='v1.0')

fig = plt.figure()

for sample in pano.sample:
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_path, _, _ = pano.get_sample_data(lidar_token)
    with open(lidar_path, 'r') as f:
        plt.cla()
        points = np.fromfile(f, dtype=np.float32).reshape(-1, 5)
        x, y = points[:, 0], points[:, 1]
        plt.scatter(x, y, s=5)
        plt.show()