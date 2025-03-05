import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud
radar_path = "./data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin"

points = np.fromfile(radar_path, dtype=np.float32)
points = points.reshape(-1, 5)
print(points.shape())
#points = RadarPointCloud.from_file(radar_path)

# with open(radar_path, 'rb') as f:
#     while True:
#         line = f.readline()
#         print(line, end='')
#         if line.startswith("FIELD"):
#             field_list = line.split()[1:]
#         if line.startswith("SIZE"):
#             size_list = line.split()[1:]
#         if line.startswith("TYPE"):
#             type_list = line.lower().split()[1:]
#         if line.startswith("DATA"):
#             break
    
#     dtype =  np.dtype([(f, t + s) for f, t, s in zip(field_list, type_list, size_list)])
#     points = np.fromfile(f, dtype=dtype)