# import numpy as np

# translation = np.matrix([-1.2, 0, 1.45])
# theta = 110
# theta = theta / 180.0 * np.pi

# cam2ego = np.eye(4)
# cam2ego_r = np.matrix([[np.cos(theta), np.sin(theta), 0.0],
#                        [-np.sin(theta), np.cos(theta), 0.0],
#                        [0.0, 0.0, 1.0]])
# cam2ego_t = translation

# cam_2cam_r = np.matrix([[0.0, 0.0, 1.0],
#                       [-1.0, 0.0, 0.0],
#                       [0.0, -1.0, 0.0]])
# np.set_printoptions(precision=16)
# print(cam2ego_r @ cam_2cam_r) 
import json
import numpy as np
from math import sqrt

with open('data/nuscenes/v1.0-mini/ego_pose.json', 'r') as f:
    ego = json.load(f)

timestamp = [anno['timestamp'] for anno in ego]
trans = [anno['translation'] for anno in ego]

v = []
for i in range(len(ego)-1):
    delta_t = abs((timestamp[i] - timestamp[i+1]) / 1e6)
    delta_x = sqrt((trans[i][0]-trans[i+1][0])**2 + (trans[i][1]-trans[i+1][1])**2)
    v_i = delta_x / delta_t * 3.6
    v.append(v_i)

v = np.array(v)
a_v = np.sum(v) / len(v)
print(a_v)