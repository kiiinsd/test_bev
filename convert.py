import numpy as np

translation = np.matrix([-1.2, 0, 1.45])
theta = 110
theta = theta / 180.0 * np.pi

cam2ego = np.eye(4)
cam2ego_r = np.matrix([[np.cos(theta), np.sin(theta), 0.0],
                       [-np.sin(theta), np.cos(theta), 0.0],
                       [0.0, 0.0, 1.0]])
cam2ego_t = translation

cam_2cam_r = np.matrix([[0.0, 0.0, 1.0],
                      [-1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0]])
np.set_printoptions(precision=16)
print(cam2ego_r @ cam_2cam_r) 