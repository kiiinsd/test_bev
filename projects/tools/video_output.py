import os

import cv2

timestamps = os.listdir('viz/lidar')
timestamps = list(map(int, [ts[:-4] for ts in timestamps]))
timestamps.sort()

result = []
for ts in timestamps:
    cam = []
    for i in range(6):
        img = cv2.imread('viz/camera-{}/{}.png'.format(i, ts))
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        cam.append(img)
    row1 = cv2.hconcat([cam[2], cam[0], cam[1]])
    row2 = cv2.hconcat([cam[5], cam[3], cam[4]])
    cam_img = cv2.vconcat([row1, row2])
    lidar_img = cv2.imread('viz/lidar/{}.png'.format(ts))
    lidar_img = cv2.resize(lidar_img, (900, 900), interpolation=cv2.INTER_LINEAR)
    final_img = cv2.hconcat([cam_img, lidar_img])
    result.append(final_img)

fps = 2
h, w, channel = result[0].shape
video_size = (w, h)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter('visualize.mp4', fourcc, fps, video_size)

for img in result:
    video.write(img)

video.release()