import cv2

name = "1752731281751.png"

cam = []
for i in range(6):
    img_path = "./viz/camera-{}/".format(i) + name
    cam.append(cv2.imread(img_path))

lidar = cv2.imread('./viz/lidar/'+name)

row1 = cv2.hconcat([cam[2], cam[0], cam[1]])
row2 = cv2.hconcat([cam[5], cam[3], cam[4]])
cam = cv2.vconcat([row1, row2])

# lidar = cv2.resize(lidar, (1800,1800), None, interpolation=cv2.INTER_AREA)

# full = cv2.hconcat([cam, lidar])

cv2.imwrite('temp/viz.png', cam)