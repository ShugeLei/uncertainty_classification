import os
import cv2
import numpy as np


path = '/home/hhn/hhn/data/YBUS/images'
mean = np.array([0, 0, 0], dtype=np.float)
std = np.array([0, 0, 0], dtype=np.float)
n = 0
for i in os.listdir(path):
    img = cv2.imread(os.path.join(path, i), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean += np.sum(img, axis=(0, 1))
    n += len(img) * len(img[0])
mean /= n * 255
print(mean)
for i in os.listdir(path):
    img = cv2.imread(os.path.join(path, i), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for j in range(len(img)):
        for k in range(len(img[0])):
            std[0] += (img[j][k][0] / 255 - mean[0])**2
            std[1] += (img[j][k][1] / 255 - mean[1])**2
            std[2] += (img[j][k][2] / 255 - mean[2])**2
std = np.sqrt(std / n)
print(std)
