import random

import cv2
import numpy as np
from torchvision import transforms as T
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl


plt.switch_backend('agg')


# img_name = '2048711_092731296.jpg'
# img = cv2.imread('examples/{}'.format(img_name), 1)
# h, w = img.shape[0], img.shape[1]
# slic = cv2.ximgproc.createSuperpixelSLIC(img, region_size=20, ruler=20)
# slic.iterate(10)
# bounds = slic.getLabelContourMask()
# sp_img1 = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(bounds))
# sp_img2 = np.stack([np.zeros_like(bounds), np.zeros_like(bounds), bounds], axis=-1)
# sp_img = sp_img1 + sp_img2
# cv2.imwrite(img_name.replace('.jpg', 'imgsuperpixel.png'), sp_img.astype(np.uint8))
# sprisa = cv2.imread('examples/2048711_092731296_sprisaimg.png', 1)
# sprisa = cv2.resize(sprisa, (h, w))
# sp_img1 = cv2.bitwise_and(sprisa, sprisa, mask=cv2.bitwise_not(bounds))
# sp_img2 = np.stack([np.zeros_like(bounds), np.zeros_like(bounds), bounds], axis=-1)
# sp_img = sp_img1 + sp_img2
# cv2.imwrite(img_name.replace('.jpg', 'sprisasuperpixel.png'), sp_img.astype(np.uint8))
#
# grid_img = cv2.imread('examples/2048711_092731296_riseimg.png', 1)
# grid_img = cv2.resize(grid_img, (h, w))
# for i in range(round(h/16)-1, h-1, round(h/16)):
#     grid_img[i, :, 0], grid_img[i, :, 1], grid_img[i, :, 2] = 0, 0, 255
# for i in range(round(w/16)-1, w-1, round(w/16)):
#     grid_img[:, i, 0], grid_img[:, i, 1], grid_img[:, i, 2] = 0, 0, 255
# cv2.imwrite(img_name.replace('.jpg', 'risegrid.png'), grid_img.astype(np.uint8))

# masks = slic.getLabels()
# palette = list(set(list(masks.flatten())))
# n = len(palette)
# ss = np.random.uniform(0, 1, size=(n-int(n/2), 10, n))
# for i in range(n-int(n/2)):
#     for j in range(10):
#         hit = np.asarray(ss[i, j] < (i+int(n/2)-1)/n).nonzero()[0].tolist()
#         if len(hit) < 5:
#             break
#         m = np.zeros_like(masks, dtype=np.uint8)
#         for k in hit:
#             m += np.where(masks == k, 255, 0).astype(np.uint8)
#         m = cv2.resize(cv2.resize(m, (32, 32)), (256, 256))
#         res = np.stack([m, m, m], axis=-1) / 255 * cv2.resize(img, (256, 256))
#         cv2.imwrite('result/figures/{}_{}.png'.format(i, j), res.astype(np.uint8))
#         cv2.imwrite('result/figures/{}_{}_mask.png'.format(i, j), m)

# img_name = '2261894_104103347.jpg'
# img = cv2.imread('examples/{}'.format(img_name), 0)
# img = T.Compose([T.ToTensor(), T.Resize((256, 256))])(img)
# transform1 = T.Compose([T.RandomRotation((-30, 30)), T.Resize((224, 224))])
# transform2 = T.RandomCrop((224, 224))
# transform3 = T.Compose([T.RandomHorizontalFlip(1), T.Resize((224, 224))])
# transform4 = T.Compose([T.RandomRotation((-30, 30)), T.RandomCrop((224, 224))])
# transform5 = T.Compose([T.RandomRotation((-30, 30)), T.RandomHorizontalFlip(1), T.Resize((224, 224))])
# transform6 = T.Compose([T.RandomHorizontalFlip(1), T.RandomCrop((224, 224))])
# transform7 = T.Compose([T.RandomRotation((-30, 30)), T.RandomHorizontalFlip(1), T.RandomCrop((224, 224))])
# transforms = [transform1, transform2, transform3, transform4, transform5, transform6, transform7]
#
# for i, transformer in enumerate(transforms):
#     augmented_image = transformer(img) * 255
#     cv2.imwrite('{}.png'.format(i), augmented_image.numpy().astype(np.uint8).squeeze())

# abscissa_x = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
# y_acc = [0.994197309, 0.996031761, 0.99595958, 0.995934963, 0.995884776, 0.997899175, 1, 1, 1, 1]
# y_recall = [0.993902445, 0.996825397, 0.996731997, 0.996699691, 0.996632993, 1, 1, 1, 1, 1]
# y_screen = [_/556 for _ in [517, 504, 495, 492, 486, 476, 470, 414, 358, 356]]
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
# ax.plot(abscissa_x, y_acc)
# ax.plot(abscissa_x, y_recall)
# ax.plot([], [])
# ax.legend(['Accuracy', 'Recall', 'Screening'], loc='lower right')
# ax2 = ax.twinx()
# ax2.set_ylim(bottom=0.5, top=1)
# ax2.plot(abscissa_x, y_screen, 'green')
# plt.savefig('result/threshold_vitRS.png')

x = np.arange(3)
y1 = [38.67, 36.51, 37.05]
y2 = [17.09, 17.81, 15.47]
bar_width = 0.3
plt.rcParams.update({'font.size': 12})
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.set_ylim(top=50)
ax.bar(x, y1, bar_width, label='baseline')
ax.bar(x+bar_width, y2, bar_width, label='Reliable Soup************')
ax.set_xticks(x+bar_width/2)
ax.set_xticklabels(['EfficientNet-B0', 'ResNet-50', 'ViT-B'])
ax.legend(fontsize='12')
plt.savefig('result/bar.png')
