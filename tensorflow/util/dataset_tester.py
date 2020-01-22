import h5py
import cv2
import numpy as np
from matplotlib.pylab import cm

f = h5py.File("../Batch_1.h5", "r")

keys = list(f.keys())

print(keys)

for key in keys:
    print(f[key])

print(f["video"][0])
image = f["video"][0]
image = ((image - image.min()) * (1 / (6 - 0) * 255)).astype('uint8')
image = np.uint8(cm.jet(image)*255)
image = cv2.cvtColor(image,cv2.COLOR_RGBA2BGR)
cv2.imshow("Depth", image)

cv2.waitKey(0)