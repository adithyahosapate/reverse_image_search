import cv2
import time

img = cv2.imread('IMG_20190707_132808.jpg')
# kps: 274, descriptors: (274, 128)
start=time.time()
surf = cv2.xfeatures2d.SURF_create(1000)
(kps, descs) = surf.detectAndCompute(img, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
end=time.time()

print(end-start)