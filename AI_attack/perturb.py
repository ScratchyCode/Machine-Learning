import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt

images = []

# load images
img = cv2.imread("source.jpg",1)
altezza, larghezza = img.shape[:2]
resized = cv2.resize(img,(331,331))
images.append(resized)

img = cv2.imread("perturbation.jpg",1)
resized = cv2.resize(img,(331,331))
images.append(resized)

avg_image = images[0]
for i in range(len(images)):
    if i == 0:
        pass
    else:
        alpha = 0.2        # perturbation weight
        beta = 1.0 - alpha # target weight
        avg_image = cv2.addWeighted(images[i], alpha, avg_image, beta, 0.0)

cv2.imwrite("perturbed_resized.jpg", avg_image)

# original source size
original_size = cv2.resize(avg_image,(larghezza,altezza))
cv2.imwrite("perturbed.jpg", original_size)
