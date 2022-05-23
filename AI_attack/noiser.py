import numpy as np
import cv2

img = cv2.imread("source.jpg")
altezza, larghezza = img.shape[:2]

mean = 0
var = 200
sigma = var ** 0.5

gaussian = np.random.normal(mean, sigma, (altezza,larghezza)) #  np.zeros((224, 224), np.float32)
noisy_image = np.zeros(img.shape, np.float32)

if len(img.shape) == 2:
    noisy_image = img + gaussian
else:
    noisy_image[:, :, 0] = img[:, :, 0] + gaussian
    noisy_image[:, :, 1] = img[:, :, 1] + gaussian
    noisy_image[:, :, 2] = img[:, :, 2] + gaussian

# normalize of printable images
cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
noisy_image = noisy_image.astype(np.uint8)

cv2.normalize(gaussian, gaussian, 0, 255, cv2.NORM_MINMAX, dtype=-1)
gaussian = gaussian.astype(np.uint8)

cv2.imwrite("noise.jpg", gaussian)
cv2.imwrite("noisy.jpg", noisy_image)
