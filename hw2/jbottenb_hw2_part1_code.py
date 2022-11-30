import matplotlib.pyplot as plt
from skimage import io
from scipy.ndimage import gaussian_filter
import numpy as np


def read_image(IMG_NAME):
    return io.imread(IMG_NAME, as_gray=True)


def low_pass_filter(img, s):
    return gaussian_filter(img, sigma=s)


def high_pass_filter(img, s):
    low_img = low_pass_filter(img, s)
    return img - low_img


def average_image(img1, img2):
    return (np.array(img1) + np.array(img2)) / 2.0


def generate_hybrid_img(img1, img2, method='add'):
    if method == 'add':
        return img1 + img2
    elif method == 'average':
        return average_image(img1, img2)


IMG_DIR = 'images/'
IMG001 = 'c1.jpg'
IMG002 = 'c2.jpg'
IMG003 = 'butterfly2.jpg'
IMG004 = 'shoes2.jpg'
IMG005 = 'car.jpg'
IMG006 = 'rhino.jpg'

img1 = read_image(IMG_DIR+IMG001)
img2 = read_image(IMG_DIR+IMG002)
img3 = read_image(IMG_DIR+IMG003)
img4 = read_image(IMG_DIR+IMG004)
img5 = read_image(IMG_DIR+IMG005)
img6 = read_image(IMG_DIR+IMG006)

low_img1 = low_pass_filter(img2, 3)
high_img1 = high_pass_filter(img1, 10)

low_img2 = low_pass_filter(img3, 8)
high_img2 = high_pass_filter(img4, 9)

low_img3 = low_pass_filter(img5, 5)
high_img3 = high_pass_filter(img6, 12)

hybrid_img1 = generate_hybrid_img(high_img1, low_img1)
hybrid_img2 = generate_hybrid_img(high_img2, low_img2)
hybrid_img3 = generate_hybrid_img(high_img3, low_img3)


plt.interactive(False)
plt.imshow(img1, cmap='gray')
plt.show(block=True)
plt.imshow(img2, cmap='gray')
plt.show(block=True)
plt.imshow(low_img1, cmap='gray')
plt.show(block=True)
plt.imshow(high_img1, cmap='gray')
plt.show(block=True)
plt.imshow(hybrid_img1, cmap='gray')
plt.show(block=True)

plt.imshow(img3, cmap='gray')
plt.show(block=True)
plt.imshow(img4, cmap='gray')
plt.show(block=True)
plt.interactive(False)
plt.imshow(low_img2, cmap='gray')
plt.show(block=True)
plt.imshow(high_img2, cmap='gray')
plt.show(block=True)
plt.imshow(hybrid_img2, cmap='gray')
plt.show(block=True)

plt.imshow(img5, cmap='gray')
plt.show(block=True)
plt.imshow(img6, cmap='gray')
plt.show(block=True)
plt.interactive(False)
plt.imshow(low_img3, cmap='gray')
plt.show(block=True)
plt.imshow(high_img3, cmap='gray')
plt.show(block=True)
plt.imshow(hybrid_img3, cmap='gray')
plt.show(block=True)