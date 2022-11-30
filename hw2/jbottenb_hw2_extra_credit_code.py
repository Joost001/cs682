import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from scipy.ndimage import gaussian_filter


def read_image(IMG_NAME):
    return io.imread(IMG_NAME)


def get_image_by_filter(img):
    rows = np.shape(img)[0]
    columns = np.shape(img)[1]

    red_img = np.zeros(shape=(rows, columns, 1), dtype=int)
    green_img = np.zeros(shape=(rows, columns, 1),  dtype=int)
    blue_img = np.zeros(shape=(rows, columns, 1), dtype=int)

    for i in range(rows):
        for j in range(columns):
            red_img[i][j] = img[i][j][0]
            green_img[i][j] = img[i][j][1]
            blue_img[i][j] = img[i][j][2]

    return {"red_image": red_img, "green_image": green_img, "blue_image": blue_img}


def combine_filters(red, green, blue):
    rows = np.shape(red)[0]
    columns = np.shape(red)[1]

    out_img = np.zeros(shape=(rows, columns, 3), dtype=int)
    for i in range(rows):
        for j in range(columns):
            out_img[i][j] = (red[i][j], green[i][j], blue[i][j])
    return out_img


def low_pass_filter(img, s):
    filter_dict = get_image_by_filter(img)
    red_low_pass = gaussian_filter(filter_dict['red_image'], sigma=s)
    green_low_pass = gaussian_filter(filter_dict['green_image'], sigma=s)
    blue_low_pass = gaussian_filter(filter_dict['blue_image'], sigma=s)
    out_img = combine_filters(red_low_pass, green_low_pass, blue_low_pass)
    return out_img


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
IMG001 = 'cat.jpg'
IMG002 = 'dog.jpg'
IMG003 = 'shoes2.jpg'
IMG004 = 'butterfly2.jpg'

cat = read_image(IMG_DIR+IMG001)
dog = read_image(IMG_DIR+IMG002)
shoes = read_image(IMG_DIR+IMG003)
butterfly = read_image(IMG_DIR+IMG004)

low_img1 = low_pass_filter(dog, 5.5)
high_img1 = high_pass_filter(cat, 20)
hybrid_img1 = generate_hybrid_img(high_img1, low_img1)

low_img2 = low_pass_filter(butterfly, 12)
high_img2 = high_pass_filter(shoes, 20)
hybrid_img2 = generate_hybrid_img(high_img2, low_img2)

plt.interactive(False)
plt.imshow(dog)
plt.show(block=True)
plt.imshow(cat)
plt.show(block=True)

plt.imshow(low_img1)
plt.show(block=True)
plt.imshow(high_img1)
plt.show(block=True)
plt.imshow(hybrid_img1)
plt.show(block=True)

# img_a = dog
# img_b = cat
img_a = butterfly
img_b = shoes


low_img_pyramid_1 = low_pass_filter(dog, 5)
low_img_pyramid_2 = low_pass_filter(dog, 10)
low_img_pyramid_3 = low_pass_filter(dog, 15)
low_img_pyramid_4 = low_pass_filter(dog, 10)

high_img_pyramid_1 = high_pass_filter(cat, 5)
high_img_pyramid_2 = high_pass_filter(cat, 10)
high_img_pyramid_3 = high_pass_filter(cat, 15)
high_img_pyramid_4 = high_pass_filter(cat, 20)

plt.interactive(False)
plt.imshow(low_img_pyramid_1)
plt.show(block=True)
plt.imshow(low_img_pyramid_2)
plt.show(block=True)
plt.imshow(low_img_pyramid_3)
plt.show(block=True)
plt.imshow(low_img_pyramid_4)
plt.show(block=True)
plt.imshow(high_img_pyramid_1)
plt.show(block=True)
plt.imshow(high_img_pyramid_2)
plt.show(block=True)
plt.imshow(high_img_pyramid_3)
plt.show(block=True)
plt.imshow(high_img_pyramid_4)
plt.show(block=True)

