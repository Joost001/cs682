import numpy as np
import matplotlib.image as mpimg
import scipy
from math import sqrt
import scipy.ndimage


def read_image(IMG_NAME):
    img = mpimg.imread(IMG_NAME)
    return img


def pad_image(image):
    return np.pad(image, 1, mode='constant')


def make_output_image(rows, columns):
    return np.zeros(shape=(rows, columns, 3),  dtype=int)


def is_even(index):
    return index % 2 == 0


def cell_color(row, column, padded_array=False):
    if padded_array:
        row = row - 1
        column = column - 1
    if is_even(row) and is_even(column):
        return "red"
    elif not is_even(row) and not is_even(column):
        return "blue"
    else:
        return "green"


def count_neighbors(irow, icolumn, image, padded_array=False):
    red_count = 0
    green_count = 0
    blue_count = 0
    red_total = 0
    green_total = 0
    blue_total = 0

    for row in range(-1, 2):
        for column in range(-1, 2):
            if row != 0 or column != 0:
                if cell_color(row+irow, column+icolumn, padded_array) == 'red':
                    red_count += 1
                    red_total = red_total + image[row+irow][column+icolumn]
                elif cell_color(row+irow, column+icolumn, padded_array) == 'green':
                    green_count += 1
                    green_total = green_total + image[row+irow][column+icolumn]
                elif cell_color(row+irow, column+icolumn, padded_array) == 'blue':
                    blue_count += 1
                    blue_total += blue_total + image[row+irow][column+icolumn]
                else:
                    print('ERROR')
    return {'red_count': red_count, 'green_count': green_count, 'blue_count': blue_count,
            'red_total': red_total, 'green_total': green_total, 'blue_total': blue_total}


def red_filter(row, column, image, padded_array=False):
    counts = count_neighbors(row, column, image, padded_array)
    if cell_color(row, column, padded_array) == 'red':
        return image[row][column]
    elif (counts['red_count'] == 2) or (counts['red_count'] == 4):
        return counts['red_total'] / counts['red_count']
    else:
        raise Exception("Sorry, no conditions match")


def green_filter(row, column, image, padded_array=False):
    counts = count_neighbors(row, column, image, padded_array)
    if cell_color(row, column, padded_array) == 'green':
        return image[row][column]
    elif (counts['green_count'] == 2) or (counts['green_count'] == 4):
        return counts['green_total'] / counts['green_count']
    else:
        raise Exception("Sorry, no conditions match")


def blue_filter(row, column, image, padded_array=False):
    counts = count_neighbors(row, column, image, padded_array)
    if cell_color(row, column, padded_array) == 'blue':
        return image[row][column]
    elif (counts['blue_count'] == 2) or (counts['blue_count'] == 4):
        return counts['blue_total'] / counts['blue_count']
    else:
        raise Exception("Sorry, no conditions match")


def squared_diffences(output_img, CORRECT):
    rows = np.shape(output_img)[0]
    columns = np.shape(output_img)[1]

    tmp1 = np.zeros(shape=(rows, columns, 3))
    for i in range(0, rows):
        for j in range(0, columns):
            r = sqrt((output_img[i][j][0] - CORRECT[i][j][0])**2)
            g = sqrt((output_img[i][j][1] - CORRECT[i][j][1])**2)
            b = sqrt((output_img[i][j][2] - CORRECT[i][j][2])**2)
            tmp1[i][j] = (r, g, b)
    return tmp1


def squared_diffences2(output_img, CORRECT):
    rows = np.shape(output_img)[0]
    columns = np.shape(output_img)[1]

    tmp1 = np.zeros(shape=(rows, columns, 1))
    total = 0
    count = 0
    max_error = 0
    for i in range(0, rows):
        for j in range(0, columns):
            pixel = sqrt((np.sum(output_img[i][j]) - np.sum(CORRECT[i][j]))**2)
            tmp1[i][j] = pixel
            count += 1
            total = total + pixel
            if max_error < pixel:
                max_error = pixel
    return {"error_image": tmp1, "max_error": max_error, "average_error": total/count}


def get_soln_image(mosaic_img):
    rows = np.shape(mosaic_img)[0]
    columns = np.shape(mosaic_img)[1]
    output_img = make_output_image(rows, columns)
    padded_input = pad_image(mosaic_img)
    for i in range(1, rows+1):
        for j in range(1, columns+1):
            r = red_filter(i, j, padded_input, padded_array=True)
            g = green_filter(i, j, padded_input, padded_array=True)
            b = blue_filter(i, j, padded_input, padded_array=True)
            output_img[i - 1][j - 1] = (r, g, b)
    return output_img


def get_image_by_filter(mosaic_img):
    rows = np.shape(mosaic_img)[0]
    columns = np.shape(mosaic_img)[1]

    output_img = get_soln_image(mosaic_img)

    red_img = np.zeros(shape=(rows, columns, 1), dtype=int)
    green_img = np.zeros(shape=(rows, columns, 1),  dtype=int)
    blue_img = np.zeros(shape=(rows, columns, 1), dtype=int)

    for i in range(rows):
        for j in range(columns):
            red_img[i][j] = output_img[i][j][0]
            green_img[i][j] = output_img[i][j][1]
            blue_img[i][j] = output_img[i][j][2]

    return {"green_image": green_img, "red_image": red_img, "blue_image": blue_img}


def combine_filters(red, green, blue):
    rows = np.shape(red)[0]
    columns = np.shape(red)[1]

    out_img = np.zeros(shape=(rows, columns, 3), dtype=int)
    for i in range(rows):
        for j in range(columns):
            out_img[i][j] = (red[i][j], green[i][j], blue[i][j])
    return out_img


def freeman(input_image):
    rgb_images_dict = get_image_by_filter(input_image)
    R_G = rgb_images_dict["red_image"] - rgb_images_dict["green_image"]
    B_G = rgb_images_dict["blue_image"] - rgb_images_dict["green_image"]
    median_RG = scipy.ndimage.median_filter(R_G, size=(3, 3, 1))
    median_BG = scipy.ndimage.median_filter(B_G, size=(3, 3, 1))
    new_red = median_RG + rgb_images_dict["green_image"]
    new_blue = median_BG + rgb_images_dict["green_image"]
    return combine_filters(new_red, rgb_images_dict["green_image"], new_blue)


def generate_mosaic_image(input_img):
    rows = np.shape(input_img)[0]
    columns = np.shape(input_img)[1]

    out_img = np.zeros(shape=(rows, columns), dtype=int)
    for i in range(rows):
        for j in range(columns):
            pixels = {'red': input_img[i][j][0], 'green': input_img[i][j][1], 'blue': input_img[i][j][2]}
            out_img[i][j] = pixels[cell_color(i, j)]
    return out_img