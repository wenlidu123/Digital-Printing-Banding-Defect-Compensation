# E:\Code Item\Online disconnection location compensation item\venv
# coding: utf-8
import math
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2  # Import OpenCV library
from pyciede2000 import ciede2000
from PIL.ImageFile import ImageFile
import random
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, TiffImagePlugin, ImageCms, ImageDraw, ImageFont

from numba import jit, prange

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
# Bypass the default image encoder to improve reading speed
TiffImagePlugin.READ_LIBTIFF = True
Up_down_clipping_distance = 500

# Define constants for file name and color combinations
sigma = 1.7
COLOR_COMBINATIONS = [
    (200, 200, 200),
    (0, 200, 200),
    (200, 0, 200),
    (200, 200, 0),
    (0, 0, 200),
    (200, 0, 0),
    (0, 200, 0),
    (0, 0, 0)
]


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the timestamp when the function starts
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Record the timestamp when the function completes
        elapsed_time = end_time - start_time  # Calculate the time difference to get the function's runtime
        print(f"function： {func.__name__} ，running time: {elapsed_time} s")
        return result

    return wrapper


def compensation_process(image_path, XML_FILE):
    '''Compensate the damaged nozzles of a printer by applying color correction.

    Parameters:
        image_path: str
            The path of the TIF image file after rip.

    Returns:
        None
    '''
    try:
        image = Image.open(image_path)
    except IOError:
        print('Image file not found or invalid')
        return

    # Convert the image to a numpy array
    nparray_origin_img = np.array(image)
    nparray_origin_img[nparray_origin_img == 255] = 200

    # Check if the image is in CMYK mode
    assert nparray_origin_img.shape[2] == 4, 'Image is not in CMYK mode'

    # Get the height and width of the image
    w = nparray_origin_img.shape[1]  # Image width

    greatimage = nparray_origin_img.copy()  # Create an empty array with the same dimensions as the original image

    for XML_FILE in XML_FILEs:
        if XML_FILE[0] == 'C':
            color = 0
        elif XML_FILE[0] == 'M':
            color = 1
        elif XML_FILE[0] == 'Y':
            color = 2
        elif XML_FILE[0] == 'K':
            color = 3
        else:
            print('XML file invalid')
            return
        # Read the damaged nozzle sequences from the XML file
        special_index_data = []
        try:
            tree = ET.parse(XML_FILE)
            root = tree.getroot()
            special_index_element = root.find('damage_nozzle_sequences')
            if special_index_element is not None:
                for index_element in special_index_element.iter('index'):
                    index_value = index_element.text
                    special_index_data.append(int(index_value))
        except ET.ParseError:
            print('XML file not found or invalid')
            return

        # Get the x coordinates of the columns with damaged nozzles
        x_columns = [w - x for x in special_index_data]
        # Convert the image to LAB color space and apply Gaussian blur as a reference for loss function
        labimage = cmyk2labbig(nparray_origin_img)
        kernel_size = [3, 3]
        blurred_img_labcolorspace = cv2.GaussianBlur(labimage, kernel_size, sigma)
        # Create a compensated image by applying color correction for each channel
        compensated_img = nparray_origin_img.copy()
        compensated_img_slices = []
        # Get the number of available processor cores
        for x in x_columns:
            compensated_img_slices.append(color_correction_process(COLOR_COMBINATIONS,
                                                                   compensated_img.view(),
                                                                   blurred_img_labcolorspace.view(),
                                                                   nparray_origin_img.view(),
                                                                   x,
                                                                   color))

        # Apply the slices to greatimage in the order of x_columns
        for i, x in enumerate(x_columns):
            greatimage[:, x - 1:x + 2, :] = compensated_img_slices[i]
            # Save the compensated image as a TIF file
    good_image = Image.fromarray(greatimage, mode='CMYK')
    good_image.save('greatimg.tif')

    # Print a message to indicate the completion of the process
    print('Process completed')


def set_sign(img, color):
    """
    Set the sign for the color channel.
    :param img: The image to draw on.
    :param color: The color channel index.
    :return: The image with the sign set.
    """
    contrast_linesori = [637,
                         1612,
                         2587,
                         3562,
                         4537,
                         5512,
                         6487,
                         7462,
                         8437,
                         9412,
                         10387,
                         11362,
                         12337,
                         13312,
                         14287,
                         15262,
                         337,
                         1312,
                         2287,
                         3262,
                         4237,
                         5212,
                         6187,
                         7162,
                         8137,
                         9112,
                         10087,
                         11062,
                         12037,
                         13012,
                         13987,
                         14962
                         ]
    contrast_lines = [15600 - j for j in contrast_linesori]  # 15600 is the width of the image
    draw = ImageDraw.Draw(img)
    for x in contrast_lines:
        polygon = [(x - 10, 400), (x + 10, 400), (x, 450)]
        draw.polygon(polygon, fill=(0, 0, 0, 200))
    if color == 0:
        text = "C channel"
    elif color == 1:
        text = "M channel"
    elif color == 2:
        text = "Y channel"
    elif color == 3:
        text = "K channel"

    font = ImageFont.truetype("arial.ttf", 100)
    text_color = (0, 0, 0, 255)  # Black

    text_size = draw.textsize(text, font=font)
    text_x = (15600 - text_size[0]) // 2  # Center text horizontally
    text_y = 100  # Distance from the top of the image

    draw.text((text_x, text_y), text, font=font, fill=text_color)

    return img

def Set_contrast_lines(img, chanel):
    contrast_linesori = [637,
                         1612,
                         2587,
                         3562,
                         4537,
                         5512,
                         6487,
                         7462,
                         8437,
                         9412,
                         10387,
                         11362,
                         12337,
                         13312,
                         14287,
                         15262,
                         ]
    contrast_lines = [15600 - j for j in contrast_linesori]  # 15600 is the width of the image
    for i in contrast_lines:
        img[:, i, chanel] = 0

    return img

def lab_color_differenceonlylight(matrix1, matrix2):
    # Extract L, a, b channel values
    L1, a1, b1 = np.mean(matrix1[:, :, 0]), np.mean(matrix1[:, :, 1]), np.mean(matrix1[:, :, 2])
    L2, a2, b2 = np.mean(matrix2[:, :, 0]), np.mean(matrix2[:, :, 1]), np.mean(matrix2[:, :, 2])
    res = abs(L1 - L2)
    return res


def cie94_delta_e(matrix1, matrix2, k_L=1.0, k_C=1.0, k_H=1.0):
    # Extract L, a, b channel values
    L1, a1, b1 = np.mean(matrix1[:, :, 0]), np.mean(matrix1[:, :, 1]), np.mean(matrix1[:, :, 2])
    L2, a2, b2 = np.mean(matrix2[:, :, 0]), np.mean(matrix2[:, :, 1]), np.mean(matrix2[:, :, 2])

    delta_L = L1 - L2
    C1 = math.sqrt(a1 ** 2 + b1 ** 2)
    C2 = math.sqrt(a2 ** 2 + b2 ** 2)
    delta_C = C1 - C2
    delta_a = a1 - a2
    delta_b = b1 - b2
    delta_H_squared = delta_a ** 2 + delta_b ** 2 - delta_C ** 2

    if delta_H_squared < 0:
        delta_H_squared = 0

    delta_H = math.sqrt(delta_H_squared)

    S_L = 1
    S_C = 1 + 0.045 * C1
    S_H = 1 + 0.015 * C1

    delta_E = math.sqrt((delta_L / (k_L * S_L)) ** 2 +
                        (delta_C / (k_C * S_C)) ** 2 +
                        (delta_H / (k_H * S_H)) ** 2)

    return delta_E


def lab_color_difference2000(matrix1, matrix2):
    """
    Calculate the color difference loss function (squared error) between two Lab color space matrices.
    :param matrix1: The first Lab matrix, shape (3, 3, 3)
    :param matrix2: The second Lab matrix, shape (3, 3, 3)
    :return: The value of the color difference loss function
    """
    # Extract L, a, b channel values
    L1, a1, b1 = np.mean(matrix1[:, :, 0]), np.mean(matrix1[:, :, 1]), np.mean(matrix1[:, :, 2])
    L2, a2, b2 = np.mean(matrix2[:, :, 0]), np.mean(matrix2[:, :, 1]), np.mean(matrix2[:, :, 2])

    # Calculate C' and h' values
    C1 = math.sqrt(a1 ** 2 + b1 ** 2)
    C2 = math.sqrt(a2 ** 2 + b2 ** 2)
    C_ave = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(C_ave ** 7 / (C_ave ** 7 + 25 ** 7)))
    a1_p = (1 + G) * a1
    a2_p = (1 + G) * a2
    C1_p = math.sqrt(a1_p ** 2 + b1 ** 2)
    C2_p = math.sqrt(a2_p ** 2 + b2 ** 2)
    h1_p = math.degrees(math.atan2(b1, a1_p)) % 360
    h2_p = math.degrees(math.atan2(b2, a2_p)) % 360

    # Calculate ΔL', ΔC' and ΔH' values
    delta_L_p = L2 - L1
    delta_C_p = C2_p - C1_p
    delta_h_p = h2_p - h1_p
    if C1_p * C2_p == 0:
        delta_h_p = 0
    elif abs(delta_h_p) <= 180:
        delta_h_p = delta_h_p
    elif delta_h_p > 180:
        delta_h_p -= 360
    elif delta_h_p < -180:
        delta_h_p += 360
    delta_H_p = 2 * math.sqrt(C1_p * C2_p) * math.sin(math.radians(delta_h_p / 2))

    # Calculate the average values for L', C', and h'
    L_ave = (L1 + L2) / 2
    C_ave_p = (C1_p + C2_p) / 2
    if abs(h1_p - h2_p) > 180:
        h_ave = (h1_p + h2_p + 360) / 2
    else:
        h_ave = (h1_p + h2_p) / 2

    # Calculate T, R_C, S_L, S_C, and S_H values
    T = 1 - 0.17 * math.cos(math.radians(h_ave - 30)) + \
        0.24 * math.cos(math.radians(2 * h_ave)) + \
        0.32 * math.cos(math.radians(3 * h_ave + 6)) - \
        0.20 * math.cos(math.radians(4 * h_ave - 63))
    R_C = 2 * math.sqrt(C_ave_p ** 7 / (C_ave_p ** 7 + 25 ** 7))
    S_L = 1 + (0.015 * (L_ave - 50) ** 2) / math.sqrt(20 + (L_ave - 50) ** 2)
    S_C = 1 + 0.045 * C_ave_p
    S_H = 1 + 0.015 * C_ave_p * T

    # Calculate R_T value
    delta_theta = 30 * math.exp(-((h_ave - 275) / 25) ** 2)
    R_T = -R_C * math.sin(math.radians(2 * delta_theta))

    # Calculate CIEDE2000 color difference
    delta_E = math.sqrt(
        (delta_L_p / S_L) ** 2 +
        (delta_C_p / S_C) ** 2 +
        (delta_H_p / S_H) ** 2 +
        R_T * (delta_C_p / S_C) * (delta_H_p / S_H)
    )

    # Return the color difference value
    return delta_E


def lab_color_difference2000ku(matrix1, matrix2):
    """
    Calculate the color difference loss function (squared error) between two Lab color space matrices.
    :param matrix1: The first Lab matrix, shape (3, 3, 3)
    :param matrix2: The second Lab matrix, shape (3, 3, 3)
    :return: The value of the color difference loss function
    """
    # Extract L, a, b channel values
    L1, a1, b1 = np.mean(matrix1[:, :, 0]), np.mean(matrix1[:, :, 1]), np.mean(matrix1[:, :, 2])
    L2, a2, b2 = np.mean(matrix2[:, :, 0]), np.mean(matrix2[:, :, 1]), np.mean(matrix2[:, :, 2])
    res = ciede2000((L1, a1, b1), (L2, a2, b2))
    return res['delta_E_00']


def lab_color_difference76(matrix1, matrix2):
    """
    Calculate the color difference loss function (squared error) between two Lab color space matrices.
    :param matrix1: The first Lab matrix, shape (3, 3, 3)
    :param matrix2: The second Lab matrix, shape (3, 3, 3)
    :return: The value of the color difference loss function
    """
    # Extract L, a, b channel values
    L1, a1, b1 = np.mean(matrix1[:, :, 0]), np.mean(matrix1[:, :, 1]), np.mean(matrix1[:, :, 2])
    L2, a2, b2 = np.mean(matrix2[:, :, 0]), np.mean(matrix2[:, :, 1]), np.mean(matrix2[:, :, 2])

    # Calculate the difference for each channel
    delta_L = L1 - L2
    delta_a = a1 - a2
    delta_b = b1 - b2
    # Calculate squared error
    loss = math.sqrt(delta_L ** 2 + delta_a ** 2 + delta_b ** 2)
    return loss


def color_correction_process(color_combinations, compensated_imageorigin, blurred_imgoriginlab, imgorigin, col, chanel):
    """
    Process for correcting color in a specific column and channel of the image.
    :param color_combinations: List of color combinations to test.
    :param compensated_imageorigin: The original compensated image.
    :param blurred_imgoriginlab: The LAB color space representation of the blurred original image.
    :param imgorigin: The original image.
    :param col: The column to correct.
    :param chanel: The channel to correct.
    :return: The compensated image slice after inpainting.
    """
    originimg_slice = imgorigin[:, col - 1:col + 2]
    compensated_image_slice = compensated_imageorigin[:, col - 1:col + 2]
    blurred_img_lab_slice = blurred_imgoriginlab[:, col - 1:col + 2]
    y_coords = np.where(imgorigin[:, col, chanel] == 200)[0]
    compensated_image_slice_after_inpainting = Color_difference_gradient_descent_algorithm(y_coords, originimg_slice,
                                                                                           chanel,
                                                                                           blurred_img_lab_slice,
                                                                                           color_combinations,
                                                                                           compensated_image_slice)

    compensated_image_slice_after_inpainting[compensated_image_slice_after_inpainting == 200] = 100
    Adjacent_point_and_point_diffusion_algorithm(chanel, y_coords, compensated_image_slice_after_inpainting,
                                                 originimg_slice)

    # Set all corresponding color channel data in the column to 0 for testing purposes
    compensated_image_slice_after_inpainting[:, 1, chanel] = 0
    # Set up contrast experiment data
    return compensated_image_slice_after_inpainting
@measure_time
def Adjacent_point_and_point_diffusion_algorithm(channel, y_coords, compensated_image_slice, origin_img_slice):
    """
    Apply diffusion algorithm to adjust pixel values based on adjacent points for a specific channel.
    :param channel: The channel index to be processed.
    :param y_coords: The y coordinates of pixels to be processed.
    :param compensated_image_slice: The slice of the compensated image being processed.
    :param origin_img_slice: The slice of the original image.
    """
    global thresholds
    col = 1
    if channel == 0:
        thresholds = {
            200: lambda y, col: ([y - 1, y, y + 1], [col - 1, col + 1], 200),
            150: lambda y, col: (y, [col - 1, col + 1], 100),
            75: lambda y, col: (y, [col + random.choice([-1, 1])], 100),
            0: lambda y, col: (y, [col + random.choice([-1, 0, 1])], 100)
        }
    elif channel == 1:
        thresholds = {
            200: lambda y, col: ([y - 1, y, y + 1], [col - 1, col + 1], 100),
            120: lambda y, col: (y, [col - 1, col + 1], 100),
            75: lambda y, col: (y, [col + random.choice([-1, 1])], 100),
            0: lambda y, col: (y, [col + random.choice([-1, 0, 1])], 100)
        }
    elif channel == 2:
        thresholds = {
            200: lambda y, col: ([y - 1, y, y + 1], [col - 1, col + 1], 200),
            150: lambda y, col: (y, [col - 1, col + 1], 100),
            75: lambda y, col: (y, [col + random.choice([-1, 1])], 100),
            0: lambda y, col: (y, [col + random.choice([-1, 0, 1])], 100)
        }
    elif channel == 3:
        thresholds = {
            200: lambda y, col: ([y - 1, y, y + 1], [col - 1, col + 1], 100),
            150: lambda y, col: (y, [col - 1, col + 1], 100),
            75: lambda y, col: (y, [col + random.choice([-1, 1])], 100),
            0: lambda y, col: (y, [col + random.choice([-1, 0, 1])], 100)
        }
    else:
        print('Channel not valid at line 335')

    for y in y_coords:
        if y != (compensated_image_slice.shape[0] - 1) and y != 0:
            matrix3 = origin_img_slice[y - 1:y + 2, :, channel]
            mean = np.mean(matrix3)
            for threshold, func in thresholds.items():
                if mean >= threshold:
                    y_new, col_new, new_pix = func(y, col)
                    if isinstance(y_new, list):
                        compensated_image_slice[y_new[0], col_new, channel] = new_pix
                        compensated_image_slice[y_new[1], col_new, channel] = new_pix
                        compensated_image_slice[y_new[2], col_new, channel] = new_pix
                    else:
                        compensated_image_slice[y_new, col_new, channel] = new_pix
                    break
                else:
                    pass



def optimize_color_processing_jit(img_slice_y, blurred_slices_lab, other_channel_list):
    """
    Optimize color processing using JIT compilation.
    :param img_slice_y: The image slice along the y-axis.
    :param blurred_slices_lab: The LAB color space representation of the blurred slices.
    :param other_channel_list: List of channels other than the one being processed.
    :return: The index of the minimum loss.
    """
    img_slices_y_copy = np.repeat(img_slice_y[None, :, :, :], len(COLOR_COMBINATIONS), axis=0)
    img_slices_y_copy[:, 1, 1, other_channel_list] = COLOR_COMBINATIONS
    losses = []
    for i in range(len(COLOR_COMBINATIONS)):
        losses.append(lab_color_difference2000ku(blurred_slices_lab,
                                           cv2.GaussianBlur(cmyk2labsmall(img_slices_y_copy[i, :, :, :]), (3, 3), 1.7)))

    return np.argmin(losses)


@measure_time
def Color_difference_gradient_descent_algorithm(y_coords, img_slice_father, channel, blurred_img_slice,
                                                color_combinations,
                                                compensated_image_slice):
    """
    Apply a gradient descent algorithm to minimize color difference in a specific channel.
    :param y_coords: The y coordinates of pixels to be processed.
    :param img_slice_father: The original image slice.
    :param channel: The channel index to be processed.
    :param blurred_img_slice: The LAB color space representation of the blurred image slice.
    :param color_combinations: List of color combinations to test.
    :param compensated_image_slice: The slice of the compensated image being processed.
    :return: The compensated image slice after processing.
    """
    original_list = [0, 1, 2, 3]
    other_channel_list = list(filter(lambda x: x != channel, original_list))

    def process_y(y):
        blurred_slice = blurred_img_slice[y - 1:y + 2, :]
        if y % 2 == 0:
            img_slice_y = img_slice_father[y - 1:y + 2, :].copy()
            img_slice_y[1, 1, channel] = 0
            min_loss_index = optimize_color_processing_jit(img_slice_y, blurred_slice, other_channel_list)
            min_color_combination = color_combinations[min_loss_index]
            compensated_image_slice[y, 1, other_channel_list] = min_color_combination

    num_cores = multiprocessing.cpu_count()
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        executor.map(process_y, y_coords)

    return compensated_image_slice

# Misregistration correction
def Color_alignment(image_matrix):
    """
    Correct color misregistration in an image.
    :param image_matrix: The image matrix to be corrected.
    :return: The corrected image matrix.
    """
    height, width, _ = image_matrix.shape
    result = np.zeros_like(image_matrix)

    def left_shifted(channel_data, i):
        leftshift = np.concatenate((channel_data[:, 1:], np.zeros((height, 1))), axis=1)
        for j in range(i - 1):
            leftshift = np.concatenate((leftshift[:, 1:], np.zeros((height, 1))), axis=1)
        return leftshift

    def right_shifted(channel_data, i):
        rightshift = np.concatenate((np.zeros((height, 1)), channel_data[:, :-1]), axis=1)
        for j in range(i - 1):
            rightshift = np.concatenate((np.zeros((height, 1)), rightshift[:, :-1]), axis=1)
        return rightshift

    for channel in range(0, 4):  # Channel indices start from 0 for CMYK channels
        channel_data = image_matrix[:, :, channel]
        if channel == 0:  # Cyan channel
            shifted_channel = left_shifted(channel_data, 2)
        elif channel == 1:  # Magenta channel
            shifted_channel = channel_data
        elif channel == 2:  # Yellow channel
            shifted_channel = channel_data
        else:  # Black channel
            shifted_channel = right_shifted(channel_data, 1)

        result[:, :, channel] = shifted_channel
    return result


# Data Source KGT.icc
# Lookup table for CMYK to Lab conversion
lookup_table = np.array([
    [0, 0, 0, 0, 100, 0, 0],
    [200, 0, 0, 0, 64, -40, -45],
    [0, 200, 0, 0, 62, 70, -14],
    [0, 0, 200, 0, 94, -12, 78],
    [0, 0, 0, 200, 29, 1, 8],
    [200, 200, 0, 0, 34, 26, -55],
    [200, 0, 200, 0, 64, -54, 25],
    [200, 0, 0, 200, 19, -25, -15],
    [0, 200, 200, 0, 61, 58, 46],
    [0, 200, 0, 200, 19, 34, -2],
    [0, 0, 200, 200, 32, -4, 43],
    [200, 200, 200, 0, 34, -3, -5],
    [200, 200, 0, 200, 6, 22, -34],
    [200, 0, 200, 200, 22, -42, 18],
    [0, 200, 200, 200, 23, 31, 35],
    [200, 200, 200, 200, 9, 2, -1]
])

# Function for CMYK to Lab conversion (small matrices)
@jit(nopython=True)
def cmyk2labsmall(cmyk_matrix):
    lab_matrix = np.zeros((cmyk_matrix.shape[0], cmyk_matrix.shape[1], 3))
    for i in range(cmyk_matrix.shape[0]):
        for j in range(cmyk_matrix.shape[1]):
            cmyk = cmyk_matrix[i, j]
            found_index = -1
            for k in range(lookup_table.shape[0]):
                if np.all(lookup_table[k, :4] == cmyk):
                    found_index = k
                    break
            if found_index != -1:
                lab_matrix[i, j] = lookup_table[found_index, 4:]
    return lab_matrix

# Function for CMYK to Lab conversion (large matrices)
@jit(nopython=True, parallel=True)
def cmyk2labbig(cmyk_matrix):
    lab_matrix = np.zeros((cmyk_matrix.shape[0], cmyk_matrix.shape[1], 3))
    for i in prange(cmyk_matrix.shape[0]):
        for j in prange(cmyk_matrix.shape[1]):
            cmyk = cmyk_matrix[i, j]
            found_index = -1
            for k in range(lookup_table.shape[0]):
                if np.all(lookup_table[k, :4] == cmyk):
                    found_index = k
                    break
            if found_index != -1:
                lab_matrix[i, j] = lookup_table[found_index, 4:]
    return lab_matrix

# Alternative function for CMYK to Lab conversion
def cmyk2lab3(cmyk_matrix):
    lab_matrix = np.zeros((cmyk_matrix.shape[0], cmyk_matrix.shape[1], 3))
    for i in range(cmyk_matrix.shape[0]):
        for j in range(cmyk_matrix.shape[1]):
            cmyk = cmyk_matrix[i, j]
            index = np.where((np.all(lookup_table[:, :4] == cmyk, axis=1)))
            lab_matrix[i, j] = lookup_table[index, 4:]
    return lab_matrix

# Function for Lab to CMYK conversion
def lab2cmyk(lab_matrix):
    lab_image = Image.fromarray(lab_matrix, mode='LAB')
    # Load ICC profiles for CMYK to Lab conversion
    cmyk_profile = ImageCms.getOpenProfile("Photoshop5DefaultCMYK.icc")
    lab_profile = ImageCms.getOpenProfile("lab8.icm")

    # Create a transform from Lab to CMYK
    lab_to_cmyk_transform = ImageCms.buildTransform(lab_profile, cmyk_profile, 'LAB', 'CMYK')
    # Apply the color