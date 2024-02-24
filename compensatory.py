# E:\Code Item\Online disconnection location compensation item\venv
# coding: utf-8
import math
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2  # 导入opencv库
from pyciede2000 import ciede2000
from PIL.ImageFile import ImageFile
import random
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, TiffImagePlugin, ImageCms, ImageDraw, ImageFont

from numba import jit, prange

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
# 绕过默认图像编码器，提高读取速度
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
        start_time = time.time()  # 记录函数开始执行的时间戳
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 记录函数执行完成的时间戳
        elapsed_time = end_time - start_time  # 计算时间差，得到函数的运行时间
        print(f"function： {func.__name__} ，runingtime: {elapsed_time} s")
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
    # h = nparray_origin_img.shape[0]  # Image height
    w = nparray_origin_img.shape[1]  # Image width

    greatimage = nparray_origin_img.copy()  # 创建与原始图像相同尺寸的空数组
    # Do something with c_channel
    greatimage[greatimage == 200] = 100

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
        # Create a list of tasks based on x_columns
        ''' tasks = [
            (
                COLOR_COMBINATIONS,
                compensated_img.view(),
                blurred_img_labcolorspace.view(),
                nparray_origin_img.view(),
                i,
                0
            )
            for i in x_columns
        ]'''
        compensated_img_slices = []
        # 获取可用的处理器核心数
        for x in x_columns:
            compensated_img_slices.append(color_correction_process(COLOR_COMBINATIONS,
                                                                   compensated_img.view(),
                                                                   blurred_img_labcolorspace.view(),
                                                                   nparray_origin_img.view(),
                                                                   x,
                                                                   color))

        '''num_cores = multiprocessing.cpu_count()
        # Create a thread pool with a maximum of 8 workers
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            # Submit each task to the thread pool and store the future objects in a list
            futures = [executor.submit(color_correction_process, *task) for task in tasks]
            # Iterate over the completed futures and get their results
            for future in as_completed(futures):'''

        # 将切片应用到greatimage中，按照x_columns的顺序
        for i, x in enumerate(x_columns):
            greatimage[:, x - 1:x + 2, :] = compensated_img_slices[i]
        # Save the compensated image as a TIF file
    good_image = Image.fromarray(greatimage, mode='CMYK')
    good_image.save('greatimg.tif')

    # Print a message to indicate the completion of the process
    print('Process completed')


def set_sign(img, color):
    """
    设置墜艷盘
    :param img:
    :param contrast_lines:
    :return:
    """
    global text
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
    """
    设置墜艷盘
    :param img:
    :param contrast_lines:
    :return:
    """
    contrast_lines = [15600 - j for j in contrast_linesori]  # 15600是后面的图像的宽数
    # 创建绘图对象
    draw = ImageDraw.Draw(img)
    # 绘制三角形
    for x in contrast_lines:
        # 创建三角形的顶点坐标
        polygon = [(x - 10, 400), (x + 10, 400), (x, 450)]

        # 在图像中绘制三角形
        draw.polygon(polygon, fill=(0, 0, 0, 200))
    if color == 0:
        text = "C chanel"
    elif color == 1:
        text = "M chanel"
    elif color == 2:
        text = "Y chanel"
    elif color == 3:
        text = "K chanel"

    # 设置文本样式
    font = ImageFont.truetype("arial.ttf", 100)
    text_color = (0, 0, 0, 255)  # 黑

    # 获取文本尺寸
    text_size = draw.textsize(text, font=font)

    # 计算文本的位置
    text_x = (15600 - text_size[0]) // 2  # 文本在图像水平方向的居中位置
    text_y = 100  # 文本距离图像顶部的距离

    # 在图像中添加文字
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
    """
    设置墜艷盘
    :param img:
    :param contrast_lines:
    :return:
    """
    contrast_lines = [15600 - j for j in contrast_linesori]  # 15600是图像的宽度
    for i in contrast_lines:
        img[:, i, chanel] = 0

    return img


def lab_color_differenceonlylight(matrix1, matrix2):
    # 提取L、a、b通道的值
    L1, a1, b1 = np.mean(matrix1[:, :, 0]), np.mean(matrix1[:, :, 1]), np.mean(matrix1[:, :, 2])
    L2, a2, b2 = np.mean(matrix2[:, :, 0]), np.mean(matrix2[:, :, 1]), np.mean(matrix2[:, :, 2])
    res = abs(L1 - L2)
    return res


def cie94_delta_e(matrix1, matrix2, k_L=1.0, k_C=1.0, k_H=1.0):
    # 提取L、a、b通道的值
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


# CIEDE2000
def lab_color_difference2000(matrix1, matrix2):
    """
    计算两个Lab色彩空间矩阵之间的色差损失函数（平方误差）
    :param matrix1: 第一个Lab矩阵，shape为(3, 3, 3)
    :param matrix2: 第二个Lab矩阵，shape为(3, 3, 3)
    :return: 色差损失函数值
    """
    # 提取L、a、b通道的值
    L1, a1, b1 = np.mean(matrix1[:, :, 0]), np.mean(matrix1[:, :, 1]), np.mean(matrix1[:, :, 2])
    L2, a2, b2 = np.mean(matrix2[:, :, 0]), np.mean(matrix2[:, :, 1]), np.mean(matrix2[:, :, 2])

    # 计算C'和h'值
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

    # 计算ΔL'、ΔC'和ΔH'值
    delta_L_p = L2 - L1
    delta_C_p = C2_p - C1_p
    delta_h_p = h2_p - h1_p
    if C1_p * C2_p == 0:
        delta_h_p = 0
    elif abs(delta_h_p) <= 180:
        delta_h_p = delta_h_p
    elif delta_h_p > 180:
        delta_h_p = delta_h_p - 360
    elif delta_h_p < -180:
        delta_h_p = delta_h_p + 360
    delta_H_p = 2 * math.sqrt(C1_p * C2_p) * math.sin(math.radians(delta_h_p / 2))

    # 计算L'、C'和h'的平均值
    L_ave = (L1 + L2) / 2
    C_ave = (C1_p + C2_p) / 2
    h_ave = (h1_p + h2_p) / 2
    if C1_p * C2_p == 0:
        h_ave = h_ave
    elif abs(h1_p - h2_p) <= 180:
        h_ave = h_ave
    elif abs(h1_p - h2_p) > 180 and h_ave < 180:
        h_ave = h_ave + 180
    elif abs(h1_p - h2_p) > 180 and h_ave >= 180:
        h_ave = h_ave - 180

    # 计算T、R_C、S_L、S_C和S_H值
    T = 1 - 0.17 * math.cos(math.radians(h_ave - 30)) + \
        0.24 * math.cos(math.radians(2 * h_ave)) + \
        0.32 * math.cos(math.radians(3 * h_ave + 6)) - \
        0.20 * math.cos(math.radians(4 * h_ave - 63))
    R_C = 2 * math.sqrt(C_ave ** 7 / (C_ave ** 7 + 25 ** 7))
    S_L = 1 + (0.015 * (L_ave - 50) ** 2) / math.sqrt(20 + (L_ave - 50) ** 2)
    S_C = 1 + 0.045 * C_ave
    S_H = 1 + 0.015 * C_ave * T

    # 计算R_T值
    delta_theta = 30 * math.exp(-((h_ave - 275) / 25) ** 2)
    R_T = -R_C * math.sin(math.radians(2 * delta_theta))

    # 计算CIEDE2000色差
    delta_E = math.sqrt(
        (delta_L_p / (S_L)) ** 2 +
        (delta_C_p / (S_C)) ** 2 +
        (delta_H_p / (S_H)) ** 2 +
        R_T * (delta_C_p / (S_C)) * (delta_H_p / (S_H))
    )

    # 返回色差值
    return delta_E


def lab_color_difference2000ku(matrix1, matrix2):
    """
    计算两个Lab色彩空间矩阵之间的色差损失函数（平方误差）
    :param matrix1: 第一个Lab矩阵，shape为(3, 3, 3)
    :param matrix2: 第二个Lab矩阵，shape为(3, 3, 3)
    :return: 色差损失函数值
    """
    # 提取L、a、b通道的值
    L1, a1, b1 = np.mean(matrix1[:, :, 0]), np.mean(matrix1[:, :, 1]), np.mean(matrix1[:, :, 2])
    L2, a2, b2 = np.mean(matrix2[:, :, 0]), np.mean(matrix2[:, :, 1]), np.mean(matrix2[:, :, 2])
    res = ciede2000((L1, a1, b1), (L2, a2, b2))
    return res['delta_E_00']


def lab_color_difference76(matrix1, matrix2):
    """
    计算两个Lab色彩空间矩阵之间的色差损失函数（平方误差）
    :param matrix1: 第一个Lab矩阵，shape为(3, 3, 3)
    :param matrix2: 第二个Lab矩阵，shape为(3, 3, 3)
    :return: 色差损失函数值
    """
    # 提取L、a、b通道的值
    L1, a1, b1 = np.mean(matrix1[:, :, 0]), np.mean(matrix1[:, :, 1]), np.mean(matrix1[:, :, 2])
    L2, a2, b2 = np.mean(matrix2[:, :, 0]), np.mean(matrix2[:, :, 1]), np.mean(matrix2[:, :, 2])

    # 计算每个通道的差值
    delta_L = L1 - L2
    delta_a = a1 - a2
    delta_b = b1 - b2
    # 计算平方误差
    loss = math.sqrt(delta_L ** 2 + delta_a ** 2 + delta_b ** 2)
    return loss


def color_correction_process(color_combinations, compensated_imageorigin, blurred_imgoriginlab, imgorigin, col, chanel):
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

    # 将该列所有对应颜色通道数据置为0，方便测试
    compensated_image_slice_after_inpainting[:, 1, chanel] = 0
    # 设置对比实验数据
    return compensated_image_slice_after_inpainting

@measure_time
def Adjacent_point_and_point_diffusion_algorithm(chanel, y_coords, compensated_image_slice, origin_img_slice):
    global thresholds
    col = 1
    if chanel == 0:
        thresholds = {
            200: lambda y, col: ([y - 1, y, y + 1], [col - 1, col + 1], 200),
            150: lambda y, col: (y, [col - 1, col + 1], 100),
            75: lambda y, col: (y, [col + random.choice([-1, 1])], 100),
            0: lambda y, col: (y, [col + random.choice([-1, 0, 1])], 100)
        }
    elif chanel == 1:
        thresholds = {
            200: lambda y, col:([y - 1, y, y + 1], [col - 1, col + 1], 100),
            120: lambda y, col: (y, [col - 1, col + 1], 100),
            75: lambda y, col: (y, [col + random.choice([-1, 1])], 100),
            0: lambda y, col: (y, [col + random.choice([-1, 0, 1])], 100)
        }
    elif chanel == 2:
        thresholds = {
            200: lambda y, col: ([y - 1, y, y + 1], [col - 1, col + 1], 200),
            150: lambda y, col: (y, [col - 1, col + 1], 100),
            75: lambda y, col: (y, [col + random.choice([-1, 1])], 100),
            0: lambda y, col: (y, [col + random.choice([-1, 0, 1])], 100)
        }
    elif chanel == 3:
        thresholds = {
            200: lambda y, col: ([y - 1, y, y + 1], [col - 1, col + 1], 100),
            150: lambda y, col: (y, [col - 1, col + 1], 100),
            75: lambda y, col: (y, [col + random.choice([-1, 1])], 100),
            0: lambda y, col: (y, [col + random.choice([-1, 0, 1])], 100)
        }
    else:
        print('代码335行通道不对')


    for y in y_coords:
        if y != (compensated_image_slice.shape[0] - 1) and y != 0:
            marix3 = origin_img_slice[y - 1:y + 2, :, chanel]
            mean = np.mean(marix3)
            for threshold, func in thresholds.items():
                if mean >= threshold:
                    y_new, col_new, new_pix = func(y, col)
                    if isinstance(y_new, list):
                        compensated_image_slice[y_new[0], col_new, chanel] = new_pix
                        compensated_image_slice[y_new[1], col_new, chanel] = new_pix
                        compensated_image_slice[y_new[2], col_new, chanel] = new_pix
                    else:
                        compensated_image_slice[y_new, col_new, chanel] = new_pix
                    break
                else:
                    pass



def optimize_color_processing_jit(img_slice_y, blurred_slices_lab, other_chanel_list):
    img_slices_y_copy = np.repeat(img_slice_y[None, :, :, :], len(COLOR_COMBINATIONS), axis=0)
    img_slices_y_copy[:, 1, 1, other_chanel_list] = COLOR_COMBINATIONS
    losses = []
    for i in range(len(COLOR_COMBINATIONS)):
        losses.append(lab_color_difference2000ku(blurred_slices_lab,
                                           cv2.GaussianBlur(cmyk2labsmall(img_slices_y_copy[i, :, :, :]), (3, 3), 1.7)))

    return np.argmin(losses)


@measure_time
def Color_difference_gradient_descent_algorithm(y_coords, img_slice_father, chanel, blurred_img_slice,
                                                color_combinations,
                                                compensated_image_slice):
    original_list = [0, 1, 2, 3]
    other_chanel_list = list(filter(lambda x: x != chanel, original_list))

    # 对参照高斯模糊图像进行切片
    # 对 y_coords 进行整数类型转换

    # 处理每个y坐标的任务
    def process_y(y):
        blurred_slice = blurred_img_slice[y - 1:y + 2, :]
        if y % 2 == 0:
            # 对修补图像进行切片
            img_slice_y = img_slice_father[y - 1:y + 2, :].copy()
            # 对该点色道置为0
            img_slice_y[1, 1, chanel] = 0
            # 计算最小欧式距离并获取对应的颜色组合索引
            min_loss_index = optimize_color_processing_jit(img_slice_y, blurred_slice, other_chanel_list)
            min_color_combination = color_combinations[min_loss_index]
            # 使用最小损失对应的颜色组合更新compensated_img
            compensated_image_slice[y, 1, other_chanel_list] = min_color_combination

    num_cores = multiprocessing.cpu_count()
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        executor.map(process_y, y_coords)

    return compensated_image_slice


# 套色不准矫正
def Color_alignment(image_matrix):
    height, width, _ = image_matrix.shape
    result = np.zeros_like(image_matrix)

    def left_shifted(channel_data, i):
        leftshift = np.concatenate((channel_data[:, 1:], np.zeros((height, 1))), axis=1)
        for j in range(i - 1):
            leftshift = np.concatenate((leftshift[:, 1:], np.zeros((height, 1))), axis=1)
        return leftshift

    # 向右平移
    def right_shifted(channel_data, i):

        rightshift = np.concatenate((np.zeros((height, 1)), channel_data[:, :-1]), axis=1)
        for j in range(i - 1):
            rightshift = np.concatenate((np.zeros((height, 1)), rightshift[:, :-1]), axis=1)
        return rightshift

    for channel in range(0, 4):  # 通道的索引从1开始，对应于图像的G、B、R通道
        # 获取当前通道数据
        channel_data = image_matrix[:, :, channel]

        # 对平移后的通道进行加减操作
        if channel == 0:  # 对G通道进行+1操作
            shifted_channel = left_shifted(channel_data,2)
        elif channel == 1:  # 对B通道进行+2操作
            shifted_channel = channel_data
        elif channel == 2:  # 对B通道进行+2操作
            shifted_channel = channel_data
        else:  # 对R通道进行-1操作
            shifted_channel = right_shifted(channel_data, 1)

        # 将处理后的通道数据保存到结果图像中
        result[:, :, channel] = shifted_channel
    return result


'''def cmyk2lab(cmyk_matrix):
    cmyk_image=Image.fromarray(cmyk_matrix, mode='CMYK')
    # 加载CMYK到RGB的ICC配置文件
    cmyk_profile = ImageCms.getOpenProfile("Photoshop5DefaultCMYK.icc")
    rgb_profile = ImageCms.getOpenProfile("lab8.icm")
    # 加载CMYK到RGB的ICC配置文件
    cmyk_to_rgb_transform = ImageCms.buildTransform(cmyk_profile, rgb_profile, 'CMYK', 'LAB')
    # 应用色彩空间转换
    converted_image = ImageCms.applyTransform(cmyk_image, cmyk_to_rgb_transform)
    labmatrix=np.array(converted_image)
    return labmatrix'''
# Data Source KGT.icc
# Data Source KGT.icc
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


def cmyk2lab3(cmyk_matrix):
    lab_matrix = np.zeros((cmyk_matrix.shape[0], cmyk_matrix.shape[1], 3))
    for i in range(cmyk_matrix.shape[0]):
        for j in range(cmyk_matrix.shape[1]):
            cmyk = cmyk_matrix[i, j]
            index = np.where((np.all(lookup_table[:, :4] == cmyk, axis=1)))
            lab_matrix[i, j] = lookup_table[index, 4:]
    return lab_matrix


def lab2cmyk(lab_matrix):
    lab_image = Image.fromarray(lab_matrix, mode='LAB')
    # 加载CMYK到RGB的ICC配置文件
    cmyk_profile = ImageCms.getOpenProfile("Photoshop5DefaultCMYK.icc")
    lab_profile = ImageCms.getOpenProfile("lab8.icm")

    # 创建从CMYK到LAB的转换器
    lab_to_cmyk_transform = ImageCms.buildTransform(lab_profile, cmyk_profile, 'LAB', 'CMYK')
    # 应用色彩空间转换
    converted_image = ImageCms.applyTransform(lab_image, lab_to_cmyk_transform)
    cmykmatrix = np.array(converted_image)
    return cmykmatrix


if __name__ == '__main__':
    XML_FILEs = ['C_ture_line_position.xml', 'M_ture_line_position.xml', 'Y_ture_line_position.xml',
                 'K_ture_line_position.xml']
    halftone_cyan_testimg_tif_path = 'CMYKhalftone.tif'
    compensation_process(halftone_cyan_testimg_tif_path, XML_FILEs)