"""
该功能模块用于实现位置矩阵生成，以及图像反生成操作
"""

import PIL.Image as Image
import numpy as np
import warnings

from venv.pso.net_2.net2_predict import convertjpg

warnings.filterwarnings("ignore")

def image_to_matrix(image):
    """
    将图像转化为矩阵
    :param image:
    :return:
    """
    # 显示图片
    # image.show()
    # 图像转化矩阵
    matrix = np.asarray(image).copy()
    # 返回矩阵
    return matrix


def matrix_to_image(matrix):
    """
    将矩阵转化为图像
    :param matrix:
    :return:
    """
    # 还原新图像

    new_image = Image.fromarray(matrix)
    # 返回新图像
    return new_image


def convert_jpg(jpg_file, width=32, height=32):
    """
    将图片格式进行转化
    :param jpg_file:
    :param width:
    :param height:
    :return:
    """
    # 将图片进行读取
    image = Image.open(jpg_file)
    # 重新转化尺寸
    try:
        new_img = image.resize((width, height), Image.LANCZOS)
    except Exception as e:
        print(e)
    # 返回结果
    return new_img


def interrupt_position(matrix):
    """
    对像素矩阵进行扰动生成
    :param matrix:
    :return: 返回最后的正态分布随机数
    """
    # 求出对应矩阵的维度
    shape = (32, 32, 3)
    # 产生随机扰动数
    noise = np.random.normal(0, np.random.rand() * 10, shape)
    # 改变对应维度上的值
    temp = np.round(noise + matrix)
    # 进行取整操作
    noise_matrix = temp.astype(np.uint8)
    # 进行数组范围限制
    np.clip(noise_matrix, 0, 255)
    return noise_matrix


def interrupt_velocity(matrix):
    """
    随机生成粒子各个维度分量的速度
    :param matrix: 根据传入的初始化矩阵，获取其各维度长度
    :return:
    """
    # 产生随机速度，在-3-3之间
    v = np.random.randint(-3, 4, size=(32, 32, 3))
    return v


if __name__ == '__main__':
    img = convertjpg('net_2/dog.jpg')
    # 直接改变的话内存指向一致
    data = image_to_matrix(img)
    data1 = image_to_matrix(img)
    data = interrupt_position(data)
    # print(interrupt_velocity(matrix=data1).shape)
    l = [data]
    # print(len(l))
    # print(data[data1 != data])
    # print(data.shape[0])
    # print(data.shape[1])
    # print(np.linalg.norm(data + data))
    # new_image = matrix_to_image(data)
    # new_image.show()
    # new_image.save('image1.png')
    # print(data)
    # for i in range(5):
    interrupt_position(data)
