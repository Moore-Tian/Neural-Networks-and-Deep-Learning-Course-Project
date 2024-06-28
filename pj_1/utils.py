import numpy as np
import random
from PIL import Image
from struct import unpack
from scipy.ndimage import shift


# 读取图片文件
def read_image(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
    return img


# 读取标签文件
def read_label(path):
    with open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.fromfile(f, dtype=np.uint8)
    return lab


def accuracy_score(y_true, y_pred):
    equal = (y_true == y_pred)
    acc = np.sum(equal) / equal.size
    return acc


# 添加噪声
def add_noise(matrix, noise_rate):
    rows, cols = matrix.shape
    noisy_matrix = np.copy(matrix)
    indices = np.random.choice(rows * cols, int(rows * cols * noise_rate), replace=False)
    noisy_matrix.flat[indices] = 0
    indices = np.random.choice(rows * cols, int(rows * cols * noise_rate), replace=False)
    noisy_matrix.flat[indices] = 255
    return noisy_matrix


# 图像平移
def image_translation(images, shape):
    trans_images = np.copy(images).reshape((images.shape[0], ) + shape)
    for i in range(trans_images.shape[0]):
        x_offset = np.random.randint(-2, 3)
        y_offset = np.random.randint(-2, 3)
        new_image = shift(trans_images[i], [y_offset, x_offset], cval=255)
        trans_images[i] = new_image
    return trans_images.reshape(images.shape)


# 图像的裁剪与填充
def random_crop_and_pad(images, shape):
    width, height = shape
    left = random.randint(0, width - 22)
    top = random.randint(0, height - 22)
    right = left + 22
    bottom = top + 22

    cropped_imgs = np.copy(images).reshape((images.shape[0],) + shape)
    for i in range(images.shape[0]):
        cropped_img = cropped_imgs[i, top:bottom, left:right]

        new_img = np.ones((28, 28), dtype=np.uint8) * 255
        pad_left = (28 - 22) // 2
        pad_top = (28 - 22) // 2
        new_img[pad_top:pad_top + cropped_img.shape[0], pad_left:pad_left + cropped_img.shape[1]] = cropped_img
        cropped_imgs[i] = new_img

    return cropped_imgs.reshape(images.shape)


# 生成参数
def sin(x):
    y = np.sin(x)
    return y


def create_data(func, interval, sample_num, noise=0.0, add_outlier=False, outlier_ratio=0.001):
    X = np.random.rand(sample_num, 1) * (interval[1]-interval[0]) + interval[0]
    y = func(X)

    epsilon = np.random.normal(0, noise, (sample_num, 1))
    y = y + epsilon

    if add_outlier:
        outlier_num = int(sample_num * outlier_ratio)
        if outlier_num != 0:
            outlier_idx = np.random.randint(sample_num, size=[outlier_num, 1])
            y[outlier_idx] = y[outlier_idx] * 5
    return X, y


def mean_squared_error(y_true, y_pred):
    error = -1
    error = np.mean(abs(y_true - y_pred))
    return error


# 获取用于im2col的索引
def get_im2col_indices(X_shape, kernel_H, kernel_W, stride, pad):
    # 获取输入和输出的形状
    _, channels, in_H, in_W = X_shape
    out_H = (in_H + 2 * pad - kernel_H) // stride + 1
    out_W = (in_W + 2 * pad - kernel_W) // stride + 1

    level1 = np.tile(np.repeat(np.arange(kernel_H), kernel_W), channels)
    everyLevels = stride * np.repeat(np.arange(out_H), out_W)
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    slide1 = np.tile(np.tile(np.arange(kernel_W), kernel_H), channels)
    everySlides = stride * np.tile(np.arange(out_W), out_H)
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    k = np.repeat(np.arange(channels), kernel_H * kernel_W).reshape(-1, 1)

    return i, j, k


# 用于将NCHW格式的批次图像转换为列并拼接为二维矩阵，可将卷积运算转换为矩阵乘法
def im2col(X, kernel_H, kernel_W, stride, pad):
    # 将输入图像X用0填充
    X_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    # 获取用于im2col的索引
    i, j, k = get_im2col_indices(X.shape, kernel_H, kernel_W, stride, pad)
    # 将填充后的图像X按k,i,j拼接为列
    cols = X_padded[:, k, i, j]
    # 将拼接后的列按最后一维拼接
    cols = np.concatenate(cols, axis=-1)
    return cols


# 用于将列拼接的二维矩阵转换回NCHW格式的原矩阵
def col2im(input_grad_col, X_shape, kernel_H, kernrl_W, stride, pad):
    # 获取输入和输出的形状
    batch_size, channels, H, W = X_shape
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    X_padded = np.zeros((batch_size, channels, H_padded, W_padded))

    # 获取用于im2col的索引
    i, j, k = get_im2col_indices(X_shape, kernel_H, kernrl_W, stride, pad)
    # 将输入梯度input_grad_col按batch_size,k,i,j拼接为列
    dX_col_reshaped = np.array(np.hsplit(input_grad_col, batch_size))
    # 将拼接后的列按k,i,j添加到填充后的图像X_padded中
    np.add.at(X_padded, (slice(None), k, i, j), dX_col_reshaped)

    return X_padded[:, :, pad:-pad, pad:-pad]
