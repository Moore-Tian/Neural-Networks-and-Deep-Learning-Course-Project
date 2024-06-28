import os
from model import back_propagation, Conv_BP
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
from PIL import Image
import sys


def accuracy_score(y_true, y_pred):
    equal = (y_true == y_pred)
    acc = np.sum(equal) / equal.size
    return acc


def add_noise(matrix):
    rows, cols = matrix.shape
    noisy_matrix = np.copy(matrix)
    indices = np.random.choice(rows * cols, 5, replace=False)
    noisy_matrix.flat[indices] = 0
    indices = np.random.choice(rows * cols, 5, replace=False)
    noisy_matrix.flat[indices] = 255
    return noisy_matrix


def image_translation(image):
    x_offset = np.random.randint(-2, 3)
    y_offset = np.random.randint(-2, 3)
    new_image = image.transform(image.size, Image.AFFINE, (1, 0, x_offset, 0, 1, y_offset))
    return new_image


def random_crop_and_pad(image):
    width, height = image.size
    left = random.randint(0, width - 22)
    top = random.randint(0, height - 22)
    right = left + 22
    bottom = top + 22
    cropped_img = image.crop((left, top, right, bottom))

    new_img = Image.new("L", (28, 28), color=255)
    pad_left = (28 - 22) // 2
    pad_top = (28 - 22) // 2
    new_img.paste(cropped_img, (pad_left, pad_top))

    return new_img

# 文件夹路径
folder_path = 'train_data/train'

# 初始化一个空列表，用于存储图像的一维数组
image_train = []
image_valid = []
num_label = 12

# 遍历每个子文件夹
for i in range(1, num_label+1):
    subfolder_path = os.path.join(folder_path, str(i))
    # 遍历每个子文件夹中的BMP图像
    for j in range(1, 521):
        image_path = os.path.join(subfolder_path, f'{j}.bmp')
        image = Image.open(image_path)
        gray_image = image.convert('L')
        array = np.where(np.array(gray_image) > 1, 0, 1)
        image_train.append(np.expand_dims(array, axis=0))

        noised_image = np.where(add_noise(np.array(gray_image)) > 1, 0, 1)
        image_train.append(np.expand_dims(noised_image, axis=0))

        translated_image = image_translation(gray_image)
        image_train.append(np.expand_dims(np.where(np.array(translated_image) > 1, 0, 1), axis=0))

        croped_image = random_crop_and_pad(gray_image)
        image_train.append(np.expand_dims(np.where(np.array(croped_image) > 1, 0, 1), axis=0))

    for j in range(521, 621):
        image_path = os.path.join(subfolder_path, f'{j}.bmp')
        image = Image.open(image_path)
        gray_image = image.convert('L')
        array = np.where(np.array(gray_image) > 1, 0, 1)
        image_valid.append(np.expand_dims(array, axis=0))

# 将图像数组组合成一个矩阵
images_train = np.stack(image_train).reshape(len(image_train), 1, 28, 28)
images_valid = np.stack(image_valid).reshape(len(image_valid), 1, 28, 28)

label_train = np.arange(num_label*520*4)//(520*4)
labels_train = np.eye(num_label)[label_train]

label_valid = np.arange(num_label*100)//100
labels_valid = label_valid

# 建立BP网络
batch_size = 16
dropout = 0.1
layers = [3*28*28, 256, 128, num_label]

conv_bp = Conv_BP(in_channels=1, out_channels=3, kernel_size=(3, 3), layers=layers, dropout=dropout, classifacation=True)

conv_bp.train(images_train, labels_train, X_valid=images_valid, y_valid=labels_valid, batch_size=batch_size, epochs=1)

""" BP_net = back_propagation(layers=layers, dropout=dropout, classifacation=True) """

# 开始训练
train_accuracy_scores, valid_accuracy_scores, train_loss = conv_bp.train(images_train, labels_train, X_valid=images_valid, y_valid=labels_valid)
labels_pred = np.argmax(np.array(conv_bp.predict(images_valid)), axis=1)
print("accuracy score = ", accuracy_score(labels_valid, labels_pred))
x = np.arange(len(train_accuracy_scores))

# 训练效果展示
plt.figure(1)
plt.plot(x, train_accuracy_scores, color='red', label='train accuracy')
plt.plot(x, valid_accuracy_scores, color='blue', label='valid accuracy')
plt.xlabel('x')
plt.ylabel('y')
plt.title('accuracy')
plt.grid(True)
plt.show()

plt.figure(2)
plt.plot(x, train_loss, color='red', label='train loss')
plt.xlabel('x')
plt.ylabel('y')
plt.title('loss')
plt.show()

# 模型存储
i = 3
with open(f'{i}_W.pickle', 'wb') as f:
    pickle.dump(BP_net.Weights, f)

with open(f'{i}_b.pickle', 'wb') as f:
    pickle.dump(BP_net.Biases, f)
