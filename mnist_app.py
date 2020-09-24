import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

model_save_path = './checkpoint/mnist.ckpt'

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),   # 第1层 拉直层 将数据变为1维
    tf.keras.layers.Dense(128, activation='relu'),  # 第2层 全连接层 128个神经元
    tf.keras.layers.Dense(10, activation='softmax') # 第3层 全连接层 10个神经元
])
model.load_weights(model_save_path)

plt.figure()
plt.set_cmap('gray')
# 输入要测试的图像总数量
preNum = int(input("input the number of test picture: "))

for i in range(preNum):
    image_path = input("the path of test picture: ")
    image_path = './image/' + image_path + '.png'
    img = Image.open(image_path)

    plt.subplot(1,3,1)
    plt.imshow(img)

    # PIL有九种不同模式: 1，L，P，RGB，RGBA，CMYK，YCbCr，I，F
    # L -- 灰度图
    img = img.convert('L')

    plt.subplot(1,3,2)
    plt.imshow(img)

    img = img.resize((28,28),Image.ANTIALIAS)
    img_arr = np.array(img)

    for i in range(28):
        for j in range(28):
            if img_arr[i][j] < 210:
                img_arr[i][j] = 255
            else:
                img_arr[i][j] = 0

    plt.subplot(1,3,3)
    img = Image.fromarray(img_arr)
    plt.imshow(img)
    
    img_arr = img_arr / 255.0

    x_predict = img_arr[tf.newaxis,...]

    result = model.predict(x_predict)
    pred = tf.argmax(result,axis=1)

    print('\n')
    tf.print(pred)

    plt.show()

    
    # plt.close()


