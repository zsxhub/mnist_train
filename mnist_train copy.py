import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

# 设置输出全部数组元素  不然以 ... 省略大部分
np.set_printoptions(threshold=np.inf)

# from tensorflow.examples.tutorials.mnist import input_data
mnist = tf.keras.datasets.mnist
# mnist = input_data.read_data_sets("MNIST_data/",one_hot="true")
(train_x,train_y), (test_x,test_y) = mnist.load_data()

    # 显示
# print(train_x.shape, train_x.dtype)
# print(train_y.shape, train_y.dtype)
# plt.axis("off")
# plt.imshow(train_x[10],cmap = "gray")
# plt.show()


# 进行归一化处理,方便神经网络训练
train_x = train_x / 255.0
test_x = test_x / 255.0

# 搭建神经网络模型 描述各层网络
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),                      # 第1层 拉直层 将数据变为1维
    tf.keras.layers.Dense(128, activation='relu'),  # 第2层 全连接层 128个神经元
    tf.keras.layers.Dense(10, activation='softmax') # 第3层 全连接层 10个神经元
])
# 配置神经网络的训练方法
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

#读取模型
checkpoint_save_path = "./checkpoint/mnist.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('---------------load the model-------------')
    model.load_weights(checkpoint_save_path)
#保存模型
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_save_path,
                                                 save_weights_only = True,
                                                #  monitor = 'val_loss',
                                                 save_best_only = True)
                                        
# 执行训练过程
history = model.fit(train_x, train_y,     #用于训练的数据
            batch_size=32,      # batch的大小
            epochs=10,           # 迭代多少次
            validation_data=(test_x, test_y),   # （测试集的输入特征 测试卷标签）
            # validation_split = 从训练集划分多少比例给测试集（0~1）
            validation_freq=1,  # 每迭代多少次在测试集中测试一下正确率
            callbacks = [cp_callback])
# 打印网络的结构和参数统计
model.summary()

file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

#显示loss和acc曲线
acc = history.history['sparse_categorical_accuracy']            # 训练集准确率
val_acc = history.history['val_sparse_categorical_accuracy']    # 测试集准确率
loss = history.history['loss']                                  # 训练集loss
val_loss = history.history['val_loss']                          # 测试集loss

plt.figure(figsize=(8,8))

plt.subplot(1,2,1)
plt.plot(acc,label = 'Training Accuracy')
plt.plot(val_acc,label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss,label = 'Training loss')
plt.plot(val_loss,label = 'Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()