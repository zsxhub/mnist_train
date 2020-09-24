import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model


# 设置输出全部数组元素  不然以 ... 省略大部分
np.set_printoptions(threshold=np.inf)

mnist = tf.keras.datasets.mnist
(train_x,train_y), (test_x,test_y) = mnist.load_data()

# 进行归一化处理,方便神经网络训练
train_x = train_x / 255.0
test_x = test_x / 255.0

train_x = np.expand_dims(train_x, axis=3)
test_x = np.expand_dims(test_x, axis=3)

# 搭建神经网络模型 描述各层网络
class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')    # 激活层
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d1 = Dropout(0.2)          # dropout层

        self.c2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')  # 卷积层
        self.b2 = BatchNormalization()  # BN层
        self.a2 = Activation('relu')    # 激活层
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d2 = Dropout(0.2)          # dropout层

        self.flatten = Flatten()
        self.f1 = Dense(128, activation='relu')
        self.d3 = Dropout(0.2)
        self.f2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)
        x = self.d2(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d3(x)
        y = self.f2(x)
        return y


model = Baseline()

# 配置神经网络的训练方法
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              # Metrics 标注网络评测指标 y_是以数值形式给出，y 是以独热码形式给出。
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
            epochs=30,           # 迭代多少次
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

file = open('./acc_loss.txt','w')
file.write('sparse_categorical_accuracy\n')
file.write(str(acc)+'\n')
file.write('val_sparse_categorical_accuracy\n')
file.write(str(val_acc)+'\n')
file.write('loss\n')
file.write(str(loss)+'\n')
file.write('val_loss\n')
file.write(str(val_loss)+'\n')
file.close()

print('acc     ',acc)
print('val_acc ',val_acc)
print('loss    ',loss)
print('val_loss',val_loss)


# plt.figure(figsize=(8,8))

# plt.subplot(1,2,1)
# plt.plot(acc,label = 'Training Accuracy')
# plt.plot(val_acc,label = 'Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()

# plt.subplot(1,2,2)
# plt.plot(loss,label = 'Training loss')
# plt.plot(val_loss,label = 'Validation loss')
# plt.title('Training and Validation loss')
# plt.legend()

# plt.show()