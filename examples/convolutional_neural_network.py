
from __future__ import print_function
# from sklearn import datasets
import matplotlib.pyplot as plt
import math
import numpy as np

import torch
import torchvision.transforms as transforms
from import_ip102_by_torch import CustomDataset
import time
import torch.nn as nn
from convnet import ConvNet
# print('yes')

# Import helper functions
from mlfromscratch.deep_learning import NeuralNetwork
from mlfromscratch.utils import train_test_split, to_categorical, normalize
from mlfromscratch.utils import get_random_subsets, shuffle_data, Plot
from mlfromscratch.utils.data_operation import accuracy_score
from mlfromscratch.deep_learning.optimizers import StochasticGradientDescent, Adam, RMSprop, Adagrad, Adadelta
from mlfromscratch.deep_learning.loss_functions import CrossEntropy
from mlfromscratch.utils.misc import bar_widgets
from mlfromscratch.deep_learning.layers import Dense, Dropout, Conv2D, Flatten, Activation, MaxPooling2D
from mlfromscratch.deep_learning.layers import AveragePooling2D, ZeroPadding2D, BatchNormalization, RNN


def main():

    #----------
    # Conv Net
    #----------

    optimizer = Adam()

    # 将 PIL 图像对象转换为张量
    # 定义图像转换操作，包括缩放和裁剪
    transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 缩放图像到256x256
    transforms.CenterCrop(224),      # 中心裁剪图像到224x224
    transforms.ToTensor()            # 转换为张量
    ])

    # 定义数据集的路径和文件
    image_dir = r'C:/Users/13002/Desktop/datasets/IP102/Classification/ip102_v1.1/images'
    label_dir = r'C:/Users/13002/Desktop/datasets/IP102/Classification/ip102_v1.1'
    train_file = 'train.txt'
    val_file = 'val.txt'
    test_file = 'test.txt'

    # 创建训练、验证和测试数据集
    train_dataset = CustomDataset(image_dir, label_dir, train_file, transform=transform)
    val_dataset = CustomDataset(image_dir, label_dir, val_file, transform=transform)
    test_dataset = CustomDataset(image_dir, label_dir, test_file, transform=transform)

    
    train_dataset_X = []
    train_dataset_y = []

    for i in range(len(train_dataset)):
        image, label = train_dataset[i]
        train_dataset_X.append(image)
        train_dataset_y.append(label)

    # 将列表转换为张量
    train_dataset_X = torch.stack(train_dataset_X)
    train_dataset_y = [int(s) for s in train_dataset_y]
    train_dataset_y = np.array(train_dataset_y)
    # Convert to one-hot encoding
    train_dataset_y = to_categorical(train_dataset_y)

    train_dataset_y = torch.tensor(train_dataset_y)

    print('train_dataset finish')

    test_dataset_X = []
    test_dataset_y = []

    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        test_dataset_X.append(image)
        test_dataset_y.append(label)

    # 将列表转换为张量
    test_dataset_X = torch.stack(test_dataset_X)
    test_dataset_y = [int(s) for s in test_dataset_y]
    test_dataset_y = np.array(test_dataset_y)
    # Convert to one-hot encoding
    test_dataset_y = to_categorical(test_dataset_y)

    test_dataset_y = torch.tensor(test_dataset_y)

    
    print('test_dataset finish')

    # 定义网络结构
    clf = NeuralNetwork(optimizer=optimizer,
                        loss=CrossEntropy,
                        validation_data=(test_dataset_X, test_dataset_y))

    clf.add(Conv2D(n_filters=16, filter_shape=(3,3), stride=1, input_shape=(3,224,224), padding='same'))
    clf.add(Activation('relu'))
    clf.add(Dropout(0.25))
    clf.add(BatchNormalization())
    clf.add(Conv2D(n_filters=32, filter_shape=(3,3), stride=1, padding='same'))
    clf.add(Activation('relu'))
    clf.add(Dropout(0.25))
    clf.add(BatchNormalization())
    clf.add(Flatten())
    clf.add(Dense(256))
    clf.add(Activation('relu'))
    clf.add(Dropout(0.4))
    clf.add(BatchNormalization())
    clf.add(Dense(102))
    clf.add(Activation('softmax'))

    print ()
    clf.summary(name="ConvNet")

    # # 创建模型实例
    # model = ConvNet()

    # # 定义损失函数和优化器
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    # # 训练模型
    # for epoch in range(20):
    #     optimizer.zero_grad()
    #     outputs = model(train_dataset_X)
    #     loss = criterion(outputs, train_dataset_y)
    #     loss.backward()
    #     optimizer.step()
    #     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 20, loss.item()))

    # # 测试模型
    # with torch.no_grad():
    #     outputs = model(test_dataset_X_tensor)
    #     _, predicted = torch.max(outputs.data, 1)
    #     total = test_dataset_y.size(0)
    #     correct = (predicted == test_dataset_y_tensor).sum().item()
    #     accuracy = correct / total
    #     print('Accuracy: {:.2f}%'.format(100 * accuracy))

    train_err, val_err = clf.fit(train_dataset_X, train_dataset_y, n_epochs=20, batch_size=32)

    # Training and validation error plot
    n = len(train_err)
    training, = plt.plot(range(n), train_err, label="Training Error")
    validation, = plt.plot(range(n), val_err, label="Validation Error")
    plt.legend(handles=[training, validation])
    plt.title("Error Plot")
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.show()

    _, accuracy = clf.test_on_batch(test_dataset_X, test_dataset_y)
    print ("Accuracy:", accuracy)


    y_pred = np.argmax(clf.predict(test_dataset_X), axis=1)
    test_dataset_X = test_dataset_X.reshape(-1, 3, 224*224)
    # Reduce dimension to 2D using PCA and plot the results
    Plot().plot_in_2d(test_dataset_X, y_pred, title="Convolutional Neural Network", accuracy=accuracy, legend_labels=range(102))
  

if __name__ == "__main__":
    # 记录代码开始执行的时间
    start_time = time.time()

    main()

    # 记录代码执行结束的时间
    end_time = time.time()

    # 计算代码的运行时间
    execution_time = end_time - start_time

    # 打印代码的运行时间
    print("代码执行时间：", execution_time, "秒")  
