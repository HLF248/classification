import torchvision.transforms as transforms
from import_ip102_by_torch import CustomDataset
import numpy as np


# 定义数据集的路径和文件
image_dir = r'C:/Users/13002/Desktop/datasets/IP102/Classification/ip102_v1.1/images'
label_dir = r'C:/Users/13002/Desktop/datasets/IP102/Classification/ip102_v1.1'
train_file = 'train.txt'
val_file = 'val.txt'
test_file = 'test.txt'

# 定义数据集的转换
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# 加载 IP102 数据集
# train_dataset = CustomDataset(image_dir, label_dir, train_file, transform=data_transform)
# val_dataset = CustomDataset(image_dir, label_dir, val_file, transform=data_transform)
test_dataset = CustomDataset(image_dir, label_dir, test_file, transform=data_transform)

# 初始化每个通道的均值和标准差
channel_mean = np.zeros(3)
channel_std = np.zeros(3)

# 计算每个通道的均值和标准差
for images, _ in test_dataset:
    for i in range(3):  # 遍历三个通道
        channel_mean[i] += images[i, :, :].mean()
        channel_std[i] += images[i, :, :].std()

channel_mean /= len(test_dataset)
channel_mean = np.round(channel_mean, 3)
channel_std /= len(test_dataset)
channel_std = np.round(channel_std, 3)

print("Mean:", channel_mean)
print("Std:", channel_std)

'''
train
Mean: [0.514, 0.535, 0.377]
Std: [0.191, 0.19, 0.19]

val
Mean: [0.514, 0.533, 0.378]
Std: [0.192, 0.19, 0.19]

test
Mean: [0.514 0.535 0.377]
Std: [0.191 0.19  0.189]
'''