import os
import torch
from PIL import Image
from torch.utils.data import Dataset



device = 'cuda'

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_file = label_file
        self.transform = transform
        self.data = self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
            # print(image.size())
        # image = image.unsqueeze(0)
        image = image.to(device)
        label = torch.tensor(label, dtype=torch.long).to(device)
        return image, label

    def _load_data(self):
        data = []
        labels = self._load_labels(os.path.join(self.label_dir, self.label_file))
        with open(os.path.join(self.label_dir, self.label_file), 'r', encoding='utf-8') as file:
            for line in file:
                image_name, _ = line.strip().split()
                image_path = os.path.join(self.image_dir, image_name)
                if os.path.exists(image_path):
                    data.append((image_path, labels[image_name]))
                else:
                    print(f"Image '{image_name}' not found.")
        return data

    def _load_labels(self, file_path):
        labels = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                image_name, label = line.strip().split()
                labels[image_name] = int(label)
        return labels


# from torchvision import transforms
# # 定义数据集的路径和文件
# image_dir = r'C:/Users/13002/Desktop/datasets/IP102/Classification/ip102_v1.1/images'
# label_dir = r'C:/Users/13002/Desktop/datasets/IP102/Classification/ip102_v1.1'
# train_file = 'train.txt'
# val_file = 'val.txt'
# test_file = 'test.txt'

# # 定义数据预处理           
# def data_transform():
#     return transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         # transforms.ToTensor()
#     ])

# test_dataset = CustomDataset(image_dir, label_dir, test_file, transform=data_transform())
# # print(test_dataset.data)
# from PIL import Image

# def show_sample(sample):
#     image, label = sample
#     return image

# # 加载图片
# image = show_sample(test_dataset[1])
image = Image.open(r"C:\Users\13002\Desktop\datasets\IP102\Classification\ip102_v1.1\images\00210.jpg")

# 获取图片大小
width, height = image.size

# 计算每块的大小
block_width = width // 3
block_height = height // 3

# 切割图片
blocks = []
for i in range(3):
    for j in range(3):
        left = j * block_width
        upper = i * block_height
        right = left + block_width
        lower = upper + block_height
        block = image.crop((left, upper, right, lower))
        blocks.append(block)

# 显示切割后的图片块
for i, block in enumerate(blocks):
    block.show()
    block.save(f'block_{i}.jpg')  # 保存图片块到文件

# # 定义函数以显示PIL图像和标签
# def show_sample(sample):
#     image, label = sample  
#     plt.imshow(image)
#     plt.title(f'Label: {label}')
#     plt.show()

# # 打印训练数据集的前几个样本
# print("Train Dataset Samples:")
# for i in range(5):
#     show_sample(train_dataset[i])
    
# # 将 PIL 图像对象转换为张量
# # 定义图像转换操作，包括缩放和裁剪
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  # 缩放图像到256x256
#     transforms.CenterCrop(224),      # 中心裁剪图像到224x224
#     transforms.ToTensor()            # 转换为张量
# ])

# # # 定义函数以显示PIL图像和标签
# # def show_sample(sample):
# #     image, label = sample
# #     plt.imshow(image)
# #     plt.title(f'Label: {label}')
# #     plt.show()

# BATCH_SIZE = 32
# # 定义数据集的路径和文件
# image_dir = r'C:/Users/13002/Desktop/datasets/IP102/Classification/ip102_v1.1/images'
# label_dir = r'C:/Users/13002/Desktop/datasets/IP102/Classification/ip102_v1.1'
# train_file = 'train.txt'
# val_file = 'val.txt'
# test_file = 'test.txt'

# # 创建训练、验证和测试数据集
# train_dataset = CustomDataset(image_dir, label_dir, train_file, transform=transform)
# val_dataset = CustomDataset(image_dir, label_dir, val_file)
# test_dataset = CustomDataset(image_dir, label_dir, test_file)

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

# print(train_dataset.data)
# print(train_dataset.__getitem__(1)[0].shape)
# print(type(train_loader))

# # 打印训练数据集的前几个样本
# print("Train Dataset Samples:")
# for i in range(5):
#     show_sample(train_dataset[i])

'''
# 定义要展示的图像数量
num_samples = 10

# 创建一个包含前num_samples个样本的子数据集
subset_dataset = torch.utils.data.Subset(train_dataset, range(num_samples))

# 创建一个展示图像的图像网格
# plt.figure(figsize=(15, 8))
# for i in range(num_samples):
#     image, label = subset_dataset[i]
#     image = transforms.ToPILImage()(image)
    
#     # 在多个子图中展示每张图像
#     plt.subplot(2, 5, i + 1)  # 创建一个2x5的子图布局，每个子图依次排列
#     plt.imshow(image, cmap='gray')  # 显示灰度图像
#     plt.title(f"Label: {label}")
#     plt.axis('off')  # 关闭坐标轴

# plt.tight_layout()  # 调整子图之间的间距
# plt.show()
'''

# # 测试输出一个样本的形状和显示图像
# image, label = train_dataset[0]
# print("Image shape:", image.shape)
# print("Label:", label)

# # 将张量转换为PIL Image并显示图像
# image = transforms.ToPILImage()(image)
# plt.imshow(image)
# plt.axis('off')
# plt.show()


# # 示例输出：打印数据集大小
# print(f"Train dataset size: {len(train_dataset)}")
# print(f"Validation dataset size: {len(val_dataset)}")
# print(f"Test dataset size: {len(test_dataset)}")

# # 示例输出：打印第一个样本的图片和标签
# image, label = train_dataset[0]
# print(f"Sample image: {image}")
# print(f"Label: {label}")

# # 记录代码开始执行的时间
# start_time = time.time()

# train_dataset_X = []
# train_dataset_y = []

# image, label = train_dataset[1]
# print(image.size())
# print(type(label))

# for i in range(len(train_dataset)):
#     image, label = train_dataset[i]
#     train_dataset_X.append(image)
#     train_dataset_y.append(label)

# X = np.array(train_dataset_X)
# y = np.array(train_dataset_y)

# print("train_dataset_X shape:", X.shape)
# print("train_dataset_y shape:", y.shape)

# # 将列表转换为张量
# train_dataset_X = torch.stack(train_dataset_X)
# train_dataset_y = [int(s) for s in train_dataset_y]
# train_dataset_y = torch.tensor(train_dataset_y)

# # 输出张量的形状
# print("train_dataset_X shape:", train_dataset_X.shape)
# print("train_dataset_y shape:", train_dataset_y.shape)

# # 记录代码执行结束的时间
# end_time = time.time()

# # 计算代码的运行时间
# execution_time = end_time - start_time

# # 打印代码的运行时间
# print("代码执行时间：", execution_time, "秒")

# # 以训练集为例，查看数据集的结构
# image, label = train_dataset[1]
# print("Image structure:", image)
# print("Label:", label)
# print(type(label))