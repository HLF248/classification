from import_ip102_by_torch import CustomDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# import torchvision.models as models
# from vit import ViT
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from scipy.stats import gmean
import numpy as np
from torchvision import datasets, transforms, models, utils
from torch.utils.data import DataLoader
import time

# import matplotlib.pyplot as plt
from tqdm import tqdm
# import glob
# import pandas as pd
# from textwrap import wrap
# from utils.data_operation import top_k_accuracy
# import logging
# from torchvision.transforms import ToPILImage
# from torchvision.transforms.functional import to_pil_image
# import itertools
# from vit import ViT
# import os
from mlfromscratch.examples.training_test_plot import plot_accuracy_change, visualize_prediction, plot_error_curve

# def plot_error_curve(train_errors, val_errors):
#     """
#     绘制误差曲线图
    
#     Args:
#     - train_errors: 包含每个 epoch 的训练误差的列表
#     - val_errors: 包含每个 epoch 的验证误差的列表
#     """
#     # 绘制误差曲线图
#     plt.plot(range(1, len(train_errors) + 1), train_errors, label='Training Error')
#     plt.plot(range(1, len(val_errors) + 1), val_errors, label='Validation Error')
#     plt.xlabel('Epochs')
#     plt.ylabel('Error')
#     plt.title('Training and Validation Error')
#     plt.legend()

#     save_dir = r'../vit_eval_pic'

#     if save_dir is not None:
#         # 确保保存目录存在
#         os.makedirs(save_dir, exist_ok=True)
#         # 拼接保存路径
#         filename = f'training_val_loss_overfitting'
#         save_path = os.path.join(save_dir, filename)
#         plt.savefig(save_path)
#         print(f"图像已保存到 {save_path}")

#     # plt.show()

# def plot_accuracy_change(train_accuracy_history, epoch=None, training=True):
#     """
#     绘制准确率变化曲线
    
#     Args:
#     - train_accuracy_history: 包含每个 batch 的训练准确率的列表
#     - epoch: the order number of the iterations
#     - training: whether it is a training process
#     """
#     # 绘制准确率变化曲线
#     plt.plot(range(0, len(train_accuracy_history)), train_accuracy_history, label='Training Accuracy')
#     plt.xlabel('# of batch')
#     plt.ylabel('Accuracy (%)')
#     plt.title('Training Accuracy Change')
#     plt.legend()

#     save_dir = r'../vit_eval_pic'

#     if training:

#         if save_dir is not None:
#             # 确保保存目录存在
#             os.makedirs(save_dir, exist_ok=True)
#             # 拼接保存路径
#             filename = f'training_acc_change_epoch_{epoch}'
#             save_path = os.path.join(save_dir, filename)
#             plt.savefig(save_path)
#             print(f"图像已保存到 {save_path}")
#     else:

#         if save_dir is not None:
#             # 确保保存目录存在
#             os.makedirs(save_dir, exist_ok=True)
#             # 拼接保存路径
#             filename = f'test_acc_change'
#             save_path = os.path.join(save_dir, filename)
#             plt.savefig(save_path)
#             print(f"图像已保存到 {save_path}")

#     # plt.show()

# def visualize_prediction(model, test_loader, batch_index=0):
#     """
#     预测并可视化部分结果
    
#     Args:
#     - model: 已训练的模型
#     - test_loader: 测试数据集的 DataLoader
#     - batch_index: 要可视化的批次数，默认为第一个batch
#     """
#     classes = class_name_table()

#     # 预测并可视化部分结果
#     # 创建 test_loader 的迭代器
#     data_iter = iter(test_loader)

#     # 使用 itertools.islice 来迭代到指定的批次位置
#     for _ in itertools.islice(data_iter, batch_index):
#         pass

#     images, labels = next(data_iter)
    
#     preds = model(images).softmax(1).argmax(1)
    
#     # 将预测结果从 CUDA 设备上移动到主机内存中
#     preds_cpu = preds.cpu()

#     # 创建子图
#     fig, axs = plt.subplots(2, 4, figsize=(13, 8))

#     # 显示图像和预测结果
#     [ax.imshow(image.cpu().permute(1,2,0)) for image,ax in zip(images,axs.ravel())]
#     [ax.set_title("\n".join(wrap(f'Accutual: {classes.name[label.item()]} Predicted: {classes.name[pred.item()]}',30)),color = 'g' if label.item()==pred.item() else 'r') for label,pred,ax in zip(labels,preds_cpu,axs.ravel())]
#     [ax.set_axis_off() for ax in axs.ravel()]

#     save_dir = r'../vit_eval_pic'

#     if save_dir is not None:
#         # 确保保存目录存在
#         os.makedirs(save_dir, exist_ok=True)
#         # 拼接保存路径
#         filename = f'part_of_predictions'
#         save_path = os.path.join(save_dir, filename)
#         plt.savefig(save_path)
#         print(f"图像已保存到 {save_path}")
#     # 显示图像
#     # plt.show()


def train_with_error_tracking(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    train_errors = []
    val_errors = []
    
    best_loss = 1.6488 #1.8082 #10
    count = 0 # 连续5次训练都没有降低val_loss就停止训练

    plot_acc = 0 # 绘制1st epoch的training accuracy change by batch

    for epoch in range(num_epochs):
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 训练模型
        model.train()
        train_loss = 0.0
        correct_train = 0.0
        total_train = 0.0

        # 初始化一个空列表来存储每个 batch 的准确率
        train_accuracy_history = []
        

        # 初始化 tqdm 进度条
        train_bar = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}, LR: {current_lr:.6f}')

        for _, (images, labels) in enumerate(train_bar):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
            
            # 在每个 batch 结束后计算准确率并记录
            train_accuracy = 100 * correct_train / total_train
            
            train_accuracy_history.append(train_accuracy)

            # 更新进度条
            train_bar.set_postfix(acc=train_accuracy, loss=train_loss / len(train_loader.dataset))

            # # 将张量转换为NumPy数组
            # outputs = outputs.cpu().numpy()
            # labels = labels.cpu().numpy()
            # loss = loss.cpu().numpy()

            # # 绘制等高线图
            # plt.figure(figsize=(8, 6))
            # contour = plt.contour(outputs, labels, loss, levels=20)
            # plt.clabel(contour, inline=True, fontsize=8)
            # plt.xlabel('Parameter 1')
            # plt.ylabel('Parameter 2')
            # plt.title('Contour Plot of Loss Function')
            # plt.grid(True)
            # plt.show()

        # train_loss /= len(train_loader.dataset)
        # train_loss = round(train_loss, 4) 
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_loss = round(avg_train_loss, 4) 
        train_errors.append(avg_train_loss)

        # 查看哪些类中的样本噪声大
        if plot_acc % 10 == 0:
            plot_accuracy_change(train_accuracy_history, plot_acc)

        plot_acc += 1

        # 在验证集上评估模型
        model.eval()
        val_loss = 0.0
        correct_val = 0.0
        total_val = 0.0
        val_bar = tqdm(val_loader, total=len(val_loader), desc=f'Epoch {epoch+1}/{num_epochs}')
        with torch.no_grad():
            for _, (images, labels) in enumerate(val_bar):
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

                val_bar.set_postfix(acc=100 * correct_val / total_val, loss=val_loss / len(val_loader.dataset))

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_loss = round(avg_val_loss, 4) 
        val_errors.append(avg_val_loss)

        # 更新学习率
        scheduler.step(round(avg_val_loss, 2))

        # 保存每个epoch的模型参数
        torch.save(model.state_dict(), f'D:/my_checkpoints/vit_epoch_{epoch+1}.pth')
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            count = 0 
            # 保存模型参数
            torch.save(model.state_dict(), f'D:/my_checkpoints/vit_best.pth')
        else:
            count += 1

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss} | Validation Loss: {avg_val_loss} | Train Acc: {100 * correct_train / total_train:.1f}% | Val Acc: {100 * correct_val / total_val:.1f}%")

        if count > 5:
            print(f'the current number of iterations: {epoch+1}')
            break 

    # 绘制误差曲线图
    plot_error_curve(train_errors, val_errors)

# def class_name_table():
#     f = open(r'C:/Users/13002/Desktop/datasets/IP102/Classification/classes.txt')
#     label = []
#     name = []
#     for line in f.readlines():
#         label.append(int(line.split()[0]))
#         name.append(' '.join(line.split()[1:]))
#     classes = pd.DataFrame([label, name]).T
#     classes.columns = ['label','name']
#     return classes

def test_on_batch(model, test_loader, criterion):

    best_model_file = 'D:/my_checkpoints/vit_best.pth'
    
    # 加载最好的模型参数
    model.load_state_dict(torch.load(f'{best_model_file}'))
    print('already load the parameters of the best')
    model.eval()

    # 在测试集上评估模型
    test_loss = 0.0
    correct_test = 0.0
    total_test = 0.0

    # classes = class_name_table()

    # # 预测并可视化部分结果
    # # 创建 test_loader 的迭代器
    # data_iter = iter(test_loader)

    # # 指定要获取的批次索引
    # batch_index = 4

    # # 使用 itertools.islice 来迭代到指定的批次位置
    # for _ in itertools.islice(data_iter, batch_index):
    #     pass

    # images, labels = next(data_iter)
    
    # preds = model(images).softmax(1).argmax(1)
    
    # # 将预测结果从 CUDA 设备上移动到主机内存中
    # preds_cpu = preds.cpu()
    
    # # 创建图像转换器
    # # to_pil = ToPILImage()

    # # 创建子图
    # fig, axs = plt.subplots(2, 4, figsize=(13, 8))

    # # 显示图像和预测结果
    # [ax.imshow(image.cpu().permute(1,2,0)) for image,ax in zip(images,axs.ravel())]
    # [ax.set_title("\n".join(wrap(f'Accutual: {classes.name[label.item()]} Predicted: {classes.name[pred.item()]}',30)),color = 'g' if label.item()==pred.item() else 'r') for label,pred,ax in zip(labels,preds_cpu,axs.ravel())]
    # [ax.set_axis_off() for ax in axs.ravel()]

    # # 显示图像
    # plt.show()
    visualize_prediction(model, test_loader, batch_index=4)

    test_acc_history = []

    test_bar = tqdm(test_loader, total=len(test_loader), desc='Testing')
    # 计算损失和准确率
    with torch.no_grad():
        for _, (images, labels) in enumerate(test_bar):
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)
            
            # 在每个 batch 结束后计算准确率并记录
            test_accuracy = 100 * correct_test / total_test
            
            test_acc_history.append(test_accuracy)

            test_bar.set_postfix(acc=test_accuracy, loss=test_loss / len(test_loader.dataset))

    avg_test_loss = test_loss / len(test_loader.dataset)
    avg_test_loss = round(avg_test_loss, 4) 

    plot_accuracy_change(test_acc_history, training=False)

    print(f"Test Loss: {avg_test_loss} | Test Acc: {100 * correct_test / total_test:.3f}%")



# 定义数据预处理           
def data_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

def main():
    device = 'cuda'
    # 定义数据集的路径和文件
    image_dir = r'C:/Users/13002/Desktop/datasets/IP102/Classification/ip102_v1.1/images'
    label_dir = r'C:/Users/13002/Desktop/datasets/IP102/Classification/ip102_v1.1'
    train_file = 'train.txt'
    val_file = 'val.txt'
    test_file = 'test.txt'

    LR = 1e-7
    BATCH_SIZE = 32
    iterations = 30
    times = 5
    EPOCH = iterations * times

    num_classes = 102

    train_dataset = CustomDataset(image_dir, label_dir, train_file, transform=data_transform())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    val_dataset = CustomDataset(image_dir, label_dir, val_file, transform=data_transform())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    test_dataset = CustomDataset(image_dir, label_dir, test_file, transform=data_transform())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 迁移学习
    # 实例化模型、损失函数和优化器
    model = models.vit_b_16(pretrained=True)
    # model = models.vit_b_16(pretrained=False)

    # 改变 ViT 模型的输出层的维度，即分类的类别的数量
    model.heads.head.out_features = num_classes
    # print(model)

    # 冻结除了heads层以外的所有参数
    for name, param in model.named_parameters():
        if "heads" not in name:  # 这里假设你的heads层的名字中包含了"heads"
            param.requires_grad = False
        else:
            param.requires_grad = True

    # 自定义的ViT模型体系结构
    # model = ViT(image_size=224, patch_size=16, num_classes=num_classes, dim=768, depth=12, heads=12, mlp_dim=3072)

    # # training a model from scratch
    # for param in model.parameters():
    #     param.data.zero_()

    # 初始化分类层的参数为零
    model.heads.head.weight.data.zero_()
    model.heads.head.bias.data.zero_()

    # 定义需要优化的参数
    params_to_optimize = model.heads.head.parameters()  # 只优化新添加的分类层的参数

    # 从100次训练后的基础再进行训练
    best_model_file = 'D:/my_checkpoints/vit_best.pth'
    # 加载最好的模型参数
    model.load_state_dict(torch.load(f'{best_model_file}'))
    print('already load the parameters of the best')
    
    # model.conv_proj

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params_to_optimize, lr=LR)
    # 创建学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, verbose=True)


    train_with_error_tracking(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=EPOCH)

    test_on_batch(model, test_loader, criterion)

if __name__ == "__main__":

    # 记录代码开始执行的时间
    start_time = time.time()

    main()

    # 记录代码执行结束的时间
    end_time = time.time()

    # 计算代码的运行时间
    execution_time = end_time - start_time

    # 将秒数转换为时分秒格式
    days, remainder = divmod(execution_time, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    # 打印代码的运行时间
    print("代码执行时间：{}天{}时{}分{}秒".format(int(days), int(hours), int(minutes), int(seconds)))