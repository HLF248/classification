import matplotlib.pyplot as plt
import pandas as pd
import itertools
from textwrap import wrap
import os


plot_path = {
    'resnet50': r'D:/Research/ML-From-Scratch-master/mlfromscratch/resnet50_pic',
    'resnet101': r'D:/Research/ML-From-Scratch-master/mlfromscratch/resnet101_pic',
    'resnext50': r'D:/Research/ML-From-Scratch-master/mlfromscratch/resnext50_pic',
    'resnet152': r'D:/Research/ML-From-Scratch-master/mlfromscratch/resnet152_pic',
    'vit_base_16': r'D:/Research/ML-From-Scratch-master/mlfromscratch/vit_base_16_pic',
    'vit_base_32': r'D:/Research/ML-From-Scratch-master/mlfromscratch/vit_base_32_pic',
    'vit_large_16': r'D:/Research/ML-From-Scratch-master/mlfromscratch/vit_large_16_pic',
    'alexnet': r'D:/Research/ML-From-Scratch-master/mlfromscratch/alexnet_pic',
    'vgg11': r'D:/Research/ML-From-Scratch-master/mlfromscratch/vgg11_pic',
    'vgg11_bn': r'D:/Research/ML-From-Scratch-master/mlfromscratch/vgg11_bn_pic',
    'vgg13': r'D:/Research/ML-From-Scratch-master/mlfromscratch/vgg13_pic',
    'vgg16_bn': r'D:/Research/ML-From-Scratch-master/mlfromscratch/vgg16_bn_pic',
    'vgg19_bn': r'D:/Research/ML-From-Scratch-master/mlfromscratch/vgg19_bn_pic',
    'googlenet': r'D:/Research/ML-From-Scratch-master/mlfromscratch/googlenet_pic'
}


def class_name_table():
    f = open(r'C:/Users/13002/Desktop/datasets/IP102/Classification/classes.txt')
    label = []
    name = []
    for line in f.readlines():
        label.append(int(line.split()[0]))
        name.append(' '.join(line.split()[1:]))
    classes = pd.DataFrame([label, name]).T
    classes.columns = ['label','name']
    return classes

def plot_error_curve(train_errors, val_errors, model_name='vit_base'):
    """
    绘制误差曲线图
    
    Args:
    - train_errors: 包含每个 epoch 的训练误差的列表
    - val_errors: 包含每个 epoch 的验证误差的列表
    """
    plt.clf()
    # 绘制误差曲线图
    plt.plot(range(1, len(train_errors) + 1), train_errors, label='Training Error')
    plt.plot(range(1, len(val_errors) + 1), val_errors, label='Validation Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Training and Validation Error')
    plt.legend()

    save_dir = plot_path.get(model_name)  # 获取对应模型的图像的保存路径

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        filename = f'training_val_loss_overfitting.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"图像已保存到 {save_path}")
    else:
        filename = f'training_val_loss_overfitting.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"图像已保存到 {save_path}")
        
    # plt.show()

def plot_accuracy_change(train_accuracy_history, epoch=None, training=True, model_name='vit_base'):
    """
    绘制准确率变化曲线
    
    Args:
    - train_accuracy_history: 包含每个 batch 的训练准确率的列表
    - epoch: the order number of the iterations
    - training: whether it is a training process
    """
    # 绘制准确率变化曲线
    plt.plot(range(0, len(train_accuracy_history)), train_accuracy_history, label=f'Training Accuracy {epoch+1}' if training else 'Test Accuracy')
    plt.xlabel('# of batch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Change' if training else 'Test Accuracy Change')
    plt.legend()

    save_dir = plot_path.get(model_name)  # 获取对应模型的保存路径

    if training:

        if os.path.exists(save_dir):
            # 拼接保存路径
            filename = f'training_acc_change_epoch_{epoch+1}.png'
            save_path = os.path.join(save_dir, filename)
            # 检查是否有绘图元素存在
            if not plt.gca().has_data():
                print("没有绘制任何图形，请先绘制图形然后再保存。")
            else:
                plt.savefig(save_path)
                print(f"图像已保存到 {save_path}")
        else:
            os.makedirs(save_dir)
            # 拼接保存路径
            filename = f'training_acc_change_epoch_{epoch+1}.png'
            save_path = os.path.join(save_dir, filename)
            # 检查是否有绘图元素存在
            if not plt.gca().has_data():
                print("没有绘制任何图形，请先绘制图形然后再保存。")
            else:
                plt.savefig(save_path)
                print(f"图像已保存到 {save_path}")
    else:

        if os.path.exists(save_dir):
            
            # 拼接保存路径
            filename = f'test_acc_change.png'
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path)
            print(f"图像已保存到 {save_path}")
        else:
            os.makedirs(save_dir)
            # 拼接保存路径
            filename = f'test_acc_change.png'
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path)
            print(f"图像已保存到 {save_path}")
    # plt.show()

def visualize_prediction(model, test_loader, batch_index=0, model_name='vit_base'):
    """
    预测并可视化部分结果
    
    Args:
    - model: 已训练的模型
    - test_loader: 测试数据集的 DataLoader
    - batch_index: 要可视化的批次数，默认为第一个batch
    """
    classes = class_name_table()

    # 预测并可视化部分结果
    # 创建 test_loader 的迭代器
    data_iter = iter(test_loader)

    # 使用 itertools.islice 来迭代到指定的批次位置
    for _ in itertools.islice(data_iter, batch_index):
        pass

    images, labels = next(data_iter)
    
    preds = model(images).softmax(1).argmax(1)
    
    # 将预测结果从 CUDA 设备上移动到主机内存中
    preds_cpu = preds.cpu()

    # 创建子图
    fig, axs = plt.subplots(2, 4, figsize=(13, 8))

    # 显示图像和预测结果
    [ax.imshow(image.cpu().permute(1,2,0)) for image,ax in zip(images,axs.ravel())]
    [ax.set_title("\n".join(wrap(f'Accutual: {classes.name[label.item()]} Predicted: {classes.name[pred.item()]}',30)),color = 'g' if label.item()==pred.item() else 'r') for label,pred,ax in zip(labels,preds_cpu,axs.ravel())]
    [ax.set_axis_off() for ax in axs.ravel()]

    save_dir = plot_path.get(model_name)  # 获取对应模型的保存路径

    if os.path.exists(save_dir):
        
        # 拼接保存路径
        filename = r'part_of_predictions.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"图像已保存到 {save_path}")
    else:
        os.makedirs(save_dir)
        # 拼接保存路径
        filename = r'part_of_predictions.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"图像已保存到 {save_path}")
    
    # 清空当前 plt 对象中的内容
    plt.clf()
    # 显示图像
    # plt.show()