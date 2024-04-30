from import_ip102_by_torch import CustomDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from scipy.stats import gmean
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
device = 'cuda'

from tqdm import tqdm
from training_test_plot import plot_accuracy_change, visualize_prediction, plot_error_curve
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.init as init

def initialize_model(variant, is_pretrained=True):
    model_functions = {
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnext50': models.resnext50_32x4d,
        'resnet152': models.resnet152,
        'vit_base_16': models.vit_b_16,
        'vit_base_32': models.vit_b_32,
        'vit_large_16': models.vit_l_16,
        'alexnet': models.alexnet,
        'vgg11': models.vgg11,
        'vgg11_bn': models.vgg11_bn,
        'vgg13': models.vgg13,
        'vgg16_bn': models.vgg16_bn,
        'vgg19_bn': models.vgg19_bn,
        'googlenet': models.googlenet
    }

    if variant not in model_functions:
        raise ValueError(f"Unsupported model variant: {variant}")

    model_function = model_functions[variant]
    print(f'{variant} {"pretrained" if is_pretrained else "training from scratch"}')
    return model_function(pretrained=is_pretrained)

model_names = [
    'resnet50', # 0
    'resnet101', # 1
    'resnext50', 
    'resnet152', 
    'vit_base_16', 
    'vit_base_32', 
    'vit_large_16', 
    'alexnet',
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg16_bn',
    'vgg19_bn',
    'googlenet' # 13
]

def training_from_last(model, model_name=model_names[0]):
    # 从上次训练的结果开始训练
    best_model_file = f'D:/my_checkpoints/{model_name}_best.pth'
    model.load_state_dict(torch.load(f'{best_model_file}'))
    print('already load the parameters of the best')

def redefining_classifier(model, num_classes, model_name, isPretrained=True):

    if model_name.startswith('res'):
        # print(model.fc)
        new_classifier = nn.Linear(model.fc.in_features, num_classes)
        model.fc = new_classifier
        # print(model.fc.weight.shape)
        # print(model.fc)

        if isPretrained:
            # 冻结除了fc层以外的所有参数
            for name, param in model.named_parameters():
                if "fc" not in name:  
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            # 初始化分类层的参数为零
            model.fc.weight.data.zero_()
            model.fc.bias.data.zero_()
            # 随机初始化权重
            init.normal_(model.fc.weight.data, mean=0.0, std=0.01)
        else:
            # 初始化所有参数为零
            for param in model.parameters():
                param.data.zero_()

            # 随机初始化所有权重
            for param in model.parameters():
                
                if len(param.size()) > 1:  # 只对权重进行初始化，跳过偏置
                    init.normal_(param.data, mean=0.0, std=0.01)
                # print(param)
                # print(param.size())
            

    elif model_name.startswith('vit'):
        new_classifier = nn.Linear(model.heads.head.in_features, num_classes)
        model.heads.head = new_classifier
        if isPretrained:

            # 冻结除了heads层以外的所有参数
            for name, param in model.named_parameters():
                if "heads" not in name:  
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            # 初始化分类层的参数为零
            model.heads.head.weight.data.zero_()
            model.heads.head.bias.data.zero_()
            # 随机初始化权重
            init.normal_(model.heads.head.weight.data, mean=0.0, std=0.01)
        else:
            # 初始化所有参数为零
            for param in model.parameters():
                param.data.zero_()

            # 随机初始化所有权重
            for param in model.parameters():
                
                if len(param.size()) > 1:  # 只对权重进行初始化，跳过偏置
                    init.normal_(param.data, mean=0.0, std=0.01)

    else:
        print("程序已经暂停，请确定模型的类型后，按下回车键继续执行...")
        input()  # 等待用户按下回车键
        print("继续执行...")

    print('redefining classifier is done')

    return model

    
def train_with_error_tracking(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, model_name=model_names[0]):
    train_errors = []
    val_errors = []
    
    best_loss = 60 #1.8082 #10
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
        train_bar = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}, LR: {current_lr}')

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

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_loss = round(avg_train_loss, 4) 
        train_errors.append(avg_train_loss)

        # 查看哪些类中的样本噪声大 or 从哪些类中样本提取的特征更健壮
        if plot_acc % 10 == 0:
            plot_accuracy_change(train_accuracy_history, plot_acc, model_name=model_name)

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
        torch.save(model.state_dict(), f'D:/my_checkpoints/{model_name}_epoch_{epoch+1}.pth')
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            count = 0 
            # 保存模型参数
            torch.save(model.state_dict(), f'D:/my_checkpoints/{model_name}_best.pth')
        else:
            count += 1

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss} | Validation Loss: {avg_val_loss} | Train Acc: {100 * correct_train / total_train:.1f}% | Val Acc: {100 * correct_val / total_val:.1f}%")

        if count > 5:
            print(f'the current number of iterations: {epoch+1}')
            break 

    # 绘制误差曲线图
    plot_error_curve(train_errors, val_errors, model_name=model_name)    

def test_on_batch(model, test_loader, criterion, model_name=model_names[0]):

    best_model_file = f'D:/my_checkpoints/{model_name}_best.pth'
    # r"D:/Research/resnet50_0.497.pkl"
    
    # 加载最好的模型参数
    model.load_state_dict(torch.load(f'{best_model_file}'))
    print('already load the parameters of the best')
    model.eval()

    # 在测试集上评估模型
    test_loss = 0.0
    correct_test = 0.0
    total_test = 0.0

    visualize_prediction(model, test_loader, batch_index=14, model_name=model_name)

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

    plot_accuracy_change(test_acc_history, training=False, model_name=model_name)

    print(f"Test Loss: {avg_test_loss} | Test Acc: {100 * correct_test / total_test:.3f}%")

# 定义数据预处理           
def data_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

def creating_dataset(BATCH_SIZE):
    # 定义数据集的路径和文件
    image_dir = r'C:/Users/13002/Desktop/datasets/IP102/Classification/ip102_v1.1/images'
    label_dir = r'C:/Users/13002/Desktop/datasets/IP102/Classification/ip102_v1.1'
    train_file = 'train.txt'
    val_file = 'val.txt'
    test_file = 'test.txt'

    test_dataset = CustomDataset(image_dir, label_dir, test_file, transform=data_transform())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    train_dataset = CustomDataset(image_dir, label_dir, train_file, transform=data_transform())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    
    val_dataset = CustomDataset(image_dir, label_dir, val_file, transform=data_transform())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    return test_loader, train_loader, val_loader

def main():
    
    num_classes = 102
    LR = 1e-3
    BATCH_SIZE = 32
    iterations = 30
    times = 5
    EPOCH = iterations * times
    MODEL_NAME = model_names[3]
    isPretrained = False
    
    # 实例化一个预训练的模型
    # model = variants['resnet152']
    model = initialize_model('resnet152', isPretrained)

    model = redefining_classifier(model, num_classes, MODEL_NAME, isPretrained=isPretrained)

    model.to(device)
    
    if isPretrained:
        # 定义需要优化的参数
        if MODEL_NAME.startswith('res'):
            params_to_optimize = model.fc.parameters() 
        elif MODEL_NAME.startswith('vit'):
            params_to_optimize = model.heads.head.parameters()
        else:
            pass
    else:
        params_to_optimize = model.parameters()

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params_to_optimize, lr=LR)
    '''
    # training from scratch
    optimizer = optim.Adam(model.parameters(), lr=LR)
    '''

    # 创建学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, verbose=True)

    test_loader, train_loader, val_loader = creating_dataset(BATCH_SIZE)
    
    train_with_error_tracking(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=EPOCH, model_name=MODEL_NAME)

    test_on_batch(model, test_loader, criterion, model_name=MODEL_NAME)
    # test_on_batch(model, val_loader, criterion)

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
