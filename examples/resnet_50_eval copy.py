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
from mlfromscratch.examples.training_test_plot import plot_accuracy_change, visualize_prediction, plot_error_curve
from torch.optim.lr_scheduler import ReduceLROnPlateau

def evaluate_model_with_metrics(model, test_loader, criterion):
    model.eval()
    total_correct = 0
    total_samples = 0
    y_true = []
    y_pred = []
    # device = torch.device("cuda")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)  # 将数据移动到GPU上
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    
    accuracy = total_correct / total_samples
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 计算 ROC 曲线和 AUC
    y_true_one_hot = np.eye(len(np.unique(y_true)))[y_true]
    y_score = np.eye(len(np.unique(y_true)))[y_pred]
    roc_auc = roc_auc_score(y_true_one_hot, y_score, average='macro')

    # 计算 G-mean
    fpr, tpr, _ = roc_curve(y_true_one_hot.ravel(), y_score.ravel())
    g_mean_val = gmean([tpr[i] * (1 - fpr[i]) for i in range(len(fpr))])

    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')
    print(f'MAUC: {roc_auc:.4f}')
    print(f'G-mean: {g_mean_val:.4f}')

def train_with_error_tracking(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    train_errors = []
    val_errors = []
    
    best_loss = 10 #1.8082 #10
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
        torch.save(model.state_dict(), f'D:/my_checkpoints/202400425resnet50_pretrained/vit_epoch_{epoch+1}.pth')
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            count = 0 
            # 保存模型参数
            torch.save(model.state_dict(), f'D:/my_checkpoints/202400425resnet50_pretrained/vit_best.pth')
        else:
            count += 1

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss} | Validation Loss: {avg_val_loss} | Train Acc: {100 * correct_train / total_train:.1f}% | Val Acc: {100 * correct_val / total_val:.1f}%")

        if count > 5:
            print(f'the current number of iterations: {epoch+1}')
            break 

    # 绘制误差曲线图
    plot_error_curve(train_errors, val_errors)    

def main():
    num_classes = 102
    LR = 1e-3
    BATCH_SIZE = 32
    iterations = 30
    times = 5
    EPOCH = iterations * times
    
    # 实例化一个预定义的ResNet50模型
    model = models.resnet50(pretrained=True)
    # model.conv1

    # 加载预训练好的模型参数
    # pretrained_dict = torch.load('D:/Research/resnet50_0.497.pkl')

    # # 冻结模型的参数
    # for param in model.parameters():
    #     param.requires_grad = False

    # 冻结除了fc层以外的所有参数
    for name, param in model.named_parameters():
        if "fc" not in name:  
            param.requires_grad = False
        else:
            param.requires_grad = True


    model.fc.out_features = num_classes
    # print(model.fc.out_features)
    # print(model)

    # # 获取模型的全连接层
    # num_ftrs = model.fc.in_features
    # print(num_ftrs)

    # # 修改全连接层以适应新任务的类别数量
    # model.fc = nn.Linear(num_ftrs, num_classes)

    # # 获取当前模型的参数字典
    # model_dict = model.state_dict()

    # # 过滤掉不匹配的键值对
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # # 更新当前模型的参数字典
    # model_dict.update(pretrained_dict)

    # # 将更新后的参数字典加载到模型中
    # model.load_state_dict(model_dict)

    # 定义需要优化的参数
    params_to_optimize = model.fc.parameters()  # 只优化新添加的分类层的参数

    model.to(device)
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params_to_optimize, lr=LR)
    # 创建学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, verbose=True)

    # 定义优化器，使用 Adam 优化器
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 定义数据预处理
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    # transforms.Normalize(mean=[0.514, 0.533, 0.378], std=[0.192, 0.190, 0.190])

    # 定义数据集的路径和文件
    image_dir = r'C:/Users/13002/Desktop/datasets/IP102/Classification/ip102_v1.1/images'
    label_dir = r'C:/Users/13002/Desktop/datasets/IP102/Classification/ip102_v1.1'
    train_file = 'train.txt'
    val_file = 'val.txt'
    test_file = 'test.txt'

    test_dataset = CustomDataset(image_dir, label_dir, test_file, transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    '''
    batch_size = 16
    Test Accuracy: 0.43551881161855077
    Precision: 0.41916519639175
    Recall: 0.3202450410849115
    F1-score: 0.33078321918287856
    MAUC: 0.6572725734654075
    G-mean: 0.0
    代码执行时间： 145.12288880348206 秒

    batch_size = 128
    Test Accuracy: 0.4356
    Precision: 0.4192
    Recall: 0.3201
    F1-score: 0.3307
    MAUC: 0.6572
    G-mean: 0.0000
    代码执行时间： 128.97461819648743 秒
    '''
    train_dataset = CustomDataset(image_dir, label_dir, train_file, transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    '''
    batch_size = 16
    Test Accuracy: 0.5196
    Precision: 0.5311
    Recall: 0.4152
    F1-score: 0.4305
    MAUC: 0.7052
    G-mean: 0.0000
    代码执行时间： 263.9060547351837 秒

    batch_size = 128
    Test Accuracy: 0.5196
    Precision: 0.5310
    Recall: 0.4153
    F1-score: 0.4305
    MAUC: 0.7052
    G-mean: 0.0000
    代码执行时间： 318.45568203926086 秒
    '''
    val_dataset = CustomDataset(image_dir, label_dir, val_file, transform=data_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    '''
    batch_size = 32
    Test Accuracy: 0.4331
    Precision: 0.4253
    Recall: 0.3147
    F1-score: 0.3288
    MAUC: 0.6545
    G-mean: 0.0000
    代码执行时间： 46.70589637756348 秒

    batch_size = 64
    Test Accuracy: 0.4333
    Precision: 0.4253
    Recall: 0.3147
    F1-score: 0.3288
    MAUC: 0.6545
    G-mean: 0.0000
    代码执行时间： 41.993184089660645 秒

    batch_size = 128
    Test Accuracy: 0.4333
    Precision: 0.4253
    Recall: 0.3147
    F1-score: 0.3288
    MAUC: 0.6545
    G-mean: 0.0000
    代码执行时间： 42.33985757827759 秒

    batch_size = 256
    Test Accuracy: 0.4333
    Precision: 0.4253
    Recall: 0.3147
    F1-score: 0.3288
    MAUC: 0.6545
    G-mean: 0.0000
    代码执行时间： 41.88062238693237 秒

    Test Accuracy: 0.4064
    Precision: 0.4145
    Recall: 0.2767
    F1-score: 0.2935
    MAUC: 0.6353
    G-mean: 0.0000
    代码执行时间： 42.51197648048401 秒

    Test Accuracy: 0.4069
    Precision: 0.4160
    Recall: 0.2784
    F1-score: 0.2956
    MAUC: 0.6362
    G-mean: 0.0000
    代码执行时间： 45.934088945388794 秒
    '''

    # # 使用评估函数评估模型
    # evaluate_model_with_metrics(model, val_loader, criterion)
    train_with_error_tracking(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=EPOCH)



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