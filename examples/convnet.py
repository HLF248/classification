import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32*224*224, 256)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 102)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.dropout(x, p=0.25)
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.dropout(x, p=0.25)
        x = x.view(-1, 32*224*224)
        x = F.relu(self.batchnorm3(self.fc1(x)))
        x = F.dropout(x, p=0.4)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


