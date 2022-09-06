import torch
from torch import nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(768, 2048)
        self.fc5 = nn.Linear(2048, 1)
        self.act = torch.nn.ReLU()
        self.drop = nn.Dropout(p=0.1)
        # self.act = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        y = self.fc5(x)
        return y

class Classifier2(nn.Module):
    def __init__(self):
        super(Classifier2, self).__init__()
        self.fc1 = nn.Linear(768, 2048)
        self.fc5 = nn.Linear(2048, 5)
        self.act = torch.nn.ReLU()
        self.drop = nn.Dropout(p=0.1)
        # self.act = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        y = self.fc5(x)
        return y

class Classifier3(nn.Module):
    def __init__(self):
        super(Classifier3, self).__init__()
        self.fc1 = nn.Linear(768, 2048)
        self.fc5 = nn.Linear(2048, 3)
        self.act = torch.nn.ReLU()
        self.drop = nn.Dropout(p=0.1)
        # self.act = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        y = self.fc5(x)
        return y