import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.init as init

class ann(nn.Module):
    def __init__(self):
        super(ann,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(22,100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,1)
        )
    def forward(self, x):
        return self.model(x)

class dataset(Dataset):
    def __init__(self, train_x, train_y):
        self.x = train_x
        self.y = train_y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]

df = pd.read_csv('dataset.csv', delimiter=',', decimal = ',')
df = df.dropna()

Y = df['% Silica Concentrate']
X = df.drop(['% Silica Concentrate', 'date'], axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

X_train = X_train.values
X_test = X_test.values
Y_train = Y_train.values
Y_test = Y_test.values
print(type(X_train))

train_dataset = dataset(X_train, Y_train)
test_dataset = dataset(X_test, Y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 100, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 100, shuffle = True)

model = ann()
model = model.double()
if torch.cuda.is_available():
    model = model.cuda()

learning_rate = 0.000001
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
criterion = nn.MSELoss()

num_epochs = 10

for epoch in range(1,num_epochs,1):
    print("Epoch: ",epoch)
    running_loss = 0.0
    for i, dat in enumerate(tqdm(train_loader), 0):
        inputs, labels = dat
        inputs = inputs.resize_(100,1,22)
        labels = labels.resize_(100,1,1)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss = running_loss + loss.item()
    print("Training loss: ", running_loss/0.7)
    running_loss = 0.0
    for i, dat in enumerate(tqdm(test_loader), 0):
        inputs, labels = dat
        inputs = inputs.resize_(100,1,22)
        labels = labels.resize_(100,1,1)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(inputs)
        running_loss = running_loss + (np.linalg.norm(outputs.detach().numpy() - labels.detach().numpy()))**2
    print("Validation loss: ", running_loss/0.3)
