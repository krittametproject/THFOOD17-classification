import torch
from torch.utils.data import DataLoader as dalo
from torch.utils.data import random_split
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from torchvision import transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(path='', batch=30, transform=None):
    train_dataset = datasets.ImageFolder(path+'/train', transform=transform)
    test_dataset = datasets.ImageFolder(path+'/test', transform=transform)
    n_train = round(len(train_dataset)*0.8)
    train_data, valid_data = random_split(train_dataset, [n_train, len(train_dataset)-n_train])
    train_set = dalo(train_data, batch_size=batch, shuffle=True)
    valid_set = dalo(valid_data)
    test_set = dalo(test_dataset)
    return train_set, valid_set, test_set

def train_(model, train_set, valid_set):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    valid_loss_min = np.Inf
    print('##########  START TRAINING  ##########')
    epoch = 0
    last_epoch = len(train_set)
    for data, label in train_set:
        batch_size_train = data.size()[0]
        epoch += 1
        model.train()
        train_loss = 0
        valid_loss = 0
        #data = data.type(torch.float32).clone().detach().requires_grad(True)
        #label = label.type(torch.float32).clone().detach().requires_grad(True)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        model.eval()
        for data, label in valid_set:
            batch_size_valid = len(data)
            data = data.type(torch.float32)
            label = label.type(torch.float32)
            output = model(data)
            loss = criterion(output, label.long())
            valid_loss += loss.item()
        print('train_loss', train_loss)
        print('valid loss', valid_loss)
        train_loss = train_loss/batch_size_train
        valid_loss = valid_loss/len(valid_set)
        print('{} from {} --> Training Loss: {:.6} --> Valid Loss: {:.6}'.format(epoch, last_epoch, train_loss, valid_loss))
        if valid_loss <= valid_loss_min:
            print('**Valid loss decrease from {:.6} --> {:.6}. Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss
    print('########## FINISH TRAINING ##########')

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 512, kernel_size=5, stride=1, padding=2), #512x256x256
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2) #512x128x128
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=5, stride=1, padding=2), #256*128*128
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)#256*64*64
        )
        self.n_fc1 = 64*64*256
        self.fc1 = torch.nn.Linear(self.n_fc1, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 17)
    
    def forward(self, x):
        relu = torch.nn.ReLU()
        softmax = torch.nn.Softmax(dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, self.n_fc1)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = softmax(self.fc3(x))
        return x

if __name__ ==  '__main__':
    tran = transforms.Compose([transforms.Resize((256, 256)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.5, 0.5, 0.5],                     
                                              std=[0.5, 0.5, 0.5])])
    train_dataset, valid_dataset, test_dataset = load_data('./dataset',30,transform=tran)
    model = CNN()
    summary(model, (3, 256, 256))
    train_(model, train_dataset, valid_dataset)  

        
    



