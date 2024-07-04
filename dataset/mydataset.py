import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

def loaddata():
    path = r'G:\ffp\ExperimentPaper\FDNet\Ex3\data\pisnetnorm.mat'
    traindata = h5py.File(path,'r')
    label = traindata['label']
    data = traindata['data']
    data = np.transpose(data,axes=[2,1,0])
    train_size = len(data)
    data = data[:, 0:512, 0:512]
    data = np.reshape(data,[train_size,1,512,512])
    data = data.astype(np.float32)
    label = np.transpose(label,axes=[2,1,0])
    label = label[:,0:512,0:512]
    label = np.reshape(label, [train_size, 1, 512, 512])
    label = label.astype(np.float32)
    return data,label

def loadtest():
    path = r'G:\ffp\ExperimentPaper\selfsupervised\data\ex3\test.mat'
    traindata = h5py.File(path,'r')
    label = traindata['label']
    data = traindata['data']
    data = np.transpose(data,axes=[2,1,0])
    train_size = len(data)
    data = data[:, 0:512, 0:512]
    data = np.reshape(data,[train_size,1,512,512])
    data = data.astype(np.float32)
    label = np.transpose(label,axes=[2,1,0])
    label = label[:,0:512,0:512]
    label = np.reshape(label, [train_size, 1, 512, 512])
    label = label.astype(np.float32)
    return data,label

def loadtestsentile():
    path = r'G:\ffp\data\Experiment\TFD\logsentile-1.mat'
    traindata = h5py.File(path,'r')
    # label = traindata['label']
    data = traindata['data']
    data = np.transpose(data,axes=[2,1,0])
    train_size = len(data)
    data = data[:, 0:512, 0:512]
    data = np.reshape(data,[train_size,1,512,512])
    data = data.astype(np.float32)
    # label = np.transpose(label,axes=[2,1,0])
    # label = label[:,0:512,0:512]
    # label = np.reshape(label, [train_size, 1, 512, 512])
    # label = label.astype(np.float32)
    return data,data

def loadtestall():
    path = r'G:\ffp\data\Experiment\fdnet\testsentile.mat'
    traindata = h5py.File(path,'r')
    label = traindata['label']
    data = traindata['data']
    data = np.transpose(data,axes=[2,1,0])
    train_size = len(data)
    data = data[:, 211:723, 211:723]
    data = np.reshape(data,[train_size,1,512,512])
    data = data.astype(np.float32)
    label = np.transpose(label,axes=[2,1,0])
    label = label[:, 211:723, 211:723]
    label = np.reshape(label, [train_size, 1, 512, 512])
    label = label.astype(np.float32)
    return data,label
# def loaddata():
#     path = r'G:\ffp\code\train.mat'
#     traindata = h5py.File(path,'r')
#     label = traindata['label']
#     data = traindata['data']
#     data = np.transpose(data,axes=[2,1,0])
#     train_size = len(data)
#     data = np.reshape(data,[train_size,1,723,723])
#     data = data.astype(np.float32)
#     label = np.transpose(label,axes=[2,1,0])
#     label = np.reshape(label, [train_size, 1, 723, 723])
#     label = label.astype(np.float32)
#     return data,label
# class mydataset(Dataset):
#     def __init__(self, train_x, train_y, transform=None):
#         super(mydataset, self).__init__()
#         self.x = train_x
#         self.y = train_y
#     def __getitem__(self, idx):
#         data = self.x[idx,:,:]
#         label = self.y[idx,:,:]
#         data = np.resize((512,512))
#         data = torch.from_numpy(data)
#         label = np.resize((512,512))
#         label = torch.from_numpy(label)
#         return data, label
#     def __len__(self):
#         return len(self.x)
class mydataset(Dataset):
    def __init__(self, train_x, train_y, transform=None):
        super(mydataset, self).__init__()
        self.x = torch.from_numpy(train_x)
        self.y = torch.from_numpy(train_y)
    def __getitem__(self, idx):
        data = self.x[idx,:,:]
        label = self.y[idx,:,:]
        return data, label
    def __len__(self):
        return len(self.x)

def loadtest1():
    path = r'G:\ffp\code\test1.mat'
    traindata = h5py.File(path,'r')
    label = traindata['label1']
    data = traindata['data1']
    data = np.transpose(data,axes=[2,1,0])
    train_size = len(data)
    data = np.reshape(data,[train_size,1,723,723])
    data = data.astype(np.float32)
    label = np.transpose(label,axes=[2,1,0])
    label = np.reshape(label, [train_size, 1, 723, 723])
    label = label.astype(np.float32)
    return data,label
if __name__ == '__main__':
    data,label = loadtest()
    print(len(data))
