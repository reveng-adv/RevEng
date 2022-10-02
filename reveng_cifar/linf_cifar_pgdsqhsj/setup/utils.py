import os,sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

    
def loaddata(args):
    if args.dataset == 'mnist':
        trans = transforms.Compose([transforms.ToTensor()])
        trainset = datasets.MNIST(root=args.root+'/data', train=True, transform=trans, download=True)
        testset = datasets.MNIST(root=args.root+'/data', train=False, transform=trans, download=True)
        
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        trainset = datasets.CIFAR10(root=args.root+"/data",
                                train=True,download=True,transform=transform_train)        
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)                
        transform_test = transforms.Compose([transforms.ToTensor()])
        testset = datasets.CIFAR10(root=args.root+"/data",
                                train=False,download=True,transform=transform_test)
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)    
    else:
        print("unknown dataset")
    return train_loader, test_loader


def loadmodel(args):
    if args.dataset == 'mnist':
        from setup.setup_model import BasicCNN
        model = BasicCNN()
    elif args.dataset == 'cifar10':       
        from setup.setup_model import vgg16
        model = vgg16()
    else:
        print("unknown model")
        return
    if args.init:
        print("Loading pre-trained model")
        model.load_state_dict(torch.load("./models/"+args.dataset+args.init))
    return model


def savefile(file_name, model, dataset):
    if file_name != None:
        root = os.path.abspath(os.path.dirname(sys.argv[0]))+"/models/"+dataset
        if not os.path.exists(root):
            os.mkdir(root)
        torch.save(model.state_dict(), root+file_name)
    return


def randomdata(args, indices=None):
    if args.dataset == 'mnist':
        trans = transforms.Compose([transforms.ToTensor()])
        testset = datasets.MNIST(root=args.root+'/data', train=False, transform=trans, download=True)
        if indices is None:
            indices = torch.randperm(len(testset))[:args.n_samples]
        else:
            indices = torch.tensor(indices)
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, sampler=indices)
    elif args.dataset == 'cifar10':
        transform_test = transforms.Compose([transforms.ToTensor()])
        testset = datasets.CIFAR10(root=args.root+"/data",
                                train=False,download=True,transform=transform_test)
        if indices is None:
            indices = torch.randperm(len(testset))[:args.n_samples]
        else:
            indices = torch.tensor(indices)
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, sampler=indices)    
    else:
        print("unknown dataset")
    return test_loader, indices


class AdvData(Dataset):
    def __init__(self, list_IDs, data, predicts):
        self.predicts = predicts
        self.list_IDs = list_IDs
        self.data = data
       
    def __len__(self):
        return len(self.list_IDs)
   
    def __getitem__(self,index):
        ID = self.list_IDs[index]
               
        x = self.data[ID]
        y = self.predicts[ID]    
        return x, y


def loadadvdata(args, indices=None):
    if args.method == 'natural':
        indices = torch.tensor(np.load("./features/"+args.dataset+'/indices.npy'))
        data_loader, _ = randomdata(args, indices=indices)
    else:
        adv_data = np.load("./features/"+args.dataset+'/'+args.method+'_adv.npy')
        adv_label = np.load("./features/"+args.dataset+'/'+args.method+'_labels.npy')
        advdata = AdvData(range(adv_data.shape[0]), torch.tensor(adv_data),torch.tensor(adv_label))
        data_loader = DataLoader(advdata, batch_size=args.batch_size, shuffle=False)
    return data_loader


def loadcomadv(args):
    data_train = np.load("./features/"+args.dataset+'/adv_train.npy')
    data_test = np.load("./features/"+args.dataset+'/adv_test.npy')
    label_train = np.load("./features/"+args.dataset+'/adv_train_label.npy')
    label_test = np.load("./features/"+args.dataset+'/adv_test_label.npy')

    advtrain = AdvData(range(data_train.shape[0]), torch.tensor(data_train),torch.tensor(label_train))
    train_loader = DataLoader(advtrain, batch_size=args.batch_size, shuffle=True)
    advtest = AdvData(range(data_test.shape[0]), torch.tensor(data_test),torch.tensor(label_test))
    test_loader = DataLoader(advtest, batch_size=args.batch_size, shuffle=False) 
    return train_loader, test_loader

def loadcla(args):
    if args.dataset == 'mnist':
        from setup.setup_model import AdvClaMnist
        model = AdvClaMnist(num_class=args.num_class)
    elif args.dataset == 'cifar10':       
        from setup.setup_model import AdvClaCIFAR10
        model = AdvClaCIFAR10(num_class=args.num_class)
    else:
        print("unknown model")
        return
    if args.init:
        print("Loading pre-trained model")
        model.load_state_dict(torch.load("./models/"+args.dataset+args.init))
    return model
