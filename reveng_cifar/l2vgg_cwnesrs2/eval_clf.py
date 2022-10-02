#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 20:16:30 2021

@author: xiawei
"""

from time import perf_counter
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from setup.utils import loadcla, savefile
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PertbestData(Dataset):
    def __init__(self, list_IDs, pertbs, cates, labels):
        self.list_IDs = list_IDs
        self.pertbs= pertbs
        self.cates = cates
        self.labels = labels

    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self,index):
        ID = self.list_IDs[index]
        pertb = self.pertbs[ID]
        cate = self.cates[ID]
        label = self.labels[ID]
        return pertb,cate,label
    
def loadpertb(args):
    diff_train = np.load("./features/combined/comb_train.npy")
    diff_test = np.load("./features/combined/comb_test.npy")
    label_train = np.load("./features/combined/label_train.npy")
    label_test = np.load("./features/combined/label_test.npy")
    cate_train = np.load("./features/combined/cate_train.npy")
    cate_test = np.load("./features/combined/cate_test.npy")
    
    pertbtrain = PertbestData(range(diff_train.shape[0]),torch.tensor(diff_train),torch.tensor(cate_train),torch.tensor(label_train))
    train_loader = DataLoader(pertbtrain,batch_size = args.batch_size,shuffle = True)
    pertbtest = PertbestData(range(diff_test.shape[0]),torch.tensor(diff_test),torch.tensor(cate_test),torch.tensor(label_test))
    test_loader = DataLoader(pertbtest, batch_size = args.batch_size, shuffle = False)
    return train_loader, test_loader
    

def testClassifier(test_loader, model, verbose=False):
    model.eval()
    correct_cnt = 0
    total_cnt = 0
    for batch_idx, (x, target,_) in enumerate(test_loader):
        x, target = x.to(device), target.to(device)
        out = model(x)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
    acc = float(correct_cnt.double()/total_cnt)
    if verbose:
        print("The prediction accuracy on testset is {}".format(acc))
    return acc


def train_clf(train_loader, test_loader, model, args):
    
    model = model.train().to(device)

    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=0.9, weight_decay=args.weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    
    for epoch in range(args.num_epoch):
        # trainning
        t1_start = perf_counter()
        model.train()
        ave_loss = 0
        step = 0
        for batch_idx, (x, target, _) in enumerate(train_loader):           
            x, target = x.to(device), target.to(device)
            loss = criterion(model(x),target)
            ave_loss = ave_loss * 0.9 + loss.item() * 0.1    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if (batch_idx+1) % 500 == 0 or (batch_idx+1) == len(train_loader):
                print ('==>>> epoch: {}, batch index: {}, ave train loss: {:.6f}'.format(
                epoch, batch_idx+1, ave_loss))

        train_loss.append(ave_loss)
        train_acc.append(testClassifier(train_loader, model,verbose=True))   
        
        #testing
        model.eval()
        #ave_testloss = 0
        testloss = 0
        correct_cnt = 0
        total_cnt = 0
        for batch_idx, (x, target, _) in enumerate(test_loader):
            x, target = x.to(device), target.to(device)
            loss_batch = criterion(model(x),target)
            testloss += loss_batch.item()/len(test_loader) # average loss
            #ave_testloss = ave_testloss * 0.9 + loss_batch.item() * 0.1  # smoothed loss, latter batch have higher weights
            out = model(x)
            _, pred_label = torch.max(out.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()
   
            if (batch_idx+1) == len(test_loader):
                print ('==>>> epoch: {}, batch index: {}, test loss: {:.5f}'.format(
                epoch, batch_idx+1, testloss))
        test_loss.append(testloss)
        test_acc.append(float(correct_cnt.double()/total_cnt)) 
        t1_stop = perf_counter()
        print('epoch: {}, time: {}'.format(epoch,t1_stop-t1_start))
     
        #acc = testClassifier(test_loader, model)
        #test_acc.append(acc)
        print("Epoch: [%d/%d], Average Loss: %.4f, test.Acc: %.4f" %
              (epoch + 1, args.num_epoch, test_loss[epoch], test_acc[epoch]))
        
    
        if (epoch+1) % 5 == 0 or (epoch+1) == args.num_epoch:
            torch.save(model.state_dict(), "./savedadvclf/clf"+"_epoch"+str(epoch))
    print("smoothed training loss by epoch", train_loss)
    print("test loss by epoch", test_loss)
    print("train acc by epoch",train_acc)
    print("test acc by epoch", test_acc)
    np.savetxt(args.output+"/smoothed_training_loss_by_epoch.txt", train_loss)
    np.savetxt(args.output+"/test_loss_by_epoch.txt", test_loss)
    np.savetxt(args.output+"/test_acc_by_epoch.txt", test_acc)
    np.savetxt(args.output+"/train_acc_by_epoch.txt", train_acc)
    
    plt.figure()
    plt.plot(train_loss,'s-',color = "blue",linestyle = "dashed", label = "training")
    plt.plot( test_loss, 'o-',color = "red", linestyle = "dashed",label = "test")
    plt.legend()
    plt.xlabel("epoch")
    plt.title("Loss by Epoch")
    plt.savefig(args.output+"/loss")
    plt.show()
    
    plt.figure()
    plt.plot(train_acc,'s-',color = "blue",linestyle = "dashed", label = "train acc")
    plt.plot(test_acc,'o-',color = "red", linestyle = "dashed",label = "test acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.title("Test Acc by Epoch")
    plt.savefig(args.output+"/testacc")
    plt.show()
    #savefile(args.file_name, model, args.dataset)
    #return model


def evalClassifier(test_loader,model,epoch = str(99)):
    model.load_state_dict(torch.load("./savedadvclf/clf_epoch"+epoch))
    model.eval()
    pred = []
    for batch_idx, (x, target, _) in enumerate(test_loader):
        x, target = x.to(device), target.to(device)
        out = model(x)
        #print(out)
        #pred_label = out.cpu().data.numpy().argmax()
        _, pred_label = torch.max(out.data, 1)
        #print(pred_label.detach().cpu().numpy())
        pred.extend(pred_label.detach().cpu().numpy())
    return pred

def main(args):
    print('==> Setting root dir..')
    os.chdir(args.root)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('==> Loading data..')
    train_loader, test_loader = loadpertb(args)
    diff_train = np.load("./features/combined/pertb_train.npy")
    diff_test = np.load("./features/combined/pertb_test.npy")
    n1 = diff_train.shape[0]
    n2 = diff_test.shape[0]
    print("{} number of examples in training set ".format(n1))
    print("{} number of examples in test set".format(n2))
          
    print('==> Loading model..')
    model = loadcla(args)
    model = model.to(device)
    
    #print('==> Training starts..')            
    #train_clf(train_loader, test_loader, model, args)
    #testClassifier(test_loader, model, verbose=True)
    print("==> Evaluation Starts..")
    y_pred = evalClassifier(test_loader,model,epoch = str(99))
    y_test = np.load("./features/combined/cate_test.npy")
    print("pred",y_pred[:10])
    print("true",y_test[:10])
    acc = (y_pred == y_test).sum()/len(y_test)
    print(acc)
    from sklearn.metrics import multilabel_confusion_matrix
    
    table = multilabel_confusion_matrix(y_test, y_pred)
    print(table)
    from sklearn.metrics import confusion_matrix
    confusion = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix\n')
    print(confusion)
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred, target_names=("cw","cwnes","cwrs")))
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Training Pertbest Classifier')
    parser.add_argument("-d", '--dataset', choices=["mnist", "cifar10"], default="cifar10")  
    parser.add_argument("-n", "--num_epoch", type=int, default=3)
    #parser.add_argument("-f", "--file_name", default="/pertb_combined")
    parser.add_argument("-l", "--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--root", default="./")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_class", type=int, default=3)
    parser.add_argument("--init",default = None)# default="/vgg16")
    parser.add_argument("--output", default="./savedadvclf")
    args = parser.parse_args()
    #print(args)
    main(args)
