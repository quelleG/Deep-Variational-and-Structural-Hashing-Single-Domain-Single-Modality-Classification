# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:55:39 2020

@ Author: ZhangYuan 16051635

@ E-mail:  1441965928@qq.com
           zydoc@foxmail.com

"""
import os
import math
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#from torchvision.datasets import MNIST
#from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter       


defaultcfg = {
    4 : [ 32, 'M', 32, 'M', 64],
}

class VAE(nn.Module):
    def __init__(self, dataset='cifar10', depth=4, init_weights=True, 
                 cfg=None, struct_length=4, struct_group_num=36):
        super(VAE, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        if dataset == 'cifar10':
            num_classes = 10

        self.feature = self.make_feature_layers(cfg, True)

        self.encoder = nn.Sequential(
              nn.Linear(4096, 500), 
              nn.BatchNorm1d(500),
              nn.ReLU(inplace=True))
            
        self.fc_logvar = nn.Sequential(
              nn.Linear(500, 500),
              nn.BatchNorm1d(500),
              nn.ReLU(inplace=True))
        
        self.fc_mu = nn.Sequential(
              nn.Linear(500, 500),
              nn.BatchNorm1d(500),
              nn.ReLU(inplace=True))
        
        self.struct_layer_before = nn.Sequential(
                nn.Linear(500, 500), 
                nn.BatchNorm1d(500), 
                nn.ReLU(inplace=True),
                nn.Linear(500, 300),  
                nn.BatchNorm1d(300), 
                nn.ReLU(inplace=True))
        
        self.struct = []
        for i in range(struct_group_num):
            self.struct.append(self.make_struct_layer(struct_length))
        
        self.classifier = nn.Linear(struct_length * struct_group_num, 
                                    num_classes)

        if init_weights:
            self._initialize_weights()
            
            
    def make_struct_layer(self, length):
        layers = [nn.Linear(300, length), nn.BatchNorm1d(length), 
                  nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)


    def encode(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        return self.reparametrize(mu, logvar), mu, logvar


    def reparametrize(self, mu, logvar):
        z = torch.randn(mu.size(0), mu.size(1))
        if torch.cuda.is_available():
            z = z.cuda()  
        return mu + z * torch.exp(logvar/2)
    
    
    def forward(self, x):
        x = self.feature(x)
        x, mu, logvar = self.encode(x) 
        x = self.struct_layer_before(x)
        t = []
        for ti in self.struct:
            t.append(ti(x))
        
        hash_raw = torch.cat(t, dim=1)
        y = torch.sigmoid(self.classifier(hash_raw))
        return y, mu, logvar
    
    
    def make_feature_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, 
                                   padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), 
                               nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Slimming VAE&&Struct CIFAR training')
#    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
#                        help='train with channel sparsity regularization')
#    parser.add_argument('--s', type=float, default=0.0001,
#                        help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--struct-length', type=int, default=4, metavar='N',
                    help='the length of struct block (default: 4)')
    parser.add_argument('--struct-group-num', type=int, default=36, metavar='N',
                    help='the num of struct block group (default: 36)')
    parser.add_argument('--eta', type=float, default=0.000001,
                        help='Loss = BCE + eta * KLD (eta default: 0.000001)') 
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train (default: 160)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                        help='path to save prune model (default: current directory)')
    parser.add_argument('--depth', default=4, type=int,
                        help='depth of the feature network')
    
    args = parser.parse_args()
    args.cuda = True  #not args.no_cuda and torch.cuda.is_available()
    
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, 
                           transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
         
               
    model = VAE(dataset='cifar10', depth=args.depth, init_weights=True, 
                 cfg=None, struct_length=args.struct_length, 
                 struct_group_num=args.struct_group_num)
    
    if args.cuda and torch.cuda.is_available():
        model.cuda()
    
    optimizer = optim.SGD(model.parameters(), 
                          lr=args.lr, momentum=args.momentum, 
                          weight_decay=args.weight_decay)
#    optimizer = optim.Adam(model.parameters(), lr=args.lr, 
#                           weight_decay=args.weight_decay)
    
#    if args.resume:
#        if os.path.isfile(args.resume):
#            print("=> loading checkpoint '{}'".format(args.resume))
#            checkpoint = torch.load(args.resume)
#            args.start_epoch = checkpoint['epoch']
#            best_prec1 = checkpoint['best_prec1']
#            model.load_state_dict(checkpoint['state_dict'])
#            optimizer.load_state_dict(checkpoint['optimizer'])
#            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
#                  .format(args.resume, checkpoint['epoch'], best_prec1))
#        else:
#            print("=> no checkpoint found at '{}'".format(args.resume)) 
    
    # additional subgradient descent on the sparsity-induced penalty term
    def updateBN():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.grad.data.add_(args.s*torch.sign(m.weight.data)) # L1
            if isinstance(m, nn.BatchNorm1d):
                m.weight.grad.data.add_(args.s*torch.sign(m.weight.data)) # L1

    def BKloss(output, target, mu, logvar):
        criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            criterion = criterion.cuda()
        BCE = criterion(output, target)
        
        KLD = torch.sum(mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)).mul_(-0.5)
        return BCE + args.eta * KLD
     

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output, mu, logvar = model(data) 
            loss = BKloss(output, target, mu, logvar)
            writer.add_scalar('Train/Loss', loss.item(), epoch)
            writer.flush()
            loss.backward()
#            if args.sr:# True
#            updateBN()
            optimizer.step()
            
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{}]'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset)))

    
    def test(epoch):
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output, mu, logvar = model(data) 
            loss = BKloss(output, target, mu, logvar)
            test_loss += float(loss)
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1] 
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        prec1 = 100. * float(correct) / float(len(test_loader.dataset))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset), prec1))
        
        # Record loss and accuracy from the test run into the writer
        writer.add_scalar('Test/Loss', loss, epoch)
        writer.add_scalar('Test/Accuracy(top1)', prec1, epoch)
        writer.flush()
        return prec1
    
    
    def save_checkpoint(state, is_best, filepath):
        torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), 
                            os.path.join(filepath, 'model_best.pth.tar'))
    

    writer = SummaryWriter('runs/VAE2')
    best_prec1 = 0.
#    dummy_input = torch.rand(20, 3, 32, 32)
#    if torch.cuda.is_available():
#        dummy_input = dummy_input.cuda()
#    writer.add_graph(model, (dummy_input,))
#    writer.flush()
    for epoch in range(args.start_epoch, args.epochs):
        if epoch in [args.epochs*0.5, args.epochs*0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
                
        train(epoch)
        prec1 = test(epoch)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filepath=args.save)
    
        for i, (name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name, param) 
        writer.flush()

    print("Best accuracy: "+str(best_prec1))
    writer.close()


if __name__ == '__main__':
    main()
