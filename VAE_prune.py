import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import math


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
                                    num_classes).cuda()

        if init_weights:
            self._initialize_weights()
            
            
    def make_struct_layer(self, length):
        layers = [nn.Linear(300, length), nn.BatchNorm1d(length), 
                  nn.ReLU(inplace=True)]
        return nn.Sequential(*layers).cuda()


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
    # Prune settings
    parser = argparse.ArgumentParser(description='PyTorch Slimming VAE prune')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--depth', type=int, default=4,
                        help='depth of the VAEnet')
    parser.add_argument('--percent', type=float, default=1,
                        help='scale sparse rate (default: 0.5)')
    parser.add_argument('--model', default='C:/Users/14419/VAE_HASH_L1/logs/checkpoint.pth.tar', type=str, metavar='PATH',
                        help='path to the model (default: none)')
    parser.add_argument('--save', default='C:/Users/14419/VAE_HASH_L1/logs', type=str, metavar='PATH',
                        help='path to save pruned model (default: none)')
    parser.add_argument('--eta', type=float, default=0.000001,
                        help='loss rate between class and kl (default: 0.000001)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    
    model = VAE()
    if args.cuda:
        model.cuda()
    
    if args.model:
        if os.path.isfile(args.model):
            print("=> loading checkpoint '{}'".format(args.model))
            checkpoint = torch.load(args.model)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
#            print(checkpoint['state_dict'])
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                  .format(args.model, checkpoint['epoch'], best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    print(model)
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]
    
    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size
    
    y, i = torch.sort(bn)
    thre_index = int(total * args.percent) - 1
    thre = y[thre_index] - 0.1
#    print(y)
    
    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
#            print(weight_copy)
            if torch.cuda.is_available():
                mask = weight_copy.gt(thre.cuda()).float().cuda()    
                ## attention cuda
            else:
                mask = weight_copy.gt(thre).float()
                
            pruned = float(pruned + mask.shape[0] - torch.sum(mask))
#            print(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            
            if int(torch.sum(mask)) == 0:       ####
                cfg.append(1)
            else:
                cfg.append(int(torch.sum(mask)))
                
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
            
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')
    
    pruned_ratio = pruned/total
    print('Pre-processing Successful! pruned_ratio: ' + str(pruned_ratio))
#    print(pruned/total)
    
    
    def BKloss(output, target, mu, logvar):
        criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            criterion = criterion.cuda()
        BCE = criterion(output, target)
        
        KLD = torch.sum(mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)).mul_(-0.5)
        return BCE + args.eta * KLD
    
    
    # simple test model after Pre-processing prune (simple set BN scales to zeros)
    def test(model):
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        if args.dataset == 'cifar10':
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
                batch_size=args.test_batch_size, shuffle=True, **kwargs)
        elif args.dataset == 'cifar100':
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
                batch_size=args.test_batch_size, shuffle=True, **kwargs)
        else:
            raise ValueError("No valid dataset is given.")
        model.eval()
        correct = 0
        test_loss = 0   

        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output, mu, logvar = model(data) 
            loss = BKloss(output, target, mu, logvar)
            test_loss += float(loss)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
        print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
                correct, len(test_loader.dataset), 
                100. * correct / len(test_loader.dataset)))
        
        return float(correct) / float(len(test_loader.dataset))
    
    
    acc = test(model)
    print(acc)
    
    # Make real prune
    print(cfg)
    newmodel = VAE()#vgg(dataset=args.dataset, cfg=cfg)
    if args.cuda and torch.cuda.is_available():
        newmodel.cuda()
    
    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    savepath = os.path.join(args.save, "prune.txt")
    with open(savepath, "w") as fp:
        fp.write("Configuration: \n"+str(cfg)+"\n")
        fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
        fp.write("Test accuracy: \n"+str(acc))
    
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
#            print(idx1)
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
#        elif isinstance(m0, nn.Linear):
##            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
#            print(start_mask.cpu().numpy())
#            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
#            if idx0.size == 1:
#                idx0 = np.resize(idx0, (1,))
#            m1.weight.data = m0.weight.data[:, idx0].clone()
#            m1.bias.data = m0.bias.data.clone()
    
    torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))
    
    print(newmodel)
    model = newmodel
    test(model)
    
if __name__ == '__main__':
    main()   
    
