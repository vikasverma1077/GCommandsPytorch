from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np


criterion = nn.CrossEntropyLoss().cuda()



def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam




def train(loader, model, optimizer, epoch, cuda, log_interval, args, lamba_mod_mean, verbose=True):
    model.train()
    global_epoch_loss = 0
    for batch_idx, (data, target) in enumerate(loader):
        bs = target.size(0) 
        if cuda:
            data, target = data.cuda(), target.cuda()
        if args.train == 'vanilla':
            data, target = Variable(data), Variable(target)        
            output = model(data)
            loss = F.nll_loss(output, target)
        
        elif args.train == 'mixup':
            inputs, targets_a, targets_b, lam = mixup_data(data, target, args.mixup_alpha, args.use_cuda)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
            output = model(inputs)
            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
            
        elif args.train == 'mixup_new':
            inputs, targets_a, targets_b, lam = mixup_data(data, target, args.mixup_alpha, args.use_cuda)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
            output = model(inputs)
            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
            loss_scale = torch.abs(loss.detach().data.clone())
            
            num_class = args.num_classes # TO DO 
            y_onehot = Variable(torch.cuda.FloatTensor(bs, num_class).zero_())
            y_onehot.scatter_(1, target.view(bs, 1), 1)
            
            x = torch.autograd.Variable(data)
            f = model(x)
            b = y_onehot - torch.softmax(f, dim=1)
            loss_new = torch.sum(f * b, dim=1)
            #negative_index = torch.nonzero(loss_new.data < args.threshold).squeeze().detach().data.clone()
            #loss_new = (1.0 - lamba_mod_mean) * torch.sum(loss_new[negative_index]) / bs
            loss_new = (1.0 -lamba_mod_mean) * torch.sum(torch.abs(loss_new)) / bs
            loss = loss - (args.mixup_eta * loss_new)
            loss_new_scale = torch.abs(loss.detach().data.clone())
            loss = (loss_scale / loss_new_scale) * loss    
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_epoch_loss += loss.item()
        if verbose:
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(loader.dataset), 100.
                    * batch_idx / len(loader), loss.item()))
    return global_epoch_loss / len(loader.dataset)


def test(loader, model, cuda, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(loader.dataset)
    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
    return test_loss
