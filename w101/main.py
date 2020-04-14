#! /usr/bin/env python
import copy
import time
import argparse
import numpy as np
import os.path as osp
from PIL import Image

from utils import Evaluator

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.autograd import Variable

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Arguments
parser = argparse.ArgumentParser(description='PyTorch Deeplab v3 Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--gamma', type=float, default=2, metavar='M',
                    help='learning rate decay factor (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before '
                         'logging training status')
parser.add_argument('--save', type=str, default='model.pt',
                    help='file on which to save model weights')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

# Data loaders
root_dir = '/media/sda1/vision2020_01/Data'  # Change this based on the server you are using
train_loader = torch.utils.data.DataLoader(
    datasets.VOCSegmentation(
        root_dir, image_set='train', download=False, year='2012',
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])]),
        target_transform=transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
            transforms.ToTensor()])
        ),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.VOCSegmentation(
        root_dir, image_set='val', download=False, year='2012',
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])]),
        target_transform=transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
            transforms.ToTensor()])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

# Model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
cp_model = copy.deepcopy(model)

if args.cuda:
    model.cuda()

load_model = False
if osp.exists(args.save):
    with open(args.save, 'rb') as fp:
        state = torch.load(fp)
        model.load_state_dict(state)
        load_model = True

# Optimizer & loss
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
criterion = nn.CrossEntropyLoss(ignore_index=255)
metrics = Evaluator(21)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda() * 255.
        data, target = Variable(data), Variable(target).long().squeeze_(1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output['out'], target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(epoch):
    model.eval()
    metrics.reset()
    test_loss = 0.
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda() * 255.
        data, target = Variable(data), Variable(target).long().squeeze_(1)
        with torch.no_grad():
            output = model(data)
        test_loss += criterion(output['out'], target).item()
        pred = output['out'].cpu()
        pred = F.softmax(pred, dim=1).numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        metrics.add_batch(target, pred)

    Acc = metrics.Pixel_Accuracy()
    Acc_class = metrics.Pixel_Accuracy_Class()
    mIoU = metrics.Mean_Intersection_over_Union()
    FWIoU = metrics.Frequency_Weighted_Intersection_over_Union()

    print('Validation:')
    print('[Epoch: %d, numImages: %5d]' % (epoch, len(test_loader.dataset)))
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(
        Acc, Acc_class, mIoU, FWIoU))
    print('Loss: %.3f' % test_loss)
    return test_loss


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed
       by 10 at every specified step
       Adapted from PyTorch Imagenet example:
       https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    best_loss = None
    if load_model:
        best_loss = test(0)
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train(epoch)
            test_loss = test(epoch)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s '.format(
                epoch, time.time() - epoch_start_time))
            print('-' * 89)

            if best_loss is None or test_loss < best_loss:
                best_loss = test_loss
                with open(args.save, 'wb') as fp:
                    state = model.state_dict()
                    torch.save(state, fp)
            else:
                adjust_learning_rate(optimizer, args.gamma, epoch)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
