import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import seaborn as sns
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
from data.get_data import get_dataset

import torch.backends.cudnn as cudnn
from utils.utils import AverageMeter, accuracy
import os
import shutil
import time
from args import args

if args.arch == 'tc':
    # 采用torchvision提供的预训练的模型
    from torchvision.models import resnet34
else:
    # 采用自己设计的resnet34
    from models.resnet import resnet34
torch.manual_seed(2)
images_path = 'Data/'
train_laoder, test_loader = get_dataset(images_path)
model = resnet34(pretrained=True)
model = torch.nn.DataParallel(model.cuda())
cudnn.benchmark = True

# model parameters settings
epochs = 100

device = "device:0" if torch.cuda.is_available() else "cpu"

def train(train_loader, model, optimizer, criterion, lr, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda()
        model.train()
        output = model(input)
        loss = criterion(output, target)
        
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            output = ('Epoch: [{0}][{1}/{2}],\t lr: {lr:.5f}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), 
                loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)

def validate(test_loader, model, criterion, lr, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            if i % 10 == 0 or i == len(test_loader) - 1:
                output = ('Test: [{0}/{1}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(test_loader),  loss=losses,
                    top1=top1, top5=top5))
                print(output)
                

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)
    return top1.avg
def save_checkpoint(state, is_best):
    
    
    check_dir = "/home/wql/MC/GTZAN/codes/checkpoints/" + str(t) + '{}'.format(args.arch)
    if not os.path.isdir(check_dir):
        os.mkdir(check_dir)
    filename = '%s/ckpt.pth.tar' % (check_dir)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))

best_prec1 = 0    
lr = 0.01
t = time.time()
t = time.localtime(t)
t = time.strftime("%Y_%m_%d %H:%M:%S", t)
for epoch in range(epochs):
    
    if (epoch + 1) in [30, 50, 70, 90]:
        lr *= 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,  weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().cuda()
    train(train_laoder, model, optimizer, criterion, lr, epoch)

    if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
        prec1 = validate(test_loader, model, criterion, lr, epoch, )

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        

        output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
        print(output_best)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
