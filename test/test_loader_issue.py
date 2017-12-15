'''    
Created on 20171214

@author: yangbo
'''
import sys
import time
sys.path.insert(0, '../src')
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tee import StdoutTee, StderrTee

BASE = ".."

class Args(dict):
    def __getattr__(self, item):
        return self[item]

args = Args()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def test_loader():
    global args
    args.batch_size = 128
    args.workers = 4

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=BASE+'/data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=BASE+'/data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    meter_train = AverageMeter()
    meter_test = AverageMeter()
    time_begin = time.time()
    for epoch in range(300):
        meter_train.reset()
        meter_test.reset()
        time_start = time.time()
        for i, (input, target) in enumerate(train_loader):
#             print('epoch', epoch, 'train input', input.size(),'target', target.size())
            pass
        time_end = time.time()
        meter_train.update(time_end-time_start)
        print('Load train data for epoch {} takes {:.2f} seconds'.format(epoch, meter_train.sum))
        time_start = time.time()
        for i, (input, target) in enumerate(val_loader):
#             print('epoch', epoch, 'test input', input.size(),'target', target.size())
            pass
        time_end = time.time()
        meter_test.update(time_end-time_start)
        # print meter for one epoch
        print('Load test data for epoch {} takes {:.2f} seconds'.format(epoch, meter_test.sum))
    cost = time.time()-time_begin
    print('Total cost: {:.2f} seconds = {:.2f} minutes'.format(cost, cost/60.))
    
if __name__ == '__main__':
    with StdoutTee(BASE+'/logs/test', buff=1024), StderrTee(BASE+'/logs/test_error', buff=1024):
        test_loader()
    
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
