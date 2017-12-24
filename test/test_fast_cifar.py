# -*- encoding: UTF-8 -*-
'''
Created on 2017年12月22日

@author: yangbo
'''
from fast_cifar import FastCIFAR10
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
import logging
from torchvision.datasets.cifar import CIFAR10

def main():
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    basic_transforms = [
        transforms.ToTensor(),
        normalize
    ]
    fast_cifar10 = FastCIFAR10('../data', train=True, 
        transform=transforms.Compose(basic_transforms), final_shape=(50000,3,32,32))
    fast_cifar10.reset()
    
    cifar10 = CIFAR10('../data', train=True, 
        transform=transforms.Compose(basic_transforms))
    cifar10_1 = CIFAR10('../data', train=True, 
        transform=transforms.Compose(basic_transforms))
    
    f1,_ = fast_cifar10[0]
    c1,_ = cifar10[0]
    c2,_ = cifar10_1[0]
    print('f1,c1 Equals', np.array_equal(f1,c1))
    print('c1,c2 Equals', np.array_equal(c1,c2))
#     count = 0
#     for img,label in fast_cifar10:
#         print('class', label)
#         np_img = np.asarray(img)
#         plt.figure(figsize=(1,1))
#         plt.xticks([])
#         plt.yticks([])
#         plt.imshow(np_img, shape=(1,1))
#         plt.show()
#         count += 1
#         if count >= 10:
#             break


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.debug('Start')
    main()
