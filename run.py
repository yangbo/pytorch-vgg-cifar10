"""
Script to training the CIFAR10 with different CNN architectures.

Create: 2017-12-14
Author: Bob Yang (bo.yang@telecwin.com) 北京塔尔旺科技有限公司 http://www.telecwin.com
"""
import subprocess
import os

if not os.path.exists('logs'):
    os.mkdir('logs')

def run_arch(arch, in_half=False, option=''):
    """ Run training of the net architecture 'arch'. 
    Args:
        in_half: True to run in half precision mode.
    """
    half_opt = ''
    if in_half:
        half_opt = '--half'
    cmd = "python src/main.py --arch={model} {half} {option}".format(
        model=arch, half=half_opt, option=option)
    print('Run {arch} ...'.format(arch=arch))
    subprocess.call(cmd, stderr=subprocess.STDOUT)

def run_all(in_half):
    """ Run training for all the network architectures.
    Args:
        in_half: true to run in half float precision mode.
    """
    half_opt = ''
    if in_half:
        half_opt = '--half'
    for model in 'vgg11 vgg11_bn vgg13 vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn'.split(' '):
            cmd = "python main.py  --arch={model}  --save-dir=save_{model} {half}".format(model=model, half=half_opt)
            subprocess.call(cmd, stderr=subprocess.STDOUT)


if __name__ == '__main__':
    run_arch('vgg19_bn', in_half=False, option='--workers=1 --log-prefix=log_da_ --save-dir=save_da_vgg19_bn')
