import torch
import sys


def infer():
    best_epoch = 273   # from 0 start
    checkpoint_file = '../save_da_vgg19_bn/checkpoint_{}.tar'.format(best_epoch)
    checkpoint = torch.load(checkpoint_file)
    print('best epoch is', checkpoint['epoch'])
    
    # import vgg
    sys.path.insert(0, '../src')
    import vgg
    
    model = vgg.vgg19_bn()
    model.features = torch.nn.DataParallel(model.features)
    state = checkpoint['state_dict']
    model.load_state_dict(state)
    print(model.parameters)

if __name__ == '__main__':
    infer()