from __future__ import print_function
from __future__ import division
from PIL import Image
import os.path
import numpy as np
import sys
import multiprocessing
from math import ceil
import ctypes
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
from multiprocessing import Array, Process
import logging as log
import torch

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

# configure logging for multiprocessing
log.basicConfig(level=log.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def _transform_proc(shm_array, shape, range_index, transformer):
    """ 
    Args:
        shape ( tuple ) : the original data shape
        range_index ( (start, end) tuple ): index of samples, end is exclusive.
    """
    np_array = np.ctypeslib.as_array(shm_array.get_obj())
    np_array = np_array.reshape(shape)
    pid = multiprocessing.current_process().pid
    log.debug('[pid=%d] To transform the data[%s]', pid, range_index)
    for index in range(*range_index):
        img = np_array[index]
        img = img.astype(np.uint8)          # can not use ndarray.view() because that will interpret double bytes as uint8 other than cast it!
        img = Image.fromarray(img, 'RGB')
        img = transformer(img)
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        else:
            img = np.asarray(img)
        row = np_array[index].reshape(-1)
        row[:] = img.reshape(-1)

def numpy_to_ctyps_type(numpy_type):
    type_map = {
        'uint8': ctypes.c_uint8,
        'uint16': ctypes.c_uint16,
        'uint32': ctypes.c_uint32,
        'float32': ctypes.c_float,
        'float64': ctypes.c_double
    }
    type_of_ctyps = type_map[numpy_type]
    return type_of_ctyps

class FastCIFAR10(data.Dataset):
    """`FastCIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        num_workers (int, optional): The number of workers to parallel processing 
            transform functions. Default is the number of CPU.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                download=False, num_workers=None, final_shape=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.num_workers = num_workers
        self.final_shape = final_shape  # the data final shape after transform
        
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the pickled numpy arrays
        self._load_data()
        if self.final_shape == None:
            self.final_shape = self.train_data.shape
        # allocate shared memory
        self._shm_train_raw = Array(ctypes.c_float,         # always float type
                                    self.train_data.size)
        # wrap the shared memory array into numpy ndarray
        self._shm_train_nda = np.ctypeslib.as_array(self._shm_train_raw.get_obj())

    def reset(self):
        """ Reset the data set, now just do transformation again. """
        self._transform_dataset()
        
    def _transform_parallel(self):
        if self.num_workers == None:
            num_workers = multiprocessing.cpu_count()
        # partition the data
        shape = self.train_data.shape
        row_num = shape[0]  # suppose dimension as N.C.W.H
        rows_per_worker = ceil(row_num / num_workers)
        workers = []
        index_list = list(range(0,row_num, rows_per_worker))
        for n,start in enumerate(index_list):
            if n == len(index_list)-1:
                end = row_num
            else:
                end = index_list[n+1]
            args = (self._shm_train_raw, shape, (start, end), self.transform)
            a = np.ctypeslib.as_array(self._shm_train_raw.get_obj())
            a = a.reshape(self.train_data.shape)
            proc = Process(target=_transform_proc, args=args)
            proc.deamon = True
            proc.start()
            workers.append(proc)
        # wait for all subprocess finish
        for proc in workers:
            proc.join()
        # reshape
        self._shm_train_nda = self._shm_train_nda.reshape(self.final_shape)

    def _transform_dataset(self):
        """ Transform the train data parallel in multiple subprocess.
            But the transformation will not change the row order.
        """
        # 1. Copy data-set to shared memory
        # 2. Do transform in parallel, suspend current execution until all subprocess done.
        
        self._shm_train_nda = self._shm_train_nda.reshape(self.train_data.shape)    # we need to use the original data shape instead of final shape
        # 1.
        self._shm_train_nda[:] = self.train_data[:]   # Copy the original data into share memory based ndarray
        # 2.
        self._transform_parallel()
    
    def _load_data(self):
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self._shm_train_nda[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
            
        return img, target
    
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True


    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)


class CIFAR100(FastCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `FastCIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

if __name__ == '__main__':
    print('fast_cifar in main.')