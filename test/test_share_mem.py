# -*- encoding: UTF-8 -*-
'''
Created on 2017年12月17日

参考共享内存的文档：
- https://docs.python.org/3.6/library/multiprocessing.html
- https://docs.python.org/3.6/library/multiprocessing.html#multiprocessing.Value
- https://docs.python.org/3.6/library/multiprocessing.html#multiprocessing.Array
- https://docs.python.org/3.6/library/multiprocessing.html#module-multiprocessing.sharedctypes
- https://docs.python.org/3.6/library/ctypes.html#module-ctypes
- https://docs.python.org/3.6/library/ctypes.html#fundamental-data-types

The multiprocessing.sharedctypes module provides functions for allocating 
ctypes objects from shared memory which can be inherited by child processes.

ctype Array 只能是一些简单的值数组，而要变为 numpy，则需要从这些 raw bytes 中创建
一个 ndarray 对象。用下面的函数可以从 buffer 中创建 ndarray：

   https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.frombuffer.html

但是要从 ctype array 来创建一个 numpy ndarray，需要参考下面的方法：
- https://stackoverflow.com/questions/4355524/getting-data-from-ctypes-array-into-numpy
- https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.ctypeslib.html#numpy.ctypeslib.as_array


其他参考：
   How does numpy organize memory:
      https://docs.scipy.org/doc/numpy-1.13.0/reference/internals.html#internal-organization-of-numpy-arrays
  - data buffer
  - array metadata  

@author: yangbo
'''
from multiprocessing import Process, Value, Array, RawArray, Pipe
import numpy as np
import os
import sys


def test_shared_mem_performance():
    pass

def procfn(shared_array, pipe):
    print('sub process started...')
    na = np.ctypeslib.as_array(shared_array.get_obj())
    na[0:100] = np.random.randn(100)
    print('sub', na)
    print('sub process finish.', os.getpid())
    pipe.send('[pipe] sub process finish.')
    
if __name__ == '__main__':
    # 1.分配固定大小的共享内存
    size = 1000*700*1000*2
    shared_array1 = Array('d', size)  # 最大不能超过2G个元素
    # 2. 用 as_array 创建一个 numpy ndarray 对象
    na = np.ctypeslib.as_array(shared_array1.get_obj())
    
    na[:] = 7   # 测试一下 numpy 的函数
    print('na', na)
    print('na', na.shape)
    
    # 创建一个管道
    parent_conn, child_conn = Pipe()
    
    # 启动一个进程执行 procfn 函数
    proc = Process(target=procfn, args=(shared_array1,child_conn))
    proc.daemon = True
    proc.start()
    
    # 等待子进程输出内容
    print(parent_conn.recv())
    proc.join()     # 等待子进程结束
    
    # 使用 numpy 的函数
    nb = na.reshape((2,-1))
    print('nb', nb.shape)
    print('nb', nb)
    
    print('Press a key to exit.')
    sys.stdin.read(1)
    