# pytorch-vgg-cifar10

This is the PyTorch implementation of VGG network trained on CIFAR10 dataset.

It can run in Windows.

### Requirements. 

[PyTorch](https://github.com/pytorch/pytorch)

[PyTorch for Windows](https://zhuanlan.zhihu.com/p/26871672) (In Chinese)

[torchvision](https://github.com/pytorch/vision)

python tee : pip install tee

### Download the model
The trained VGG model. 92.4% Accuracy [VGG](http://www.cs.unc.edu/~cyfu/cifar10/model_best.pth.tar)

### Evaluation 
	
	wget http://www.cs.unc.edu/~cyfu/cifar10/model_best.pth.tar
	python main.py --resume=./model_best.pth.tar -e
	
### Train with script! (16-bit precision) 
	
	./run.sh 
	
	or for windows:
	python run.py
	
Using the run.sh script to generate the training log and models of different versions of VGG in 16-bit or 32-bit precision.	
Then use the ipython notebook plot.ipynb to view the results.
	
![alt text](vgg_plot.png)

### Issues

When run in Windows with run.py, it will throw 'Memeory Error' exception
after 71 epoch. It is very strange. I still don't know how to fix it.

The stack trace like:

```
Traceback (most recent call last):
  File "main.py", line 294, in <module>
  File "<string>", line 1, in <module>
  File "F:\anaconda3\envs\ptc\lib\multiprocessing\spawn.py", line 105, in spawn_main
    exitcode = _main(fd)
  File "F:\anaconda3\envs\ptc\lib\multiprocessing\spawn.py", line 115, in _main
    self = reduction.pickle.load(from_parent)
MemoryError
  File "main.py", line 125, in main

  File "main.py", line 153, in train

  File "F:\anaconda3\envs\ptc\lib\site-packages\torch\utils\data\dataloader.py", line 310, in __iter__
    return DataLoaderIter(self)
  File "F:\anaconda3\envs\ptc\lib\site-packages\torch\utils\data\dataloader.py", line 167, in __init__
    w.start()
  File "F:\anaconda3\envs\ptc\lib\multiprocessing\process.py", line 105, in start
    self._popen = self._Popen(self)
  File "F:\anaconda3\envs\ptc\lib\multiprocessing\context.py", line 223, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
  File "F:\anaconda3\envs\ptc\lib\multiprocessing\context.py", line 322, in _Popen
    return Popen(process_obj)
  File "F:\anaconda3\envs\ptc\lib\multiprocessing\popen_spawn_win32.py", line 65, in __init__
    reduction.dump(process_obj, to_child)
  File "F:\anaconda3\envs\ptc\lib\multiprocessing\reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
BrokenPipeError: [Errno 32] Broken pipe
```

The BrokenPipeError is caused by the MemoryError. The children process die so
the pipe is broken.
