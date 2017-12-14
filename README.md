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


