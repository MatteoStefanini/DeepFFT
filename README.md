# DeepFFT
PyTorch FFT implementation for version 0.3 (before the built-in torch.fft was released in PyTorch 0.4) and GPU support with CUDA9.1.
We experimented in the Fourier domain with Deep Learning architectures for classifications tasks revolving around sequences, such as Video Action Recognition and Virtual Machines Classification.

deepUCF11 is the DeepFFT model training code for Video Action Recognition task over the UCF-11 dataset. Feature folder should contain frame's video features, which we extracted and stored with standard pretrained ResNet-152.

deepVM (available in a separate indipendent repository) contains the models tested over the Virtual Machines Dataset. We outperformed the state-of-the-art for Virtual Machine Identification with DeepConv and DeepFFT models. Paper have been accepted to CLOSER 2019.

## Requirements
+ Python 3.x
+ PyTorch 
+ Anaconda packages
+ Tensorboard-pytorch
