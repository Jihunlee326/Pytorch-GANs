| title | category | tag |
|:--------:|:--------:|:--------:|
|Advanced GANs|generative model|GAN|

# Pytorch GANs for ultrasound images
PyTorch implementations of Generative Adversarial Networks.  
Generative Adversarial Network using ultrasound image.

* To do list
  - 초음파영상 불러오는 함수 만들기
  - 학습 결과 이미지 첨부하기
  - Network 이미지화 하기


## Development Environment
* NVIDIA GTX 1080 ti
* cuda 8.0
* python 3.5.3
* pytorch 0.4.0
* torchvision 0.2.1
* CPU also possible



### Vanilla GAN
_Generative Adversarial Network_

[[Paper]](https://arxiv.org/abs/1406.2661) [[Code]](models/GAN/network.py)


### DCGAN
_Deep Convolutional Generative Adversarial Network_

[[Paper]](https://arxiv.org/abs/1511.06434) [[Code]](models/DCGAN/network.py)


### ACGAN
_Auxiliary Classifier Generative Adversarial Network_

[[Paper]](https://arxiv.org/abs/1610.09585) [[Code]](models/ACGAN/network.py)


## Acknowledgements
This implementation has been based on [this repository](https://github.com/eriklindernoren/PyTorch-GAN) and tested with Pytorch over ver0.4.0 on Windows 10 and Ubuntu 14.04.
