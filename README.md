# Pytorch GANs for ultrasound images
PyTorch implementations of Generative Adversarial Networks.  
Generative Adversarial Network using ultrasound image.

|<center>title</center>|<center>category</center>|<center>tag</center>|
|:--------|:--------:|--------:|
|<center>Advanced GANs</center>|<center>generative model</center>|<center>GAN</center>|

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
