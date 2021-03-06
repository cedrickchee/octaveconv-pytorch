# Octave Convolution (OctConv)

## Pytorch implementation of Octave Convolution with other similar operation

This is **third-party/un-official** implementation of the following papers which are presented in [Recent_Convolution.pdf](https://github.com/cedrickchee/octaveconv-pytorch/blob/master/Recent_Convolution.pdf):

1. Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution
[paper](https://arxiv.org/abs/1904.05049).
![](fig/OctConv_detailed_design.png)
2. Adaptively Connected Neural Networks.(CVPR 2019)
[paper](https://arxiv.org/abs/1904.03579).
![](fig/adaptive_conv.png)
3. Res2net:A New Multi-scale Backbone Architecture
[paper](https://arxiv.org/abs/1904.01169).
![](fig/res2net.png)

## Plan

- [x] Add Res2Net block with SE-layer
- [x] Add Adaptive-Convolution: both pixel-aware and dataset-aware (done)
- [ ] Add HetConv (optional): (_if I have time :slightly_smiling_face:_)
- [ ] Train on CIFAR
- [ ] Train on ImageNet (_Who can help me train this repo on ImageNet?_)

## Requirements

- Python 3
  - Tested with Python 3.6
- PyTorch
  - Tested with version 1.0.1

## Usage

Check model files under the `nn` directory.

```python
from nn.OCtaveResnet import resnet50
from nn.res2net import se_resnet50
from nn.AdaptiveConvResnet import PixelAwareResnet50, DataSetAwareResnet50

model = resnet50().cuda()
model = se_resnet50().cuda()
model = PixelAwareResnet50().cuda()
model = DataSetAwareResnet50().cuda()
```

## Credits:

Referenced these implementations:

1. OctaveConv: MXNet implementation [here](https://github.com/terrychenism/OctaveConv)
2. AdaptiveCov: Offical tensorflow implementation [here](https://github.com/wanggrun/Adaptively-Connected-Neural-Networks)  

## Other Implementations

- PyTorch
  - https://github.com/gan3sh500/octaveconv-pytorch
  - https://github.com/vivym/OctaveConv.pytorch
- TensorFlow / Keras
  - https://github.com/CyberZHG/keras-octave-conv
  - https://github.com/l5shi/Octave-Conv-Keras

## License

MIT License
