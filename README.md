# Lab7-Segmentation
Today's lab is about semantic segmentation. We will use the state-of-the-art architecture *DeepLab v3* to train a model to segment the Pascal VOC dataset. As its name suggest, this method is the third version of the DeepLab family, so we're going to review the first 2 before moving to the one we'll use. 

## DeepLab 1 & 2
<p align="center"><img src="https://miro.medium.com/max/1400/1*MVLmei6xOqScKjwffk4ZXg.png"/></p>

The input image goes through the backbone, which uses **atrous convolutions** (also called dilated convolutions) and **ASPP**. Then, the output from the network is interpolated and goes through the fully connected CRF to fine tune the result and get the final output.

### Atrous convolution
<p align="center"><img src="https://miro.medium.com/max/1130/1*-r7CL0AkeO72MIDpjRxfog.png"/></p>

Atrous convolution allows us to enlarge the field of view of filters to incorporate larger context. It thus offers an efficient mechanism to control the field-of-view and finds the best trade-off between accurate localization (small field-of-view) and context assimilation (large field-of-view).

### Atrous Spatial Pyramid Pooling (ASPP) [v2]
<p align="center"><img src="https://miro.medium.com/max/1368/1*_8p_KTPr5N0HSeIKV35G_g.png"/></p>

The feature maps pass trhough parallel atrous convolution with different rates and the results are fuse together. ASPP helps to account for different object scales which can improve the accuracy.

### DeepLab v3+
<p align="center"><img src="https://miro.medium.com/max/2000/1*2mYfKnsX1IqCCSItxpXSGA.png"/></p>

- Encoder-Decoder Architecture to recover location/spatial information.
- Modified Aligned Xception as backbone and Atrous Separable Convolution to obtain a faster and stronger network.

## This lab
You will run DeepLab v3 on the Pascal VOC segmentation dataset. Unlike the previous lab, you will no do ablation studies to the architecture, but you'll focus on the training schedule and parameters (lr, input image size, optimizer, etc.). The idea is to analyze how these factors affect the performance of the same architecture. I'll evaluate 2 main results:

- At least 4 experiments based on the DeepLab v3 with ResNet-50 as backbone (the baseline does not count!).
- Comparisson of the method using ResNet-101 as backbone with and without pre-trained weights.

Dataset:
You don't have to download it, the Data is already stored in both machines 
```
Malik: /media/sda1/vision2020_01/Data
BCV001: /media/user_home2/vision2020_01/Data
```

## References
https://towardsdatascience.com/review-deeplabv3-atrous-convolution-semantic-segmentation-6d818bfd1d74
https://towardsdatascience.com/review-deeplabv1-deeplabv2-atrous-convolution-semantic-segmentation-b51c5fbde92d

## Deadline
April 15, 11:59pm
