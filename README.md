# Image-Adaptive-3DLUT
Learning Image-adaptive 3D Lookup Tables for High Performance Photo Enhancement in Real-time

## Downloads
### [Paper](https://www4.comp.polyu.edu.hk/~cslzhang/paper/PAMI_LUT.pdf), [Supplementary](https://www4.comp.polyu.edu.hk/~cslzhang/paper/Supplement_LUT.pdf), [Dataset](https://connectpolyu-my.sharepoint.com/:f:/g/personal/16901447r_connect_polyu_hk/EqNGuQUKZe9Cv3fPG08OmGEBbHMUXey2aU03E21dFZwJyg?e=QNCMMZ), [PCT patent]
The whole datasets used in the paper are over 300G. Here I only provide the FiveK dataset resized into 480p resolution (including 8-bit sRGB, 16-bit XYZ inputs and 8-bit sRGB targets). To obtain the images of original resolution, it is recommended to transform from the original [FiveK](https://data.csail.mit.edu/graphics/fivek/) dataset. 

## Abstract
Recent years have witnessed the increasing popularity of learning based methods to enhance the color and tone of photos. However, many existing photo enhancement methods either deliver unsatisfactory results or consume too much computational and memory resources, hindering their application to high-resolution images (usually with more than 12 megapixels) in practice. In this paper, we learn image-adaptive 3-dimensional lookup tables (3D LUTs) to achieve fast and robust photo enhancement. 3D LUTs are widely used for manipulating color and tone of photos, but they are usually manually tuned and ﬁxed in camera imaging pipeline or photo editing tools. We, for the ﬁrst time to our best knowledge, propose to learn 3D LUTs from annotated data using pairwise or unpaired learning. More importantly, our learned 3D LUT is image-adaptive for ﬂexible photo enhancement. We learn multiple basis 3D LUTs and a small convolutional neural network (CNN) simultaneously in an end-to-end manner. The small CNN works on the down-sampled version of the input image to predict content-dependent weights to fuse the multiple basis 3D LUTs into an image-adaptive one, which is employed to transform the color and tone of source images efﬁciently. Our model contains less than 600K parameters and takes less than 2 ms to process an image of 4K resolution using one Titan RTX GPU. While being highly efﬁcient, our model also outperforms the state-of-the-art photo enhancement methods by a large margin in terms of PSNR, SSIM and a color difference metric on two publically available benchmark datasets. 

## Framework
<img src="figures/framework2.png" width="1024px"/> 

## Usage

### Requirements
Python3, requirements.txt

### Build
cd trilinear
sh make.sh
### Training

### Testing

## Citation
```
@article{zeng2020lut,
  title={Learning Image-adaptive 3D Lookup Tables for High Performance Photo Enhancement in Real-time},
  author={Zeng, Hui and Cai, Jianrui and Li, Lida and Cao, Zisheng and Zhang, Lei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={},
  number={},
  pages={},
  year={2020},
  publisher={IEEE}
}
```
