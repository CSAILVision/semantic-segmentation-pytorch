# Semantic Segmentation on MIT ADE20K dataset in PyTorch

This is a PyTorch implementation of semantic segmentation models on MIT ADE20K scene parsing dataset.

ADE20K is the largest open source dataset for semantic segmentation and scene parsing, released by MIT Computer Vision team. Follow the link below to find the repository for our dataset and implementations on Caffe and Torch7:
https://github.com/CSAILVision/sceneparsing

Pretrained models can be found at:
http://sceneparsing.csail.mit.edu/model/

<img src="./teaser/ADE_val_00000045.png" width="900"/>
<img src="./teaser/ADE_val_00001130.png" width="900"/>
From left to right: Test Image, Ground Truth, Predicted Result

## Highlights

### Syncronized Batch Normalization on PyTorch
This module differs from the built-in PyTorch BatchNorm as the mean and standard-deviation are reduced across all devices during training. For example, when one uses `nn.DataParallel` to wrap the network during training, PyTorch's implementation normalizes the tensor on each device using the statistics only on that device, which accelerated the computation and is also easy to implement, but the statistics might be inaccurate. Instead, in this synchronized version, the statistics will be computed over all training samples distributed on multiple devices. 

The importance of synchronized batch normalization in object detection has been recently proved with a an extensive analysis in the paper [MegDet: A Large Mini-Batch Object Detector](https://arxiv.org/abs/1711.07240). And we empirically find that it is also important for segmentation.

The implementation is reasonable due to the following reasons:
- This implementation is in pure-python. No C++ extra extension libs.
- Easy to use.
- It is completely compatible with PyTorch's implementation. Specifically, it uses unbiased variance to update the moving average, and use sqrt(max(var, eps)) instead of sqrt(var + eps).

***To the best knowledge, it is the first pure-python implementation of sync bn on PyTorch, and also the first one completely compatible with PyTorch. It is also efficient, only 20% to 30% slower than un-sync bn.*** We especially thank [Jiayuan Mao](http://vccy.xyz/) for his kind contributions. For more details about the implementation and usage, refer to [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch).

### Dynamic scales of input for training with multiple GPUs 
Different from image classification task, where the input images are resized to a certain scale such as 224x224 or 229x229, it is better for semantic segmentation and object detection networks that the aspect ratios of input images remain what they original are. It is not trivial on PyTorch, because the dataloader loads a pile of images first, then the `nn.DataParallel` module automatically splits them to multiple GPUs. That is, the images are concatenated first, then distributed. The concatenation, of course, requires the images to be of the same size.

Alternatively, we re-implement the `DataParallel` module, and make it support distributing data to multiple GPUs in python dict. You are free to put a lot of stuff in a dict. At the same time, the dataloader also operates differently. *Now the batch size of a dataloader always equals to the number of GPUs*, cause each elements will be sent to a GPU later. In this way, for example, if you'd like to put 2 images on each GPU, its the dataloader's job. We also need to make it compatible with the multi-processing. Note that the file index for the multi-processing dataloader is stored on the master process, which is in contradict to our goal that each worker maintains its own file list. So we use a trick that although the master process still gives dataloader an index for `__getitem__` function, we just ignore such request and send a random batch dict. Also, *the multiple workers forked by the dataloader all have the same seed*, you will find that multiple workers will yield exactly the same data, if we use the above-mentioned trick directly. Therefore, we add one line of code which sets the defaut seed for `numpy.random` before activating multiple worker in dataloader.


## Supported models
We split our models into encoder and decoder, where encoders are usually modified directly from classification networks, and decoders consist of final convolutions and upsampling.

Encoder:
- resnet50_dilated16
- resnet50_dilated8

***Coming soon***:
- resnet101_dilated16
- resnet101_dilated8

(resnetXX_dilatedYY: customized resnetXX with dilated convolutions, output feature map is 1/YY of input size.)

Decoder:
- c1_bilinear (1 conv + bilinear upsample)
- c1_bilinear_deepsup (c1_blinear + deep supervision trick)
- psp_bilinear (pyramid pooling + bilinear upsample, see PSPNet paper for details)
- psp_bilinear_deepsup (psp_bilinear + deep supervision trick)

***Coming soon***:
- UPerNet based on Feature Pyramid Network (FPN) and Pyramid Pooling Module (PPM), with down-sampling rate of 4, 8 and 16. It doesn't need dilated convolution, a operator that is time-and-memory consuming. *Without bells and whistles*, it is comparable or even better compared with PSPNet, while requires much shorter training time and less GPU memory.


## Performance:
IMPORTANT: We use our self-trained base model on ImageNet. The model takes the input in BGR form (consistent with opencv) instead of RGB form as used by default implementation of PyTorch. The base model will be automatically downloaded when needed.

### Main Results
|               | MS Test | Mean IoU   | Accuracy | Overall | Training Time | 
|---------------|:----:|:----:|:-----:|:--------------:|:-------:|
|ResNet-50_dilated8 + c1_bilinear_deepsup | No | - | - | - | 27.5 hours | 
|ResNet-50_dilated8 + psp_bilinear_deepsup | No | 40.60 | 79.66 | 60.13 | 33.4 hours |
|Same as above | Yes | 41.31 | 80.14 | 60.73 | Same as above | 
|ResNet-101_dilated8 + c1_bilinear_deepsup | No | - | - | - | hours | 
|ResNet-101_dilated8 + psp_bilinear_deepsup | No | - | - | - | hours |
|Same as above | Yes | - | - | - | Same as above | 
|UPerNet-50 (coming soon!) | No | - | - | - | hours | 
|UPerNet-50 (coming soon!) | Yes| - | - | - | Same as above | 


## Environment
The code is developed under the following configurations.
- Hardware: 2-8 Pascal Titan X GPUs (change ```[--num_gpus NUM_GPUS]``` accordingly)
- Software: Ubuntu 16.04.3 LTS, CUDA 8.0, ***Python3.5***, ***PyTorch 0.4.0***

*Warning:* We don't support the outdated Python 2 anymore. PyTorch 0.4.0 or higher is required to run the codes.

## Training
1. Download the ADE20K scene parsing dataset:
```bash
chmod +x download_ADE20K.sh
./download_ADE20K.sh
```
2. Train a network (default: resnet50_dilated8_deepsup). During training, checkpoints will be saved in folder ```ckpt```.
```bash
python3 train.py
```

3. Input arguments: (see full input arguments via ```python3 train.py -h ```)
```bash
usage: train.py [-h] [--id ID] [--arch_encoder ARCH_ENCODER]
                [--arch_decoder ARCH_DECODER]
                [--weights_encoder WEIGHTS_ENCODER]
                [--weights_decoder WEIGHTS_DECODER] [--fc_dim FC_DIM]
                [--list_train LIST_TRAIN] [--list_val LIST_VAL]
                [--root_dataset ROOT_DATASET] [--num_gpus NUM_GPUS]
                [--batch_size_per_gpu BATCH_SIZE_PER_GPU]
                [--num_epoch NUM_EPOCH] [--epoch_iters EPOCH_ITERS]
                [--optim OPTIM] [--lr_encoder LR_ENCODER]
                [--lr_decoder LR_DECODER] [--lr_pow LR_POW] [--beta1 BETA1]
                [--weight_decay WEIGHT_DECAY]
                [--deep_sup_scale DEEP_SUP_SCALE] [--fix_bn FIX_BN]
                [--num_class NUM_CLASS] [--workers WORKERS]
                [--imgSize IMGSIZE] [--imgMaxSize IMGMAXSIZE]
                [--padding_constant PADDING_CONSTANT]
                [--segm_downsampling_rate SEGM_DOWNSAMPLING_RATE]
                [--random_flip RANDOM_FLIP] [--seed SEED] [--ckpt CKPT]
                [--disp_iter DISP_ITER]
```


## Evaluation
1. Evaluate a trained network on the validation set:
```bash
python3 eval.py --id MODEL_ID
```

2. Input arguments: (see full input arguments via ```python3 eval.py -h ```)
```bash
usage: eval.py [-h] --id ID [--suffix SUFFIX] [--arch_encoder ARCH_ENCODER]
               [--arch_decoder ARCH_DECODER] [--fc_dim FC_DIM]
               [--list_val LIST_VAL] [--root_dataset ROOT_DATASET]
               [--num_val NUM_VAL] [--num_class NUM_CLASS]
               [--batch_size BATCH_SIZE] [--imgSize IMGSIZE]
               [--imgMaxSize IMGMAXSIZE] [--padding_constant PADDING_CONSTANT]
               [--segm_downsampling_rate SEGM_DOWNSAMPLING_RATE] [--ckpt CKPT]
               [--visualize VISUALIZE] [--result RESULT] [--gpu_id GPU_ID]
```


## Reference

If you find the code or pre-trained models useful, please cite the following paper:

Scene Parsing through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso and A. Torralba. Computer Vision and Pattern Recognition (CVPR), 2017. (http://people.csail.mit.edu/bzhou/publication/scene-parse-camera-ready.pdf)

    @inproceedings{zhou2017scene,
        title={Scene Parsing through ADE20K Dataset},
        author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        year={2017}
    }
    
Unified Perceptual Parsing for Scene Understanding. T. Xiao, Y. Liu, B. Zhou, Y. Jiang, and J. Sun. arXiv preprint

    @article{xiao2018unified,
      title={Unified Perceptual Parsing for Scene Understanding},
      author={Xiao, Tete and Liu, Yingcheng and Zhou, Bolei and Jiang, Yuning and Sun, Jian},
      journal={arXiv preprint},
      year={2018}
    }
    
Semantic Understanding of Scenes through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso and A. Torralba. arXiv:1608.05442. (https://arxiv.org/pdf/1608.05442.pdf)

    @article{zhou2016semantic,
      title={Semantic understanding of scenes through the ade20k dataset},
      author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
      journal={arXiv preprint arXiv:1608.05442},
      year={2016}
    }
