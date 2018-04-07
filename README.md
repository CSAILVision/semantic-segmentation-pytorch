# Semantic Segmentation on MIT ADE20K dataset in PyTorch

This is a PyTorch implementation of semantic segmentation models on MIT ADE20K scene parsing dataset.

ADE20K is the largest open source dataset for semantic segmentation and scene parsing, released by MIT Computer Vision team. Follow the link below to find the repository for our dataset and implementations on Caffe and Torch7:
https://github.com/CSAILVision/sceneparsing

Pretrained models can be found at:
http://sceneparsing.csail.mit.edu/model/

<img src="./teaser/ADE_val_00000278.png" width="900"/>
<img src="./teaser/ADE_val_00001519.png" width="900"/>
[From left to right: Test Image, Ground Truth, Predicted Result]

## Highlights [NEW!]

### Syncronized Batch Normalization on PyTorch
This module differs from the built-in PyTorch BatchNorm as the mean and standard-deviation are reduced across all devices during training. The importance of synchronized batch normalization in object detection has been recently proved with a an extensive analysis in the paper [MegDet: A Large Mini-Batch Object Detector](https://arxiv.org/abs/1711.07240), and we empirically find that it is also important for segmentation.

The implementation is reasonable due to the following reasons:
- This implementation is in pure-python. No C++ extra extension libs.
- Easy to use.
- It is completely compatible with PyTorch's implementation. Specifically, it uses unbiased variance to update the moving average, and use sqrt(max(var, eps)) instead of sqrt(var + eps).

***To the best knowledge, it is the first pure-python implementation of sync bn on PyTorch, and also the first one completely compatible with PyTorch. It is also efficient, only 20% to 30% slower than un-sync bn.*** We especially thank [Jiayuan Mao](http://vccy.xyz/) for his kind contributions. For more details about the implementation and usage, refer to [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch).

### Dynamic scales of input for training with multiple GPUs 
Different from image classification task, where the input images are resized to a fixed scale such as 224x224, it is better to keep original aspect ratios of input images for semantic segmentation and object detection networks.

So we re-implement the `DataParallel` module, and make it support distributing data to multiple GPUs in python dict. At the same time, the dataloader also operates differently. *Now the batch size of a dataloader always equals to the number of GPUs*, each element will be sent to a GPU. It is also compatible with multi-processing. Note that the file index for the multi-processing dataloader is stored on the master process, which is in contradict to our goal that each worker maintains its own file list. So we use a trick that although the master process still gives dataloader an index for `__getitem__` function, we just ignore such request and send a random batch dict. Also, *the multiple workers forked by the dataloader all have the same seed*, you will find that multiple workers will yield exactly the same data, if we use the above-mentioned trick directly. Therefore, we add one line of code which sets the defaut seed for `numpy.random` before activating multiple worker in dataloader.


## Supported models
We split our models into encoder and decoder, where encoders are usually modified directly from classification networks, and decoders consist of final convolutions and upsampling.

Encoder: (resnetXX_dilatedYY: customized resnetXX with dilated convolutions, output feature map is 1/YY of input size.)
- resnet34_dilated16, resnet34_dilated8
- resnet50_dilated16, resnet50_dilated8

***Coming soon***:
- resnet101_dilated16, resnet101_dilated8

Decoder:
- c1_bilinear (1 conv + bilinear upsample)
- c1_bilinear_deepsup (c1_blinear + deep supervision trick)
- ppm_bilinear (pyramid pooling + bilinear upsample, see PSPNet paper for details)
- ppm_bilinear_deepsup (psp_bilinear + deep supervision trick)

***Coming soon***:
- UPerNet based on Feature Pyramid Network (FPN) and Pyramid Pooling Module (PPM), with down-sampling rate of 4, 8 and 16. It doesn't need dilated convolution, a operator that is time-and-memory consuming. *Without bells and whistles*, it is comparable or even better compared with PSPNet, while requires much shorter training time and less GPU memory.


## Performance:
IMPORTANT: We use our self-trained base model on ImageNet. The model takes the input in BGR form (consistent with opencv) instead of RGB form as used by default implementation of PyTorch. The base model will be automatically downloaded when needed.

<table><tbody>
    <th valign="bottom">Architecture</th>
    <th valign="bottom">MS Test</th>
    <th valign="bottom">Mean IoU</th>
    <th valign="bottom">Pixel Accuracy</th>
    <th valign="bottom">Overall Score</th>
    <th valign="bottom">Training Time</th>
    <tr>
        <td>ResNet-50_dilated8 + c1_bilinear_deepsup</td>
        <td>No</td><td>34.88</td><td>76.54</td><td>55.71</td>
        <td>27.5 hours</td>
    </tr>
    <tr>
        <td rowspan="2">ResNet-50_dilated8 + psp_bilinear_deepsup</td>
        <td>No</td><td>41.26</td><td>79.73</td><td>60.50</td>
        <td rowspan="2">33.4 hours</td>
    </tr>
    <tr>
        <td>Yes</td><td>42.04</td><td>80.23</td><td>61.14</td>
    </tr>
    <tr>
        <td>ResNet-101_dilated8 + c1_bilinear_deepsup</td>
        <td>-</td><td>-</td><td>-</td><td>-</td>
        <td>- hours</td>
    </tr>
    <tr>
        <td>ResNet-101_dilated8 + ppm_bilinear_deepsup</td>
        <td>-</td><td>-</td><td>-</td><td>-</td>
        <td>- hours</td>
    </tr>
    <tr>
        <td>UPerNet-50 (coming soon!)</td>
        <td>-</td><td>-</td><td>-</td><td>-</td>
        <td>- hours</td>
    </tr>
    <tr>
        <td>UPerNet-101 (coming soon!)</td>
        <td>-</td><td>-</td><td>-</td><td>-</td>
        <td>- hours</td>
    </tr>
</tbody></table>

The speed is benchmarked on a server with 8 NVIDIA Pascal Titan Xp GPUs (12GB GPU memory), except for ResNet-101_dilated8, which is benchmarked on a server with 8 NVIDIA Tesla P40 GPUS (22GB GPU memory), because of the insufficient memory issue when using dilated conv on a very deep network.

## Environment
The code is developed under the following configurations.
- Hardware: 2-8 GPUs (with at least 12G GPU memories) (change ```[--num_gpus NUM_GPUS]``` accordingly)
- Software: Ubuntu 16.04.3 LTS, CUDA 8.0, ***Python3.5***, ***PyTorch 0.4.0***

*Warning:* We don't support the outdated Python 2 anymore. PyTorch 0.4.0 or higher is required to run the codes.

## Training
1. Download the ADE20K scene parsing dataset:
```bash
chmod +x download_ADE20K.sh
./download_ADE20K.sh
```
2. Train a network (default: ResNet-50_dilated8 + psp_bilinear_deepsup). During training, checkpoints will be saved in folder ```ckpt```.
```bash
python3 train.py --num_gpus NUM_GPUS
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
1. Evaluate a trained network on the validation set. Add ```--visualize``` option to output visualizations shown in teaser.
```bash
python3 eval.py --id MODEL_ID --suffix SUFFIX
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
               [--visualize] [--result RESULT] [--gpu_id GPU_ID]
```


## Reference

If you find the code or pre-trained models useful, please cite the following papers:

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
