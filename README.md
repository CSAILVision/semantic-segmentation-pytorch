# semantic-segmentation-pytorch
Pytorch implementation for semantic segmentation/scene parsing on MIT ADE20K dataset.

Supported models:
- VGG16-dilated8
- VGG19-dilated8
- ResNet34-dilated16
- ResNet34-dilated8
- ResNet50-dilated16
- ResNet50-dilated8


## Training
1. Download the ADE20K scene parsing dataset:
```bash
chmod +x download_ADE20K.sh
./download_ADE20K.sh
```
2. Train a network: (default: ResNet34-dilated8)
```bash
python train.py
```

3. Input arguments: see full input arguments ```python train.py -h ```
```bash
usage: train.py [-h] [--id ID] [--arch_encoder ARCH_ENCODER]
                [--arch_decoder ARCH_DECODER]
                [--weights_encoder WEIGHTS_ENCODER]
                [--weights_decoder WEIGHTS_DECODER] [--fc_dim FC_DIM]
                [--list_train LIST_TRAIN] [--list_val LIST_VAL]
                [--root_img ROOT_IMG] [--root_seg ROOT_SEG]
                [--num_gpus NUM_GPUS]
                [--batch_size_per_gpu BATCH_SIZE_PER_GPU]
                [--num_epoch NUM_EPOCH] [--optim OPTIM]
                [--lr_encoder LR_ENCODER] [--lr_decoder LR_DECODER]
                [--lr_step LR_STEP] [--beta1 BETA1]
                [--weight_decay WEIGHT_DECAY] [--fix_bn FIX_BN]
                [--num_val NUM_VAL] [--workers WORKERS] [--imgSize IMGSIZE]
                [--segSize SEGSIZE] [--segDepth SEGDEPTH] [--flip FLIP]
                [--seed SEED] [--ckpt CKPT] [--vis VIS]
                [--disp_iter DISP_ITER] [--eval_epoch EVAL_EPOCH]
                [--ckpt_epoch CKPT_EPOCH]
```


## Evaluation
1. Evaluate a trained network:
```bash
python eval.py --id MODEL_ID
```
2. Input arguments: see full input arguments ```python eval.py -h ```
```bash
usage: eval.py [-h] --id ID [--suffix SUFFIX] [--arch_encoder ARCH_ENCODER]
               [--arch_decoder ARCH_DECODER] [--fc_dim FC_DIM]
               [--list_val LIST_VAL] [--root_img ROOT_IMG]
               [--root_seg ROOT_SEG] [--num_val NUM_VAL]
               [--batch_size BATCH_SIZE] [--imgSize IMGSIZE]
               [--segSize SEGSIZE] [--segDepth SEGDEPTH] [--ckpt CKPT]
               [--visualize VISUALIZE] [--result RESULT]
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
    
Semantic Understanding of Scenes through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso and A. Torralba. arXiv:1608.05442. (https://arxiv.org/pdf/1608.05442.pdf)

    @article{zhou2016semantic,
      title={Semantic understanding of scenes through the ade20k dataset},
      author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
      journal={arXiv preprint arXiv:1608.05442},
      year={2016}
    }
