#!/bin/bash

# Image and model names
TEST_IMG=ADE_val_00001519.jpg
MODEL_PATH=baseline-resnet50dilated-ppm_deepsup
RESULT_PATH=./

ENCODER=$MODEL_PATH/encoder_epoch_20.pth
DECODER=$MODEL_PATH/decoder_epoch_20.pth

# Download model weights and image
if [ ! -e $MODEL_PATH ]; then
  mkdir $MODEL_PATH
fi
if [ ! -e $ENCODER ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER
fi
if [ ! -e $DECODER ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER
fi
if [ ! -e $TEST_IMG ]; then
  wget -P $RESULT_PATH http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016/images/validation/$TEST_IMG
fi

# Inference
python3 -u test.py \
  --model_path $MODEL_PATH \
  --test_imgs $TEST_IMG \
  --arch_encoder resnet50dilated \
  --arch_decoder ppm_deepsup \
  --fc_dim 2048 \
  --result $RESULT_PATH
