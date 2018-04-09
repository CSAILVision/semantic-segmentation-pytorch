TEST_IMG=ADE_val_00001519.jpg
MODEL_PATH=baseline-resnet50_dilated8-ppm_bilinear_deepsup
RESULT_PATH=./

ENCODER=$MODEL_PATH/encoder_epoch_20.pth
DECODER=$MODEL_PATH/decoder_epoch_20.pth

if [ ! -e $ENCODER ]; then
  mkdir $MODEL_PATH
fi
if [ ! -e $ENCODER ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER
fi
if [ ! -e $DECODER ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER
fi
if [ ! -e $TEST_IMG ]; then
  wget -P $RESULT_PATH http://sceneparsing.csail.mit.edu//data/ADEChallengeData2016/images/validation/$TEST_IMG
fi

python3 -u test.py \
  --model_path $MODEL_PATH \
  --test_img $TEST_IMG \
  --arch_encoder resnet50_dilated8 \
  --arch_decoder ppm_bilinear_deepsup \
  --fc_dim 2048 \
  --result $RESULT_PATH
