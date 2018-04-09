MODEL_PATH=./baseline-resnet50_dilated8-ppm_bilinear_deepsup
RESULT_PATH=./
mkdir $MODEL_PATH

wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/baseline-resnet50_dilated8-ppm_bilinear_deepsup/encoder_epoch_20.pth
wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/baseline-resnet50_dilated8-ppm_bilinear_deepsup/decoder_epoch_20.pth
wget -P $RESULT_PATH http://sceneparsing.csail.mit.edu//data/ADEChallengeData2016/images/validation/ADE_val_00000278.jpg

python3 -u test.py \
  --model_path $MODEL_PATH \
  --test_img ./ADE_val_00000278.jpg \
  --arch_encoder resnet50_dilated8 \
  --arch_decoder ppm_bilinear_deepsup \
  --fc_dim 2048 \
  --result $RESULT_PATH
