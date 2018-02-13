MODEL_PATH=./baseline-resnet34_dilated8-psp_bilinear
mkdir $MODEL_PATH

wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/baseline-resnet34_dilated8-psp_bilinear/encoder_best.pth
wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/baseline-resnet34_dilated8-psp_bilinear/decoder_best.pth

wget http://sceneparsing.csail.mit.edu//data/ADEChallengeData2016/images/validation/ADE_val_00000278.jpg

python -u test.py \
  --model_path $MODEL_PATH \
  --test_img ./ADE_val_00000278.jpg \
  --arch_encoder resnet34_dilated8 \
  --arch_decoder psp_bilinear \
  --fc_dim 512 \
  --result .
