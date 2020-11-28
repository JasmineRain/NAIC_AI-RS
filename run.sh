#! /bin/bash
echo 'start'
echo '--- start dataset preprocessing ---'
python divide_data.py
echo '--- finish dataset preprocessing ---'
echo '--- start training models ---'
python train.py --model_type="Deeplabv3+" --batch_size=64 --backbone="resnest101"
python train.py --model_type="Deeplabv3+" --batch_size=128
python train.py --model_type="DANet" --batch_size=96 --backbone="resnet101"
python train.py --model_type="HRNet_OCR" --batch_size=96
echo '--- finish training models ---'
echo '--- start compressing models ---'
python compress.py
echo '--- finish compressing models ---'
echo '--- start generating submit.zip ---'
zip -r ./submit/submit.zip ./submit/danet ./submit/deeplab_resnest ./submit/deeplab_resnet ./submit/model_hrnet.pth ./submit/model_define.py ./submit/model_predict.py
echo '--- finish ---'