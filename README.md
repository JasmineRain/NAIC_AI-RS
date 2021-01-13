# Repo for NAIC AI + RemoteSensing Image Segmentation
## Team 木已成舟
### Final Rank 14

此Repo包含了本队伍所有代码文件以及内容概述


### 初赛说明
目录说明如下：  
-- dilated：DANet所使用的dilated resnet backbone代码  
-- loss：损失函数代码  
-- models，modules，utils_Deeplab：所有模型代码  
-- traning_log_stage_1：初赛A榜训练日志  
-- *.py：工具方法以及训练文件  
-- ensemble_test：最终使用的推理文件，用于复现A，B榜所有提交结果（路径按需设置）

PS：由于文件大小限制，本队伍所使用的模型已放在百度网盘，链接见[方案说明](https://github.com/JasmineRain/NAIC_AI-RS/blob/master/METHOD.md)

### 复赛说明
复赛要求基本一致，从8类预测提升到了15类

首先给出效果预览，图片取自于网上随意找的遥感图像，实际实现过程中未使用外部数据集
![avatar](./demo/P0063.png)
![avatar](./demo/demo.png)

#### 算法说明
- 本队伍选取采用的模型为DANet, Deeplabv3+, HRNet_OCR
- backbone选取resnet-101, resnest-101
- 预处理为常规的normalization
- 数据增广使用简单的翻转和旋转，如data_loader文件所示
- 优化器选取SGD，学习率调整策略为CosineAnnealingWarmRestarts
- 最终使用4模型集成，包括两种backbone的Deeoplabv3+，resnet101的DANet，以及HRNet_OCR，集成方法为logits求平均
- 由于提交压缩包大小限制，本队伍使用了huffman编码对模型的参数文件进行压缩，确保大小合适
- 由于复赛测试数据大小可变，本队伍对大于1024分辨率的图片进行裁剪，裁剪大小自适应地从1024-512选取，间隔64像素，
以达到裁剪出的边角料最少的效果，避免边角料效果过差。另外，为了消除拼接过于明显的问题，本队伍采用了膨胀预测的方法，
即裁剪时上下左右多裁剪若干像素，取出预测结果的中间部分作为目标区域拼接
- 代码使用Pytorch 1.6.0，复赛训练环境为4x2080Ti+4x1080Ti
- 实现的其它trick包括Sync BN, Label Smooth, Online Edge Label Smooth, Pseudo Label等

#### 复现说明（仅用于云脑平台复现，本项目为完整代码，云脑平台为简化版的复现代码）
说明pdf为Desciption.pdf
直接在云脑项目根目录执行 ```bash run.sh``` 即可全自动复现模型，注意，最终将在./submit/目录生成一个submit.zip，
即为复赛系统要求格式的文件，最终复赛提交的.zip文件[链接](http://212.64.70.65/static/images/submit.zip)
本队伍使用到的预训练模型(均为官方论文发布的模型)：
[resnet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)
[resnest101](https://s3.us-west-1.wasabisys.com/resnest/torch/resnest101-22405ba7.pth)
[hrnetocr](http://212.64.70.65/static/images/hrnet_ocr_cs_trainval_8227_torch11.pth)

#### Reference
[ResNeSt](https://github.com/zhanghang1989/ResNeSt#pretrained-models)
[HRNet_OCR](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR)
