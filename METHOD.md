# 方案说明
## Team 木已成舟

- 数据预处理：仅将mask进行one-hot编码，用于监督模型训练，未进行其他处理，未使用额外数据集以及伪标签；数据增强仅使用了水平翻转、垂直翻转和旋转  
- 环境配置：2080Ti * 4，使用pytorch 1.6.0
- 模型选择：使用DANet和Deeplabv3+，其中两模型均使用了pytorch官方提供的resnet101预训练模型作为backbone；使用SGDR对模型进行训练，
选择多个局部最优点的结果进行集成，最终集成结果即为A，B榜最高分数的结果。未使用其他训练以及测试trick。损失函数使用基本的Cross Entrophy Loss，
未进行类别加权
- 训练日志见training_log_stage_1目录
- 直接在 ```train.py``` 中修改GPU ID，执行 ```python train.py``` 即可复现模型
- 修改数据目录并执行 ```python ensemble_test.py``` 即可得到最终提交的results
- 最终用于推理的模型见百度网盘 [模型](https://pan.baidu.com/s/16hvTw3Vm12vPQivPRrHeqA) 提取码：cevy