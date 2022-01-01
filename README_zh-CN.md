# Change Detection Laboratory

使用 [PaddlePaddle](https://www.paddlepaddle.org.cn/) 开发的基于深度学习遥感影像变化检测项目，可作为算法开发、训练框架，也可作为基线测试平台。

*CDLab也拥有 [PyTorch版本](https://github.com/Bobholamovic/CDLab)。目前，本仓库比 PyTorch 版本拥有更丰富的模型实现、数据集接口以及配置文件。*

[English](README.md) | 简体中文

## 依赖库

> opencv-python==4.1.1  
  paddlepaddle-gpu==2.2.0  
  visualdl==2.2.1  
  pyyaml==5.1.2  
  scikit-image==0.15.0  
  scikit-learn==0.21.3  
  scipy==1.3.1  
  tqdm==4.35.0

在 Python 3.7.4，Ubuntu 16.04 环境下测试通过。

## 快速上手

在 `src/constants.py` 文件中将相应常量修改为数据集存放的位置。

### 数据预处理

针对数据集的预处理脚本存放在 `scripts/` 目录下。

### 模型训练

运行如下指令以从头训练一个模型：

```bash
python train.py train --exp_config PATH_TO_CONFIG_FILE
```

在 `configs/` 目录中已经包含了一部分现成的配置文件，可供直接使用。*注意，由于时间和精力的限制，配置文件中提供的超参数可能并没有经过充分调优，您可以通过修改超参数进一步提升模型的效果。*

训练脚本开始执行后，首先会输出配置信息到屏幕，然后会有出现一个提示符，指示您输入一些笔记。这些笔记将被记录到日志文件中。如果在一段时间后，您忘记了本次实验的具体内容，这些笔记可能有助于您回想起来。当然，您也可以选择按下回车键直接跳过。

如果需要从一个检查点（checkpoint）开始继续训练，运行如下指令：

```bash
python train.py train --exp_config PATH_TO_CONFIG_FILE --resume PATH_TO_CHECKPOINT
```

以下是对其它一些常用命令行选项的介绍：

- `anew`: 如果您希望指定的检查点只是用于初始化模型参数，指定此选项。请注意，从一个不兼容的模型中获取部分层的参数对待训练的模型进行初始化也是允许的。
- `save_on`: 如果需要在进行模型评估的同时储存模型的输出结果，指定此选项。项目默认采用基于 epoch 的训练器。在每个 epoch 末尾，训练器也将在验证集上评估模型的性能。
- `log_off`: 指定此选项以禁用日志文件。
- `tb_on`: 指定此选项以启用 tensorboard 日志。
- `debug_on`: 指定此选项以在程序崩溃处自动设置断点，便于进行事后调试。

在训练过程中或训练完成后，您可以在 `exp/DATASET_NAME/weights/` 目录下查看模型权重文件，在 `exp/DATASET_NAME/logs/` 目录下查看日志文件，在 `exp/DATASET_NAME/out/` 目录下查看输出的变化图。

### 模型评估

使用如下指令在测试集上评估已训练好的模型：

```bash
python train.py eval --exp_config PATH_TO_CONFIG_FILE --resume PATH_TO_CHECKPOINT --save_on --subset test
```

本项目也提供在大幅栅格影像上进行滑窗测试的功能。使用如下指令：

```bash
python sw_test.py --exp_config PATH_TO_CONFIG_FILE --resume PATH_TO_CHECKPOINT --ckp_path PATH_TO_CHECKPOINT --t1_dir PATH_TO_T1_DIR --t2_dir PATH_TO_T2_DIR --gt_dir PATH_TO_GT_DIR
```

对于 `src/sw_test.py` 文件，其它一些常用的可选命令行参数包括：
- `--window_size`: 设置滑窗大小。
- `--stride`: 设置滑窗滑动步长。
- `--glob`: 指定在 `t1_dir`、`t2_dir`、和 `gt_dir` 中匹配文件名的 pattern （通配符）。
- `--threshold`: 指定将模型输出的变化概率二值化时使用的阈值。

不过，请注意当前 `src/sw_test.py` 功能有限，并不支持一些较为复杂的自定义预处理和后处理模块。

## 预置模型列表

模型名称 | 对应名称 | 链接
:-:|:-:|:-:
CDNet | `CDNet` | [paper](https://doi.org/10.1007/s10514-018-9734-5)
FC-EF | `Unet` | [paper](https://ieeexplore.ieee.org/abstract/document/8451652)
FC-Siam-conc | `SiamUnet-conc` | [paper](https://ieeexplore.ieee.org/abstract/document/8451652)
FC-Siam-diff | `SiamUnet-diff` | [paper](https://ieeexplore.ieee.org/abstract/document/8451652)
STANet | `STANet` | [paper](https://www.mdpi.com/2072-4292/12/10/1662)
DSIFN | `IFN` | [paper](https://www.sciencedirect.com/science/article/pii/S0924271620301532)
SNUNet | `SNUNet` | [paper](https://ieeexplore.ieee.org/document/9355573)
BIT | `BIT` | [paper](https://ieeexplore.ieee.org/document/9491802)
L-UNet | `LUNet` | [paper](https://ieeexplore.ieee.org/document/9352207)
DSAMNet | `DSAMNet` | [paper](https://ieeexplore.ieee.org/document/9467555)
P2V-CD | `P2V` | 

## 预置数据集列表

数据集名称 | 对应名称 | 链接
:-:|:-:|:-:
SZTAKI AirChange Benchmark set: Szada set | `AC-Szada` | [website](http://web.eee.sztaki.hu/remotesensing/airchange_benchmark.html)
SZTAKI AirChange Benchmark set: Tiszadob set | `AC-Tiszadob` | [website](http://web.eee.sztaki.hu/remotesensing/airchange_benchmark.html)
Onera Satellite Change Detection dataset | `OSCD` | [website](https://rcdaudt.github.io/oscd/)
Synthetic images and real season-varying remote sensing images | `SVCD` | [google drive](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9)
WHU building change detection dataset | `WHU` | [website](http://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)
LEVIR building change detection dataset | `LEVIRCD` | [website](https://justchenhao.github.io/LEVIR/)

## 开发指南

请参见 `docs/` 目录中的内容。

本项目基于[此模板](https://github.com/Bobholamovic/DuduLearnsToCode-Template)开发，针对 PaddlePaddle 框架做了一些迁移和改进。关于这部分的设计思想可参考[此处](https://github.com/Bobholamovic/DuduLearnsToCode-Template/blob/main/take_a_look_if_you_want.md)。

*注意，本仓库仍在开发中，因此尚未添加任何开源证书。*