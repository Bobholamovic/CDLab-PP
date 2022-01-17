# Change Detection Laboratory

*Note: This project is currently under development, and no stable version has yet been released.*

Developing and benchmarking deep learning-based remote sensing change detection methods using [PaddlePaddle](https://www.paddlepaddle.org.cn/).

*CDLab also has a [PyTorch version](https://github.com/Bobholamovic/CDLab).*

English | [简体中文](README_zh-CN.md)

## Prerequisites

> opencv-python==4.1.1  
  paddlepaddle-gpu==2.2.0  
  visualdl==2.2.1  
  pyyaml==5.1.2  
  scikit-image==0.15.0  
  scikit-learn==0.21.3  
  scipy==1.3.1  
  tqdm==4.35.0

Tested using Python 3.7.4 on Ubuntu 16.04.

## Get Started

In `src/constants.py`, change the dataset locations to your own.

### Data Preprocessing

In `scripts/` there are preprocessing scripts for several datasets。

### Model Training

To train a model from scratch, use

```bash
python train.py train --exp_config PATH_TO_CONFIG_FILE
```

A few configuration files regarding different datasets and models are provided in the `configs/` folder for ease of use. *Note that the hyperparameters are not elaborately investigated to obtain a fully optimized performance.*

As soon as the program starts and prints out the configurations, there will be a prompt asking you to write some notes. What you write will be recorded into the log file to help you remember what you did, or you can simply skip this step by pressing `[Enter]`.

To resume training from some checkpoint, run the code with the `--resume` option.

```bash
python train.py train --exp_config PATH_TO_CONFIG_FILE --resume PATH_TO_CHECKPOINT
```

Other frequently used commandline options include:

- `--anew`: Add it if the checkpoint is just used to initialize model weights. Note that loading an incompatible checkpoint is supported as a feature, which is useful when you are trying to utilize a well pretrained model for finetuning.
- `--save_on`: By default, an epoch-based trainer is used for training. At the end of each training epoch, the trainer evaluates the model on the validation dataset. If you want to save the model output during the evaluation process, enable this option.
- `--log_off`: Disable logging.
- `--vdl_on`: Enable VisualDL summaries.
- `--debug_on`: Useful when you are debugging your own code. In debugging mode, no checkpoint or model output will be written to disk. In addition, a breakpoint will be set where an unhandled exception occurs, which allows you to locate the causes of the crash or do some cleanup jobs.

During or after the training process, you can check the model weight files in `exp/DATASET_NAME/weights/`, the log files in `exp/DATASET_NAME/logs/`, and the output change maps in `exp/DATASET_NAME/out/`.

### Model Evaluation

To evaluate a model on the test subset, use

```bash
python train.py eval --exp_config PATH_TO_CONFIG_FILE --resume PATH_TO_CHECKPOINT --save_on --subset test
```

This project also provides the funtionality of sliding-window test on large raster images. Use the following command:

```bash
python sw_test.py --exp_config PATH_TO_CONFIG_FILE \
  --resume PATH_TO_CHECKPOINT --ckp_path PATH_TO_CHECKPOINT \
  --t1_dir PATH_TO_T1_DIR --t2_dir PATH_TO_T2_DIR --gt_dir PATH_TO_GT_DIR
```

Other frequently used commandline options of `src/sw_test.py` include:
- `--window_size`: Set the size of the sliding window.
- `--stride`: Set the stride of the sliding window.
- `--glob`: Specify a wildcard pattern to match files in `t1_dir`, `t2_dir`, and `gt_dir`.
- `--threshold`: Set the threshold used to convert the probability map to the change map.

Note however that currently `src/sw_test.py` does not support custom pre-processing or post-processing modules.

## Supported Models

Model | Name | Link
:-:|:-:|:-:
CDNet | `CDNet` | [paper](https://doi.org/10.1007/s10514-018-9734-5)
FC-EF | `UNet` | [paper](https://ieeexplore.ieee.org/abstract/document/8451652)
FC-Siam-conc | `SiamUNet-conc` | [paper](https://ieeexplore.ieee.org/abstract/document/8451652)
FC-Siam-diff | `SiamUNet-diff` | [paper](https://ieeexplore.ieee.org/abstract/document/8451652)
STANet | `STANet` | [paper](https://www.mdpi.com/2072-4292/12/10/1662)
DSIFN | `IFN` | [paper](https://www.sciencedirect.com/science/article/pii/S0924271620301532)
SNUNet | `SNUNet` | [paper](https://ieeexplore.ieee.org/document/9355573)
BIT | `BIT` | [paper](https://ieeexplore.ieee.org/document/9491802)
L-UNet | `LUNet` | [paper](https://ieeexplore.ieee.org/document/9352207)
DSAMNet | `DSAMNet` | [paper](https://ieeexplore.ieee.org/document/9467555)

## Supported Datasets

Dataset | Name | Link
:-:|:-:|:-:
Synthetic images and real season-varying remote sensing images | `SVCD` | [google drive](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9)
WHU building change detection dataset | `WHU` | [website](http://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)
LEVIR building change detection dataset | `LEVIRCD` | [website](https://justchenhao.github.io/LEVIR/)

## Performance on SVCD Dataset

Model | Precision | Recall | F1 | OA
:-:|:-:|:-:|:-:|:-:
CDNet | 93.21 | 87.72 | 90.39 | 97.70
FC-EF | 94.14 | 88.78 | 91.38 | 97.93
FC-Siam-conc | 95.40 | 92.06 | 93.70 | 98.47
FC-Siam-diff | 95.74 | 92.80 | 94.24 | 98.60
STANet | 90.07 | 97.91 | 93.82 | 98.41
DSIFN | 97.31 | 95.32 | 96.31 | 99.10
SNUNet | 97.76 | 97.22 | 97.49 | 99.38
BIT | 96.71 | 95.09 | 95.90 | 99.00
L-UNet | 96.83 | 95.54 | 96.18 | 99.06
DSAMNet | 91.65 | 98.37 | 94.89 | 98.69

## License

This project is released under the [Unlicense](/LICENSE).

## Contributing

Any kind of contributions to improve this repository is welcome.