# Label Evolution Based on Local Contrast Measure for Single-Point Supervised Infrared Small Target Detection

## Abstract

In recent years, the surge in algorithms leveraging convolutional neural networks (CNN) for infrared small target detection has garnered considerable attention. However, the predominant training approach for these CNN models is fully supervised, necessitating extensive pixel-level annotations. Given the intricacies of infrared small target detection, which essentially is an image segmentation task, obtaining detailed pixellevel masks proves exceptionally laborious and time-intensive. In this paper, we present a label evolution framework based on local contrast measure(LELCM) for single-point supervised infrared small target detection. Recognizing that under limited supervision, CNNs tend to first identify fundamental features like contrast, leading to preliminary CNN outputs that highlight more comprehensive targets. Prior to training, we expand single-point initial pseudo-labels that address the class imbalance challenge during CNN training. Moreover, we introduce an advanced label updating strategy that harnesses the variance in confidence levels seen in the CNN outputs to update these pseudo-labels. This dual iterative process ensures both the CNN output and pseudo-labels coalesce over time, resulting in the finalized pixel-level segmentation masks. Moreover, our proposed method seamlessly integrates with any existing fully supervised CNN models. Through rigorous experimentation, we affirm the effectiveness of our method. Our results reveal that our approach not only attains target detection rates (Pd) on par with full supervision models but also achieves 80% of the full supervisory effect in terms of Intersection over Union (IoU).

## Use

创建conda环境

```
conda create -n lelcm python=3.8
conda activate lelcm
```

安装pytorch

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

安装依赖库

```
pip install -r requirements.txt
```

下载数据集NUAA-SIRST数据集放到项目文件夹下面：

```
LELCM/
  └── data/
  	└── SIRST
```

## Train a model

```
python train.py --train_dataset_name SIRST --valid_dataset_name SIRST --masks_update_path  ./data/SIRST/mask_update/ --dataset_dir  ./data/SIRST
```

## Evaluation/Test

```
python test.py --pth_dir ./runs/2025-01-09 11_49_01_train_SIRST_full/1.pth.tar --dataset_dir ./data/SIRST
```

## Reference

```
@article{Yang2024LabelEB,
  title={Label Evolution Based on Local Contrast Measure for Single-Point Supervised Infrared Small-Target Detection},
  author={Dongning Yang and Haopeng Zhang and Ying Li and Zhiguo Jiang},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  volume={62}
}
```

