# PyTorch SSD
PyTorch implementation of [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325).

## Results
### PASCAL VOC
* Training: 07+12 trainval
* Evaluation: 07 test

| Model                | Input size | mAP<sub>0.5</sub> |
|----------------------|:----------:|:-----------------:|
| SSD300               | 300        |                   |
| SSD512               | 512        |                   |
| MobileNetV2 SSDLite  | 320        |                   |

### COCO
* Training: train2017
* Evaluation: val2017

| Model                | Input size | mAP<sub>0.5:0.95</sub> |
|----------------------|:----------:|:----------------------:|
| SSD300               | 300        |                        |
| SSD512               | 512        |                        |
| MobileNetV2 SSDLite  | 320        |                        |

## Requirements
* Python ≥ 3.6
* Install libraries: `pip install -r requirements.txt`

## Data preparation
### PASCAL VOC
```bash
cd datasets/voc/

wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
wget http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar

python prepare.py --root VOCdevkit/
```
### COCO
```bash
cd datasets/coco/

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

python prepare.py --root .
```

## Training


## Evaluation

