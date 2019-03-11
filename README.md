# mxnet-fast-neural-style

A mxnet implementation of fast style transfer, inspired by:
- [https://github.com/lengstrom/fast-style-transfer](https://github.com/lengstrom/fast-style-transfer)
- [https://github.com/zhaw/neural_style](https://github.com/zhaw/neural_style)
- [https://github.com/dmlc/mxnet/tree/master/example/neural-style](https://github.com/dmlc/mxnet/tree/master/example/neural-style)

releated papers:
- Johnson's [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://cs.stanford.edu/people/jcjohns/eccv16/)
- Ulyanov's [Instance Normalization](https://arxiv.org/abs/1607.08022)

## example

We added styles from various paintings to a photo of Chicago. Click on thumbnails to see full applied style images.

<div align = 'center'>
<a href = 'examples/style/wave.jpg'>
<img src = 'https://raw.githubusercontent.com/SineYuan/mxnet-fast-neural-style/master/images/sunrise.jpg' width = '200px'></a>
<img src = 'https://raw.githubusercontent.com/SineYuan/mxnet-fast-neural-style/master/images/The_Great_Wave_off_Kanagawa.jpg' height = '160px'>

<br>
<a href = 'examples/style/wave.jpg'>
<img src = 'https://raw.githubusercontent.com/SineYuan/mxnet-fast-neural-style/master/images/guangzhou_tower.jpg' height = '200px'>
<img src = 'https://raw.githubusercontent.com/SineYuan/mxnet-fast-neural-style/master/images/guangzhou_tower-sunrise.jpg' width = '200px'>
<a href = 'https://raw.githubusercontent.com/SineYuan/mxnet-fast-neural-style/master/images/guangzhou_tower-wave.jpg'>
<img src = 'https://raw.githubusercontent.com/SineYuan/mxnet-fast-neural-style/master/images/guangzhou_tower-wave.jpg' width = '200px'></a>

<br>
<a href = 'examples/style/wave.jpg'>
<img src = 'https://raw.githubusercontent.com/SineYuan/mxnet-fast-neural-style/master/images/river_night.jpg' width = '200px'>
<img src = 'https://raw.githubusercontent.com/SineYuan/mxnet-fast-neural-style/master/images/river_night-sunrise.jpg' height = '200px'>
<a href = 'https://raw.githubusercontent.com/SineYuan/mxnet-fast-neural-style/master/images/river_night-wave.jpg'>
<img src = 'https://raw.githubusercontent.com/SineYuan/mxnet-fast-neural-style/master/images/river_night-wave.jpg' height = '200px'></a>

<br>
<a href = 'examples/style/wave.jpg'>
<img src = 'https://raw.githubusercontent.com/SineYuan/mxnet-fast-neural-style/master/images/cloud_sea.jpg' width = '200px'>
<img src = 'https://raw.githubusercontent.com/SineYuan/mxnet-fast-neural-style/master/images/cloud_sea-sunrise.jpg' height = '200px'>
<a href = 'https://raw.githubusercontent.com/SineYuan/mxnet-fast-neural-style/master/images/cloud_sea-wave.jpg'>
<img src = 'https://raw.githubusercontent.com/SineYuan/mxnet-fast-neural-style/master/images/cloud_sea-wave.jpg' height = '200px'></a>
</div>

some pretrained model you can find in `checkpoints` directory.

## Prerequisites

1. [MXNet](https://github.com/dmlc/mxnet/)
2. Pretrained VGG19 params file : [vgg19.params](https://github.com/dmlc/web-data/raw/master/mxnet/neural-style/model/vgg19.params)
3. Training data if you want to train your own models. The example models is trained on MSCOCO [[Download Link](http://msvocds.blob.core.windows.net/coco2014/train2014.zip)](about 12GB)

## Usage

### Training Style Transfer Networks

```
python train.py --style-image path/to/style/img.jpg \
  --checkpoint-dir path/to/save/checkpoint \
  --vgg-path path/to/vgg19.params \
  --content-weight 1e2 \
  --style-weight  1e1 \
  --epochs 2 \
  --batch-size 20 \
  --gpu 0
```

for more detail see the help information of `train.py`

```
python train.py -h
```

### Transform images

```
python transform.py --in-path path/to/input/img.jpg \
  --out-path path/dir/to/output \
  --checkpoint path/to/checkpoint/params \
  --resize 720 480 \
  --gpu 0
```

for more detail see the help information of `transform.py`

```
python transform.py -h
```
