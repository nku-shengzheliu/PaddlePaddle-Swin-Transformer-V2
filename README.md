# Swin Transformer V2: Scaling Up Capacity and Resolution, [arxiv](https://arxiv.org/pdf/2111.09883) 

PaddlePaddle training/validation code and pretrained models for **Swin Transformer V2**.

The official pytorch implementation is [here](https://github.com/microsoft/Swin-Transformer).

This implementation is developed by [PaddleViT](https://github.com/BR-IDL/PaddleViT.git).

<p align="center">
<img src="./swin_v2.PNG" alt="drawing" width="50%" height="50%"/>
    <h4 align="center">Comparison of the WindowAttention module between Swin Transformer V1 and Swin Transformer V2</h4>
</p>

## Update 

* Update (2021-11-27): Complete the modification of WindowAttention module according to the original paper
  - [x] post-norm configuration
  - [x] scaled cosine attention
  - [x] log-spaced continuous relative position bias

## Code modification explanation

The code modification explanation is [here](./modification.md)

## Models trained from scratch using PaddleViT

| Model                         | Acc@1 | Acc@5 | #Params | FLOPs  | Image Size | Crop_pct | Interpolation | Link         |
|-------------------------------|-------|-------|---------|--------|------------|----------|---------------|--------------|
| swin_b_224 |       |       | 88.9M   | 15.3G | 224        | 0.9      | Log-CPB | coming soon |

> *The results are evaluated on ImageNet2012 validation set.


## Requirements
- Python>=3.6
- yaml>=0.2.5
- [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)>=2.1.0
- [yacs](https://github.com/rbgirshick/yacs)>=0.1.8

## Data 
ImageNet2012 dataset is used in the following folder structure:
```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── ILSVRC2012_val_00000293.JPEG
│  ├── ILSVRC2012_val_00002138.JPEG
│  ├── ......
```

## Usage
To use the model with pretrained weights, download the `.pdparam` weight file and change related file paths in the following python scripts. The model config files are located in `./configs/`.

For example, assume the downloaded weight file is stored in `./swin_base_patch4_window7_224.pdparams`, to use the `swin_base_patch4_window7_224` model in python:
```python
from config import get_config
from swin import build_swin as build_model
# config files in ./configs/
config = get_config('./configs/swinv2_base_patch4_window7_224.yaml')
# build model
model = build_model(config)
# load pretrained weights, .pdparams is NOT needed
model_state_dict = paddle.load('./swinv2_base_patch4_window7_224')
model.set_dict(model_state_dict)
```

## Evaluation
To evaluate Swin Transformer model performance on ImageNet2012 with a single GPU, run the following script using command line:
```shell
sh run_eval.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
    -cfg='./configs/swinv2_base_patch4_window7_224.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./swinv2_base_patch4_window7_224'
```

<details>
<summary>
Run evaluation using multi-GPUs:
</summary>


```shell
sh run_eval_multi.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
    -cfg='./configs/swinv2_base_patch4_window7_224.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./swinv2_base_patch4_window7_224'
```

</details>


## Training
To train the Swin Transformer model on ImageNet2012 with single GPU, run the following script using command line:
```shell
sh run_train.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_singel_gpu.py \
  -cfg='./configs/swinv2_base_patch4_window7_224.yaml' \
  -dataset='imagenet2012' \
  -batch_size=32 \
  -data_path='/dataset/imagenet' \
```

<details>

<summary>
Run training using multi-GPUs:
</summary>


```shell
sh run_train_multi.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
    -cfg='./configs/swinv2_base_patch4_window7_224.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \
```

</details>

## Reference
```
@article{liu2021swin,
  title={Swin Transformer V2: Scaling Up Capacity and Resolution},
  author={Liu, Ze and Hu, Han and Lin, Yutong and Yao, Zhuliang and Xie, Zhenda and Wei, Yixuan and Ning, Jia and Cao, Yue and Zhang, Zheng and Dong, Li and others},
  journal={arXiv preprint arXiv:2111.09883},
  year={2021}
}
```

