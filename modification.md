# 代码改动说明

写在开头：我个人水平有限，对于Swin Transformer的代码理解可能没有很透彻，在修改过程中有错误的话欢迎大家及时指正！也希望能借这个机会多多交流~~

## Model Architecture

Swin Transformer V2 相比于V1版本提出的三个改动集中在`swin_transformer.py`的`WindowAttention`模块，分别为：

* 将pre-norm更改为post-norm
* 将点乘attention计算方式更改为cosine attention，并添加用于scaled的参数$\tau$
* 使用continuous relative position bias替代原本直接学习relative position bias的方式，并将线性的相对坐标更改为log-spaced coordinates

### 1. Post-norm

直接修改`swin_transformer.py`的`SwinTransformerBlock`中的代码顺序，向后移动`self.norm1(x)`和`self.norm2(x)`到attention以及mlp操作后，shortcut操作之前，例如：

```python
# x = self.norm2(x)       # Swin-T v1, pre-norm
x = self.mlp(x)         # [bs,H*W,C]
x = self.norm2(x)       # Swin-T v2, post-norm
if self.drop_path is not None:
	x = h + self.drop_path(x)
else:
	x = h + x
```

注意代码中额外添加了`self.norm3`，对应原文的：

> For SwinV2-H and SwinV2-G, we further introduce a layer normalization unit on the main branch every 6 layers.

对于大模型，每隔6个`SwinTransformerBlock`就做一次额外的layer norm。可以通过设置**config**里的`EXTRA_NORM`参数开启。

## 2. Attention计算方式

### 2.1 Dot product attention

原始的swin transformer self-attention计算方式：
$$
\text { Attention }(Q, K, V)=\operatorname{SoftMax}\left(Q K^{T} / \sqrt{d}+B\right) V
$$
Softmax内前面的点乘attention计算对应`WindowAttention`模块如下代码：

```python
qkv = self.qkv(x).chunk(3, axis=-1)
q, k, v = map(self.transpose_multihead, qkv)
q = q * self.scale  # i.e., sqrt(d)
attn = paddle.matmul(q, k, transpose_y=True) 
```

### 2.2 Scaled cosine attention

V2提出的scaled cosine attention计算方式：
$$
\operatorname{Sim}\left(\mathbf{q}_{i}, \mathbf{k}_{j}\right)=\cos \left(\mathbf{q}_{i}, \mathbf{k}_{j}\right) / \tau+B_{i j}
$$
其中$\tau$每个layer的每个head都不同，是可学习参数，且限定最小取值为0.01。

代码更改如下：

首先在`__init__`中定义$\tau$：

```python
# Swin-T v2, Scaled cosine attention
self.tau = paddle.create_parameter(
                shape = [num_heads, window_size[0]*window_size[1], window_size[0]*window_size[1]],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Constant(1))
```

然后在`forward`中：

```python
qkv = self.qkv(x).chunk(3, axis=-1)     # {list:3}
q, k, v = map(self.transpose_multihead, qkv)       # [bs*num_window=1*64,4,49,32] -> [bs*num_window=1*16,8,49,32]-> [bs*num_window=1*4,16,49,32]->[bs*num_window=1*1,32,49,32]

# Swin-T v2, Scaled cosine attention, Eq.(2)
qk = paddle.matmul(q, k, transpose_y=True)        # [bs*num_window=1*64,num_heads=4,49,49] -> [bs*num_window=1*16,num_heads=8,49,49] -> [bs*num_window=1*4,num_heads=16,49,49] -> [bs*num_window=1*1,num_heads=32,49,49]
q2 = paddle.multiply(q, q).sum(-1).sqrt().unsqueeze(3)
k2 = paddle.multiply(k, k).sum(-1).sqrt().unsqueeze(3)
attn = qk/paddle.clip(paddle.matmul(q2, k2, transpose_y=True), min=1e-6)
attn = attn/paddle.clip(self.tau.unsqueeze(0), min=0.01)
```

## 3.Log-Spaced CPB策略

## 3.1 Continuous relative position bias

作者在将训练好的模型迁移到更高分辨率以及更大尺度的window size时，发现直接使用双三次插值的方式去扩充relative position bias会导致性能下降很多，如文章的Tabel1第一行所示。因此V2版本使用了**连续相对位置偏差**的方式，这里我认为连续(continuous)指的是利用一个小网络（比如两层全连接中间带一个ReLu）学习每个相对位置坐标对应的bias，利用小网络的泛化性去适应更大尺寸的window size（这里理解的不是很透彻，还需要再研究一下）。

* 原始模型的代码：

首先在`WindowAttention`的`__init__`方法中定义relative_position_bias_table ，并根据当前block对应的window size计算relative_position_index：

```python
self.relative_position_bias_table = paddle.create_parameter(
            shape=[(2 * window_size[0] -1) * (2 * window_size[1] - 1), num_heads],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))

# relative position index for each token inside window
coords_h = paddle.arange(self.window_size[0])
coords_w = paddle.arange(self.window_size[1])
coords = paddle.stack(paddle.meshgrid([coords_h, coords_w])) # [2, window_h, window_w]
coords_flatten = paddle.flatten(coords, 1) # [2, window_h * window_w]
# 2, window_h * window_w, window_h * window_h
relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1)
# winwod_h*window_w, window_h*window_w, 2
relative_coords = relative_coords.transpose([1, 2, 0])
relative_coords[:, :, 0] += self.window_size[0] - 1
relative_coords[:, :, 1] += self.window_size[1] - 1
relative_coords[:, :, 0] *= 2* self.window_size[1] - 1
# [window_size * window_size, window_size*window_size]
relative_position_index = relative_coords.sum(-1)
self.register_buffer("relative_position_index", relative_position_index)
```

在`forward`过程中，使用如下方式调用：

```python
def get_relative_pos_bias_from_pos_index(self):
    table = self.relative_position_bias_table # N x num_heads
    # index is a tensor
    index = self.relative_position_index.reshape([-1]) # window_h*window_w * window_h*window_w
    # NOTE: paddle does NOT support indexing Tensor by a Tensor
    relative_position_bias = paddle.index_select(x=table, index=index)
    return relative_position_bias
def forward(......):
    ......
    relative_position_bias = relative_position_bias.transpose([2, 0, 1])
    attn = attn + relative_position_bias.unsqueeze(0)
    ......
```

* V2对应代码：

`__init__`中：

```python
## Swin-T v2, small meta network, Eq.(3)
self.cpb = Mlp_Relu(in_features=2,  # delta x, delta y
                    hidden_features=512,  # TODO: hidden dims
                    out_features=self.num_heads,
                    dropout=dropout)
```

还需解决的点在于中间隐藏层维度取多少，这里我设置了512。相对坐标的index计算过程在下面一节会说。

`forward`中：

```python
def get_continuous_relative_position_bias(self):
    # The continuous position bias approach adopts a small meta network on the relative coordinates
    continuous_relative_position_bias = self.cpb(self.log_relative_position_index)
    return continuous_relative_position_bias
def forward(......):
    ......
    ## Swin-T v2
    relative_position_bias = self.get_continuous_relative_position_bias()
    relative_position_bias = relative_position_bias.reshape(
        [self.window_size[0] * self.window_size[1],
         self.window_size[0] * self.window_size[1],
         -1])

    # nH, window_h*window_w, window_h*window_w
    relative_position_bias = relative_position_bias.transpose([2, 0, 1])
    attn = attn + relative_position_bias.unsqueeze(0)
    ......
```

### 3.2 Log-spaced coordinates

此外，作者提到：

> When transferred across largely varied window sizes, there will be a large portion of relative coordinate range requiring extrapolation. 

原先的线性编码计算patch之间的相对位置偏差会导致模型在迁移到更大尺寸的window size时，插值的变化范围也会间隔较大。因此提出：

>we propose to use the log-spaced coordinates instead of the original linear-spaced ones

log-spaced coordinates文章中对应公式4：
$$
\begin{aligned}
&\widehat{\Delta x}=\operatorname{sign}(x) \cdot \log (1+|\Delta x|) \\
&\widehat{\Delta y}=\operatorname{sign}(y) \cdot \log (1+|\Delta y|)
\end{aligned}
$$
但是我感觉$\operatorname{sign}(·)$里面应该是$\Delta x$和$\Delta y$，对应的修改后代码：

```python
# relative position index for each token inside window
coords_h = paddle.arange(self.window_size[0])
coords_w = paddle.arange(self.window_size[1])
coords = paddle.stack(paddle.meshgrid([coords_h, coords_w])) # [2, window_h, window_w]
coords_flatten = paddle.flatten(coords, 1) # [2, window_h * window_w]
# 2, window_h * window_w, window_h * window_h
relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1)
# winwod_h*window_w, window_h*window_w, 2
relative_coords = relative_coords.transpose([1, 2, 0])

## Swin-T v2, log-spaced coordinates, Eq.(4)
log_relative_position_index = paddle.multiply(relative_coords.cast(dtype='float32').sign(),
paddle.log((relative_coords.cast(dtype='float32').abs()+1)))
self.register_buffer("log_relative_position_index", log_relative_position_index)
```













