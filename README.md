# Tensorflow VGG16 for my own data

This is a Tensorflow implemention of VGG 16 to train my own classification network on my own data

## Usage
Use this to build the VGG object in test_vgg16.py

The `images` is a tensor with shape `[None, 224, 224, 3]`. I wrote datasets.py to implement the read imges and lable for train.
>Trick: the tensor can be a placeholder, a variable or even a constant.

All the VGG layers (tensors) can then be accessed using the vgg object. For example, `vgg.conv1_1`, `vgg.conv1_2`, `vgg.pool5`, `vgg.prob`, ...

`test_vgg16.py` and `test_vgg19.py` contain the sample usage.

## 修改vgg训练代码

要完成微调，首先得写一下loss，写loss前，首先得把数据的标签one-hot类型的给写了，因为我的只有25类，而且数据文件夹的格式是这样的:

```
-data
|
|---0
|	|
|	-----xxx.bmp
|	-----xxx.bmp
|---1
.....
```

所以写了一段代码将文件的路径和label数据都存入txt，需要的时候直接读取，并转换成onehot数据即可。在module1.py中，主要代码：

```
"""
将图像路径和label写入txt文件
author:monkey
"""
idx = -1
for dirpath,dirs,files in os.walk(datapath):
    for file in files:
        label = labels[idx]
        writepath = os.path.join(dirpath,file)
        filetxt.write(writepath)
        filetxt.write(' ')
        filetxt.write(label)
        filetxt.write('\n')
    idx += 1
```

写标签文件之后，写一个dataset的读取batch和label的代码，实现随机读取图像为batch的功能，在dataset.py中。

