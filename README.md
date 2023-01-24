# tensorflow-cats-and-dogs-classification
The most classic entry-level deep-learning task.This is the Python implementation of this task(using tensorflow2).

# dataset
The dataset we use comes from the open source communication platform provided by Tianchi Lab (part of Alibaba Group), and the uploader is named aliyun2514290857.
You can download this dataset at [https://tianchi.aliyun.com/dataset/129528](https://tianchi.aliyun.com/dataset/129027)
After you download the dataset, make a folder named "data", and unzip zhe dataset in it.
After that, you can see two floders in it: train, validation

# configuration
python==3.8.13

tensorflow==2.6.0

if you want use gpu for training, install tensorflow-gpu==2.6.0, too.

model import layer: 64*64*3

# train
``python train.py``

# solution
as device NVIDIA GeForce RTX 3060 Ti, after 20 epochs of training, the model can gives out accuracy about 85%

## enjoy
