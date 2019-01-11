import inspect
import os

import numpy as np
import tensorflow as tf
import time

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:
    def __init__(self, vgg16_npy_path=None ,trainable=True, dropout=0.5):
        if vgg16_npy_path is not None:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None
        
        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout
        print("npy file loaded")

    def build(self, rgb, train_mode=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        #第一层3通道变64通道
        self.conv1_1 = self.conv_layer(bgr, 3, 64 , "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64 , 64 , "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1') #/2
        # 112*112*64

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128,  "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        # 56*56*128

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256,  "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256 , "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256 , "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')
        # 28*28*256

        self.conv4_1 = self.conv_layer(self.pool3, 256 , 512 , "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')
        #14*14*512

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')
        #7*7*512

        self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        #在build函数参数中，添加trian_mode，在网络参数中添加dropout
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        elif self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)
        

        self.fc7 = self.fc_layer(self.relu6 , 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
        elif self.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        self.fc8 = self.fc_layer(self.relu7,4096,1000, "fc8")
        self.relu8 = tf.nn.relu(self.fc8)
        if train_mode is not None:
            self.relu8 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu8, self.dropout), lambda: self.relu8)
        elif self.trainable:
            self.relu8 = tf.nn.dropout(self.relu8, self.dropout)

        self.fc9 = self.fc_layer(self.relu8,1000,25, "fc9")

        self.prob = tf.nn.softmax(self.fc9, name="prob")

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            #filt = self.get_conv_filter(name) 
            #修改前，conv_layers函数无参数in_channels, out_channels，且无get_conv_var函数↑，修改后↓添加get_conv_var函数
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            #conv_biases = self.get_bias(name)
            #修改前，修改后删掉该行，因为已经获取了bias参数
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu
    
    #修改vgg16添加
    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        #添加get_conv_var函数，其中内嵌get_var函数，继续添加get_var函数
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")
        
        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases
    
    #修改vgg16添加
    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]           # 如果在预训练模型中有这个参数，则值初始化成预训练模型中的值
        else:
            value = initial_value

        if self.trainable:                              #如果是训练模式，则var是变量，并保存在vardict中，在init中添加这两个变量
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var            

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            #shape = bottom.get_shape().as_list() #shape 直接改为输入了
            #dim = 1
            #for d in shape[1:]:
            #    dim *= d
            #weights = self.get_fc_weight(name)
            #biases = self.get_bias(name)
            #修改前，修改后,并且添加get_fc_var
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])  #将dim改为in_Size

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc
    
    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases
    
    #删掉这个函数
    #def get_conv_filter(self, name):
    #    return tf.constant(self.data_dict[name][0], name="filter")

    #删掉这个函数
    #def get_bias(self, name):
    #    return tf.constant(self.data_dict[name][1], name="biases")
    #删掉这个函数
    #def get_fc_weight(self, name):
    #    return tf.constant(self.data_dict[name][0], name="weights")
