import numpy as np
import tensorflow as tf

import vgg16
import utils
import Datasets


iteration = 3000
lr = 0.0001
batchsize = 64

print_interval = 10 #每10个输出结果
num_steps = 100     #每100保存模型

#with tf.device('/gpu:0'):
with tf.Session() as sess:
    images = tf.placeholder("float", [batchsize, 224, 224, 3])
    gt = tf.placeholder("float", [batchsize, 25])
    train_mode = tf.placeholder(tf.bool)


    vgg = vgg16.Vgg16('./vgg16.npy')
    vgg.build(images,gt,lr,train_mode)
        
    #添加了该句
    dataset = Datasets.dataset('./train_label.txt')
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for iter in range(iteration):
        batch,labels = dataset.getbatch(batchsize)
        feed_dict = {images: batch,gt:labels,train_mode: True}
        prob,loss,acc,_= sess.run([vgg.prob ,vgg.loss ,vgg.acc,vgg.train], feed_dict=feed_dict)
            

        if (iter+1) % print_interval == 0:
            print("iter:  ",iter+1,",   loss: ",loss, ",   acc: ",acc)

        if (iter + 1) % num_steps == 0 :
            print("saving model ......... ")
            saver.save(sess,"./model/mymodel.ckpt", global_step = iter+1)

