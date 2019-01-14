import numpy as np
import tensorflow as tf

import vgg16
import utils
import Datasets


iteration = 50
#lr = 0.0001
batchsize = 64

#print_interval = 10 #每10个输出结果
#num_steps = 1000     #每100保存模型

#with tf.device('/gpu:0'):

saver = tf.train.Saver()
with tf.Session() as sess:
    images = tf.placeholder("float", [batchsize, 224, 224, 3])
    gt = tf.placeholder("float", [batchsize, 25])
    
    #添加了该句
    dataset = Datasets.dataset('./test_label.txt')

    saver.restore(sess,'./model/mymodel.ckpt')
    prob_net = tf.get_default_graph().get_tensor_by_name('prob:0')
    loss_net = tf.get_default_graph().get_tensor_by_name('loss:0')
    acc_net = tf.get_default_graph().get_tensor_by_name('acc:0')

    for iter in range(iteration):
        batch,labels = dataset.getbatch(batchsize)
        feed_dict = {images: batch,gt:labels,train_mode: True}
        prob1,loss,acc= sess.run([prob_net,loss_net ,acc_net], feed_dict=feed_dict)
            
     
        print("iter:  ",iter+1,",   loss: ",loss, ",   acc: ",acc)
        utils.print_prob(prob1,'./synset.txt')

