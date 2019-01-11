import numpy as np
import tensorflow as tf

import vgg16
import utils

img1 = utils.load_image("./test_data/tiger.jpeg")
img2 = utils.load_image("./test_data/puzzle.jpeg")

batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2), 0)

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/cpu:0'):
    with tf.Session() as sess:
        images = tf.placeholder("float", [2, 224, 224, 3])
        train_mode = tf.placeholder(tf.bool)
        feed_dict = {images: batch, train_mode: False}

        vgg = vgg16.Vgg16('./vgg16.npy')
        vgg.build(images,train_mode)
        
        #添加了该句
        sess.run(tf.global_variables_initializer())

        prob = sess.run(vgg.prob, feed_dict=feed_dict)
        print(prob)
        utils.print_prob(prob[0], './synset.txt')
        utils.print_prob(prob[1], './synset.txt')
