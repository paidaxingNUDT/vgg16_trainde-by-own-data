import os
import numpy as np
import utils
import sys
import random
import linecache

class dataset:
    def __init__(self, labelpath):
        self.cache_data = linecache.getlines(labelpath)
        self.len = len(self.cache_data)

    def readline(self,line):
        return self.cache_data[line]
 
    def getrandomline(self):
        """读取文件的任意一行"""
        return self.readline(random.randint(0,self.len-1))

    def getbatch(self,batchsize = 4):
        img = []
        labels = []
        for i in range(batchsize):
            path,label = self.getrandomline().split()
            onehot_label = [1 if i == int(label) else 0 for i in range(25)]
            img1 = utils.load_image(path)
            img1.reshape((224,224,3))
            img.append(img1)
            labels.append(onehot_label)
        return np.array(img),np.array(labels)

if __name__ == "__main__":
    img1 = utils.load_image("./test_data/tiger.jpeg")
    img2 = utils.load_image("./test_data/puzzle.jpeg")

    batch1 = img1.reshape((1, 224, 224, 3))
    batch2 = img2.reshape((1, 224, 224, 3))

    batch = np.concatenate((batch1, batch2), 0)

    datapath = "./label.txt"
    for i in range(100):
        getbatch(4,datapath)