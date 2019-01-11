import os
import numpy as np
import utils
import sys
import random

def getfilelines(filename, eol='\n', buffsize=4096):
    """计算给定文件有多少行"""
    with open(filename, 'r') as handle:
        linenum = 0
        buffer = handle.read(buffsize)
        while buffer:
            linenum += buffer.count(eol)
            buffer = handle.read(buffsize)
        return linenum

def readtline(filename, lineno, eol="\n", buffsize=4096):
    """读取文件的指定行"""
    with open(filename, 'r') as handle:
        readedlines = 0
        buffer = handle.read(buffsize)
        while buffer:
            thisblock = buffer.count(eol)
            if readedlines < lineno < readedlines + thisblock:
                # inthisblock: findthe line content, and return it
                return buffer.split(eol)[lineno - readedlines - 1]
            elif lineno == readedlines + thisblock:
                # need continue read line rest part
                part0 = buffer.split(eol)[-1]
                buffer = handle.read(buffsize)
                part1 = buffer.split(eol)[0]
                return part0 + part1
            readedlines += thisblock
            buffer = handle.read(buffsize)
        else:
            raise IndexError
 
 
def getrandomline(filename):
    """读取文件的任意一行"""
    return readtline(
        filename,
        random.randint(0, getfilelines(filename)),
        )

def getbatch(batchsize = 4,datapath = None):
    img = []
    labels = []
    for i in range(batchsize):
        path,label = getrandomline(datapath).split()
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
    batch,labels = getbatch(4,datapath)