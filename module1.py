import numpy
import os
import shutil
import random
"""
将图像路径和label写入txt文件
"""
datapath = "D:/project/tensorflow-vgg/test_data"
labels = os.listdir(datapath)

filetxt = open("test_label.txt","w")

idx = -1
for dirpath,dirs,files in os.walk(datapath):
    for file in files:
        label = labels[idx]
        one_hot_label = [1 if i == int(label) else 0 for i in range(len(labels))]
        writepath = os.path.join(dirpath,file)
        filetxt.write(writepath)
        filetxt.write(' ')
        filetxt.write(label)
        filetxt.write('\n')
    idx += 1

filetxt.close()

#"""
#将数据r5中的，没有出现在训练数据r20中的数据，挑出来存放
#"""

def getairplane(filename):
    airplane = filename[:filename.index('_')]
    if 'A-10' in airplane:
        idx1 = filename.index('-')
        idx2 = filename[idx1+1:].index('-')
        idx = idx1+idx2+1
        airplane = airplane[:idx]
    return airplane


def get_label_name(airplane):
    synset = "./synset.txt" #里面记录了机型对应的label
    synfile = open(synset,'r')
    while 1:
        line = synfile.readline()
        if line == '':
            print('not found the airplane')
            exit()
        (feiji,label) = line.split()
        if airplane == feiji:
            return label

def getpath(path,filename):
    for dirpath,dirs,files in os.walk(path):
        for file in files:
            if file == filename:
                dst = os.path.join(dirpath,file)
                return dst

def getallimgname(datapath):
    allname = []
    for dirpath,dirs,files in os.walk(datapath):
        for file in files:
            if file.endswith('bmp') == True :
                allname.append(file)
    return allname

#if __name__ == "__main__":
#    datapath_r5 = "D:/project/airplane_data/dataset-r5/image"
#    datapath_r20 = "D:/project/airplane_data/dataset-r20/img"
#    savepath = "D:/project/tensorflow-vgg/test_data"
#    for name in getallimgname(datapath_r5):
#        r20names = getallimgname(datapath_r20)
#        if name in r20names:
#            continue
#        airplane = getairplane(name)
#        label = get_label_name(airplane)
#        dst = os.path.join(savepath,label)
#        if os.path.exists(dst) == False:
#            os.makedirs(dst)
#        dst = os.path.join(dst,name)
#        src = getpath(datapath_r5,name)
#        shutil.copy(src,dst)

#"""
#将所有图像移动为测试集和训练集
#图像组织结构
#|---0
#|——————xxx.jpeg
#|---1
#"""
#if __name__ =="__main__":
#    root_dir = "D:/project/tensorflow-vgg/alldata"
#    test_dir = "D:/project/tensorflow-vgg/test_data"
#    train_dir = "D:/project/tensorflow-vgg/train_data"
#    f = getallimgname(root_dir)
#    random.shuffle(f)
#    trainnum = int(len(f)/5*4)
#    index = 0
#    for i in f:
#        airplane = getairplane(i)
#        label = get_label_name(airplane)
#        if (index < trainnum)== True :
#            index +=1
#            src = getpath(root_dir,i)
#            dst = os.path.join(train_dir,label)
#            if os.path.exists(dst) == False:
#                os.makedirs(dst)
#            dst = os.path.join(dst,i)
#            shutil.copy(src,dst)
                        
#        else:
#            index +=1 
#            src = getpath(root_dir,i)
#            dst = os.path.join(test_dir,label)
#            if os.path.exists(dst) == False:
#                os.makedirs(dst)
#            dst = os.path.join(dst,i)
#            shutil.copy(src,dst)