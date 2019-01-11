import numpy
import os
"""
将图像路径和label写入txt文件
"""
datapath = "D:/project/tensorflow-vgg-master/train_data"
labels = os.listdir(datapath)

filetxt = open("label.txt","w")

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
