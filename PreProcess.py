#coding=utf-8
#* 需要注意的是，经过resize的预处理会使得昆虫原有的形态特征发生变化，不利于后期的比较

import os
import cv2

path = "E:/Documents/Python_Script/Insect_Identification/dataset"   #TODO
files = os.listdir(path)    #返回指定的文件夹包含的文件或文件夹的名字的列表
print(files)


for filename in files: #遍历文件夹
    if not os.path.isdir(filename): #判断是否是文件夹，不是文件夹才打开     判断路径是否为目录
        print (filename)
        img = cv2.imread("E:/Documents/Python_Script/Insect_Identification/dataset/" + filename)    #TODO
        img2 = cv2.resize(img,(200,200))    #!文件名不能是中文
        cv2.imwrite("E:/Documents/Python_Script/Insect_Identification/processedData/" + filename,img2)  #TODO