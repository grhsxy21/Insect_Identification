# coding=utf-8
# 读取文件夹下图片，并进行特征提取，保存到CSV文件中

import csv
import os
import cv2
import re

import GetFeatures

path = "D:/GitHub/Insect_Identification/dataset"   #TODO
files = os.listdir(path)

for filename in files:  # 遍历文件夹
    if not os.path.isdir(filename):  # 判断是否是文件夹，不是文件夹才打开
        print (filename)
        img = cv2.imread("D:/GitHub/Insect_Identification/dataset/" + filename)    #TODO
        [P_rect, P_extend, P_spherical, P_leaf, P_circle, processedImg] = GetFeatures.GetFiveFeatures(img)
        # [P_rect, P_extend, P_spherical, P_leaf, P_circle,P_complecate, processedImg] = GetFeatures.GetFiveFeatures(img)


        # 此处用正则表达式,提取名字作为类别
        x = re.findall('([a-zA-Z]*)[0-9]+\.jpg', filename)
        species = x[0]

        # 判断名字进行分类保存
        data = [P_rect, P_extend, P_spherical, P_leaf, P_circle, species]
        # data = [P_rect, P_extend, P_spherical, P_leaf, P_circle,P_complecate, species]

        # 保存到CSV
        with open('data/data.csv', 'a+') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(data)
