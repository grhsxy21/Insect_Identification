# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2
import GetChineseName

# 用于获取特征
import GetFeatures
# 用于预测
import Predict


def cb(path):
    img = cv2.imread(path)
    #cv2.imshow("OpenCV",img)
    #cv2.waitKey(10) #一定要延时，否则不能正常显示
    # 1. 调用图像处理
    [everythingRight ,P_rect, P_extend, P_spherical, P_leaf, P_circle, processedImg] = GetFeatures.GetFiveFeatures(img)
    # [everythingRight ,P_rect, P_extend, P_spherical, P_leaf, P_circle,P_complecate, processedImg] = GetFeatures.GetFiveFeatures(self.captureFrame)
    if everythingRight:
        # 2. 调用预测模型预测
        Result = Predict.Predict_LinearSVM(P_rect, P_extend, P_spherical, P_leaf, P_circle)
        # 3. 调取结果对应中文名
        name = GetChineseName.getname(Result)
        # 4. 显示结果
        print ('processed')
        print (Result,name)

    else:
        # 状态栏
        print ('processed')


if __name__ == '__main__':
    print(cb("D:/GitHub/ZRB/Insect_Identification/ProcessedData/fly3.jpg"))
    print(cb("D:/GitHub/ZRB/Insect_Identification/ProcessedData/wo3.jpg"))
    print(cb("D:/GitHub/ZRB/Insect_Identification/ProcessedData/cang0.jpg"))
    print(cb("D:/GitHub/ZRB/Insect_Identification/dataset/zhang0.jpg"))
    print(cb("D:/GitHub/ZRB/Insect_Identification/dataset/zhizhu4.jpg"))

