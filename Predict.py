#coding=utf-8
#加载模型进行预测

# 加载提前训练好的模型
#from sklearn.externals import joblib
import joblib   #*sklearn的版本太新了
from sklearn import *
import numpy as np


def Predict_LinearSVM(P_rect, P_extend, P_spherical, P_leaf, P_circle):
# def Predict_LinearSVM(P_rect, P_extend, P_spherical, P_leaf, P_circle,P_complecate):
    lsvc = joblib.load('D:/GitHub/ZRB/Insect_Identification/model/lsvc.model')
    #*在最新版本的sklearn中，所有的数据都应该是二维矩阵，哪怕它只是单独一行或一列，所以要进行格式改正   list不能使用reshape，需要将其转化为array
    #print(np.array([P_rect, P_extend, P_spherical, P_leaf, P_circle]).reshape(-1,1))
    #print(np.array([P_rect, P_extend, P_spherical, P_leaf, P_circle]))
    print(np.array([[P_rect, P_extend, P_spherical, P_leaf, P_circle]]))
    #result = lsvc.predict(np.array([P_rect, P_extend, P_spherical, P_leaf, P_circle]).reshape(-1,1))
    result = lsvc.predict(np.array([[P_rect, P_extend, P_spherical, P_leaf, P_circle]]))
    # result = lsvc.predict([P_rect, P_extend, P_spherical, P_leaf, P_circle,P_complecate])
    return result[0]

def Predict_LinearRegression(P_rect, P_extend, P_spherical, P_leaf, P_circle):
# def Predict_LinearRegression(P_rect, P_extend, P_spherical, P_leaf, P_circle,P_complecate):
    lr = joblib.load('D:/GitHub/ZRB/Insect_Identification/model/lr.model')

    result = lr.predict([P_rect, P_extend, P_spherical, P_leaf, P_circle])
    # result = lr.predict([P_rect, P_extend, P_spherical, P_leaf, P_circle,P_complecate])
    return result[0]

def Predict_KneiborClassfier(P_rect, P_extend, P_spherical, P_leaf, P_circle):
# def Predict_KneiborClassfier(P_rect, P_extend, P_spherical, P_leaf, P_circle,P_complecate):
    knc = joblib.load('D:/GitHub/ZRB/Insect_Identification/model/lr.model')

    result = knc.predict([P_rect, P_extend, P_spherical, P_leaf, P_circle])
    # result = knc.predict([P_rect, P_extend, P_spherical, P_leaf, P_circle,P_complecate])
    return result[0]
