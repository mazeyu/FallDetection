import os
import numpy as np
import cv2
import struct
import pandas
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import *
from util import *
import pickle

dir = '//Volumes/SL-BG1/data/'




W = 512
H = 424

win = 8

def genClass(fileName_fall, fileName_ADL):
    class0 = []
    class1 = []

    for fileName in fileName_fall:
        csvName = fileName + '/Body/Fileskeleton.csv'
        data = process(csvName)
        avg = mean_(data)
        step = 30
        maxEnd = None
        maxDescent = 0
        # plt.plot(-avg[:, 1])
        # plt.savefig('fig1.jpg')

        for i in range(step, avg.shape[0]):
            if avg[i, 1] - avg[i - step, 1] > maxDescent:
                maxDescent = avg[i, 1] - avg[i - step, 1]
                maxEnd = i
        # 同过落差自动定位摔倒的片段

        for i in range(step):
            if maxEnd - step + win + i < data.shape[0]:
                class1.append(np.reshape(normalize(data[maxEnd - step + i: maxEnd - step + win + i, :, :]), -1))
        for i in range(maxEnd - step):
            if np.random.randint(3) == 0:
                class0.append(np.reshape(normalize(data[i: i + win, :, :]), -1))
        # 摔倒前和摔倒中的分为两类

    for fileName in fileName_ADL:
        csvName = fileName + '/Body/Fileskeleton.csv'
        data = process(csvName)
        for i in range(data.shape[0] - win):
            if np.random.randint(3) == 0:
                class0.append(np.reshape(normalize(data[i: i + win, :, :]), -1))
        # ADL也加入摔倒前的一类

    allIndexOfClass0 = list(range(len(class0)))
    np.random.shuffle(allIndexOfClass0)
    lenOfClass1 = len(class1)
    class0 = [class0[i] for i in allIndexOfClass0[:lenOfClass1]]

    x = np.concatenate((np.array(class0), np.array(class1)))
    y = np.array([0] * lenOfClass1 + [1] * lenOfClass1)

    print("Num of each class: %d" % lenOfClass1)

    return x, y



if __name__ == '__main__':

    trainFileName_fall = []
    trainFileName_ADL = []
    testFileName_fall = []
    testFileName_ADL = []

    trainRange = 8

    for ii in range(1, 12):
        dirs = 'Data' + str(ii)
        for styles in ['endUpSit', 'back', 'front', 'side']:
            for fileName in os.listdir(dir + dirs + '/fall/' + styles):
                if fileName[0] == '.': continue
                _fileName = dir + dirs + '/fall/' + styles + '/' + fileName
                if ii <= trainRange:
                    trainFileName_fall.append(_fileName)
                else:
                    testFileName_fall.append(_fileName)
        for styles in ['sit', 'lay', 'walk', 'grasp']:
            for fileName in os.listdir(dir + dirs + '/ADL/' + styles):
                if fileName[0] == '.': continue
                _fileName = dir + dirs + '/ADL/' + styles + '/' + fileName
                if ii <= trainRange:
                    trainFileName_ADL.append(_fileName)
                else:
                    testFileName_ADL.append(_fileName)

    print(len(trainFileName_fall))
    print(len(trainFileName_ADL))
    print(len(testFileName_fall))
    print(len(testFileName_ADL))

    # try:
    #     clf = pickle.load(open('simpleModel.pkl', 'rb'))
    # except:


    x_train, y_train = genClass(trainFileName_fall, trainFileName_ADL)
    x_test, y_test = genClass(testFileName_fall, testFileName_ADL)
    # 分别生成训练测试用例

    clf = XGBClassifier()
    clf = clf.fit(x_train, y_train)
    pickle.dump(clf, open('simpleModel.pkl', 'wb'))


    answer = clf.predict(x_test)
    print(classification_report(y_test, answer, target_names=['0', '1']))
    # 每一帧的准确率


    fileName = testFileName_ADL[np.random.randint(len(testFileName_ADL))]
    csvName = fileName + '/Body/Fileskeleton.csv'
    data = process(csvName)
    y = []
    for i in range(data.shape[0] - win):
        y.append(clf.predict_proba(np.reshape(normalize(data[i: i + win, :, :]), (1, -1)))[0][0])
    plt.plot(y)
    plt.show()
    # 选定某一次，看预测出的摔倒概率随时间的变化，可以看出很不稳定