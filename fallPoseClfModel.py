import os
import numpy as np
import cv2
import struct
import pandas
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from util import *
import pickle

dir = '//Volumes/SL-BG1/data/'




W = 512
H = 424

win = 8

def genClass(fileName_fall, fileName_ADL):
    class1 = [[], [], [], []]
    class0 = []

    for ii in range(4):
        for fileName in fileName_fall[ii]:
            csvName = fileName + '/Body/Fileskeleton.csv'
            data = process(csvName)
            avg = mean_(data)
            step = 30
            maxEnd = None
            maxDescent = 0
            for i in range(step, avg.shape[0]):
                if avg[i, 1] - avg[i - step, 1] > maxDescent:
                    maxDescent = avg[i, 1] - avg[i - step, 1]
                    maxEnd = i

            for i in range(step):
                if maxEnd - step + win + i < data.shape[0]:
                    class0.append(np.reshape(normalize(data[maxEnd - step + i: maxEnd - step + win + i, :, :]), -1))
            for i in range(maxEnd - step):
                if np.random.randint(3) == 0:
                    class1[ii].append(np.reshape(normalize(data[i: i + win, :, :]), -1))

    for fileName in fileName_ADL:
        csvName = fileName + '/Body/Fileskeleton.csv'
        data = process(csvName)
        for i in range(data.shape[0] - win):
            if np.random.randint(3) == 0:
                class0.append(np.reshape(normalize(data[i: i + win, :, :]), -1))

    allIndexOfClass0 = list(range(len(class0)))
    np.random.shuffle(allIndexOfClass0)
    lenOfClass1 = len(class1[0])
    class0 = [class0[i] for i in allIndexOfClass0[:lenOfClass1]]

    x = np.concatenate((np.array(class0), np.array(class1[0]), np.array(class1[1]), np.array(class1[2]), np.array(class1[3])))
    y = np.array([0] * lenOfClass1 + [1] * lenOfClass1 + [2] * len(class1[1]) + [3] * len(class1[2]) + [4] * len(class1[3]))

    print("Num of each class: %d" % lenOfClass1)

    return x, y



if __name__ == '__main__':

    trainFileName_fall = [[], [], [], []]
    trainFileName_ADL = []
    testFileName_fall = [[], [], [], []]
    testFileName_ADL = []

    trainRange = 8

    for ii in range(1, 12):
        dirs = 'Data' + str(ii)
        for index, styles in enumerate(['endUpSit', 'back', 'front', 'side']):
            for fileName in os.listdir(dir + dirs + '/fall/' + styles):
                if fileName[0] == '.': continue
                _fileName = dir + dirs + '/fall/' + styles + '/' + fileName
                if ii <= trainRange:
                    trainFileName_fall[index].append(_fileName)
                else:
                    testFileName_fall[index].append(_fileName)
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


    x_train, y_train = genClass(trainFileName_fall, trainFileName_ADL)
    x_test, y_test = genClass(testFileName_fall, testFileName_ADL)

    clf = XGBClassifier()
    clf = clf.fit(x_train, y_train)
    pickle.dump(clf, open('fallPoseClfModel.pkl', 'wb'))

    answer = clf.predict(x_test)
    print(classification_report(y_test, answer, target_names=['0', '1', '2', '3', '4']))



    fileName = testFileName_ADL[np.random.randint(len(testFileName_ADL))]
    csvName = fileName + '/Body/Fileskeleton.csv'
    data = process(csvName)
    y = []
    for i in range(data.shape[0] - win):
        y.append(clf.predict_proba(np.reshape(normalize(data[i: i + win, :, :]), (1, -1)))[0][0])
    plt.plot(y)
    plt.show()
    # 把每种姿势的摔倒分开来看，效果并不好