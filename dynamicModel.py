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

distinguishPose = False
numStage = 3



def genClass(fileName_fall, fileName_ADL):
    classes = {}

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

            lenStage = step // numStage
            for stage in range(numStage):
                for i in range(lenStage):
                    if maxEnd - step + win + i + stage * lenStage < data.shape[0]:
                        if distinguishPose:
                            name = 'fall%d%d' % (stage, ii)
                        else:
                            name = 'fall%d' % stage
                        if not classes.__contains__(name):
                            classes[name] = []
                        classes[name].append(np.reshape(normalize(data[maxEnd - step + i + stage * lenStage: maxEnd - step + win + i + stage * lenStage, :, :]), -1))
            # 每段摔倒分为numStage段

            for i in range(maxEnd - step):
                if np.random.randint(3) == 0:
                    name = 'ADL'
                    if not classes.__contains__(name):
                        classes[name] = []
                    classes[name].append(np.reshape(normalize(data[i: i + win, :, :]), -1))
            # 其余为另一类

    for fileName in fileName_ADL:
        csvName = fileName + '/Body/Fileskeleton.csv'
        data = process(csvName)
        for i in range(data.shape[0] - win):
            if np.random.randint(3) == 0:
                classes['ADL'].append(np.reshape(normalize(data[i: i + win, :, :]), -1))

    # 日常活动也属于另一类

    allIndexOfClass0 = list(range(len(classes['ADL'])))
    np.random.shuffle(allIndexOfClass0)
    if distinguishPose:
        lenOfClass1 = len(classes['fall00'])
    else:
        lenOfClass1 = len(classes['fall0'])
    class0 = [classes['ADL'][i] for i in allIndexOfClass0[:lenOfClass1]]

    classes_ = [np.array(class0)]
    y = [0] * lenOfClass1

    for i in range(numStage):
        if distinguishPose:
            for ii in range(4):
                classes_.append(np.array(classes['fall%d%d' % (i, ii)]))
                y.extend([ii * numStage + i + 1] * len(classes['fall%d%d' % (i, ii)]))
        else:
            classes_.append(np.array(classes['fall%d' % i]))
            y.extend([i + 1] * len(classes['fall%d' % i]))

    x = np.concatenate(classes_)
    y = np.array(y)

    print("Num of all classes: %d" % len(y))

    return x, y

def eval(fileName):
    csvName = fileName + '/Body/Fileskeleton.csv'
    print(fileName)
    data = process(csvName)
    pro = []
    for i in range(data.shape[0] - win):
        pro.append(clf.predict_proba(np.reshape(normalize(data[i: i + win, :, :]), (1, -1)))[0])
    pro = np.array(pro)
    N, n = pro.shape
    # print(N, n)
    f = np.zeros((N, n))

    f[0, 0] = pro[0][0]
    for i in range(1, n):
        f[0, i] = -np.inf
    # f(i, j)表示到xi为止，并且xi一定要属于第j类的最大得分 进行动态规划


    for i in range(1, len(pro)):
        for j in range(n):
            if j == 0:
                f[i, j] = f[i - 1, j] + np.log(pro[i][j])
            else:
                candidates = [0] + list(range((j - 1) // numStage * numStage + 1, j + 1))
                values = [f[i - 1, x] for x in candidates]
                f[i, j] = max(values) + np.log(pro[i][j])

    # 选用准则1判定
    if distinguishPose:
        score = -np.inf
        for i in range(4):
            score = max(score, max(f[:, numStage * (i + 1)] - f[:, 0]))
        return score
    else:
        return max(f[:, -1] - f[:, 0])
        # score = 0
        # for i in range(N):
        #     score += max(0, f[i, -1] - f[i, 0])
        # return score

def calThre(scores0, scores1):
    accs = []
    for i in scores0 + scores1:
        acc = 0
        for score in scores0:
            if score < i:
                acc += 1
        for score in scores1:
            if score > i:
                acc += 1
        accs.append(acc)
    thre = (scores0 + scores1)[np.argmax(accs)]
    acc = max(accs) / len(scores1 + scores0)
    # 根据两批得分决定最佳门限
    return thre, acc

def calAcc(scores0, scores1, thre):

    acc = 0
    for score in scores0:
        if score < thre:
            acc += 1
    for score in scores1:
        if score > thre:
            acc += 1

    # 根据门限在测试集计算准确率
    acc /= len(scores1 + scores0)
    return acc




if __name__ == '__main__':

    trainFileName_fall = [[], [], [], []]
    trainFileName_ADL = []

    testFileName_fall = [[], [], [], []]
    testFileName_ADL = []
    testFileName_ADL_ = [[], [], [], []]

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
        for index, styles in enumerate(['sit', 'lay', 'walk', 'grasp']):
            for fileName in os.listdir(dir + dirs + '/ADL/' + styles):
                if fileName[0] == '.': continue
                _fileName = dir + dirs + '/ADL/' + styles + '/' + fileName
                if ii <= trainRange:
                    trainFileName_ADL.append(_fileName)
                else:
                    testFileName_ADL.append(_fileName)
                    testFileName_ADL_[index].append(_fileName)

    print(len(trainFileName_fall))
    print(len(trainFileName_ADL))
    print(len(testFileName_fall))
    print(len(testFileName_ADL))


    modelName = 'dynamicModel%d%d.pkl' % (numStage, distinguishPose)


    try:
        clf = pickle.load(open(modelName, 'rb'))
    except:

        x_train, y_train = genClass(trainFileName_fall, trainFileName_ADL)
        x_test, y_test = genClass(testFileName_fall, testFileName_ADL)

        clf = XGBClassifier()
        clf = clf.fit(x_train, y_train)
        pickle.dump(clf, open(modelName, 'wb'))

        answer = clf.predict(x_test)
        print(classification_report(y_test, answer))

    scores1 = []
    scores0 = []
    for i in range(4):
        cat = trainFileName_fall[i]
        for fileName in cat:
            scores1.append(eval(fileName))
    cat = trainFileName_ADL
    for fileName in cat:
        scores0.append(eval(fileName))

    thre, acc = calThre(scores0, scores1)
    print(thre, acc)


    scores1 = []
    scores0 = []
    scores0_cat = [[], [], [], []]
    scores1_cat = [[], [], [], []]
    for i in range(4):
        cat = testFileName_fall[i]
        for fileName in cat:
            tmp = eval(fileName)
            scores1.append(tmp)
            scores1_cat[i].append(tmp)
    for i in range(4):
        cat = testFileName_ADL_[i]
        for fileName in cat:
            tmp = eval(fileName)
            scores0.append(tmp)
            scores0_cat[i].append(tmp)

    acc = calAcc(scores0, scores1, thre)
    print(acc)
    for i in range(4):
        acc = calAcc(scores0_cat[i], [], thre)
        print(['sit', 'lay', 'walk', 'grasp'][i], acc)
    for i in range(4):
        acc = calAcc([], scores1_cat[i], thre)
        print(['endUpSit', 'back', 'front', 'side'][i], acc)

