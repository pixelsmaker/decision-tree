import collections
from math import log
import operator
import treePlotter

# 载入数据集
def loadDataset(path):
    data=[]
    for line in open(path,"r"):
        line=line.strip("\n").split("\t")
        line=list(map(eval,line))
        data.append(line)
    return data


# 计算给定数据集的信息熵(香农熵)
def calcShannonEnt(dataSet):

    # 计算出数据集的总数
    numEntries = len(dataSet)

    # 用来统计标签
    labelCounts = collections.defaultdict(int)

    # 循环整个数据集，得到数据的分类标签
    for featVec in dataSet:
        # 得到当前的标签
        currentLabel = featVec[-1]

        # 将对应的标签值加一
        labelCounts[currentLabel] += 1

    # 默认的信息熵
    shannonEnt = 0.0

    for key in labelCounts:
        # 计算出当前分类标签占总标签的比例数
        prob = float(labelCounts[key]) / numEntries

        # 以2为底求对数
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt


# 按照给定的数值，将数据集分为不大于和大于两部分
def splitDataSetForSeries(dataSet, axis, value):

    # 用来保存不大于划分值的集合
    eltDataSet = []
    # 用来保存大于划分值的集合
    gtDataSet = []
    # 进行划分，保留该特征值
    for featVec in dataSet:
        if featVec[axis] <= value:
            eltDataSet.append(featVec)
        else:
            gtDataSet.append(featVec)

    return eltDataSet, gtDataSet


# 按照给定的数据对离散值特征划分数据集
def splitDataSet(dataSet, axis, value):

    # 创建一个新的列表，防止对原来的列表进行修改
    retDataSet = []

    # 遍历整个数据集
    for featVec in dataSet:
        # 如果给定特征值等于想要的特征值
        if featVec[axis] == value:
            # 将该特征值前面的内容保存起来
            reducedFeatVec = featVec[:axis]
            # 将该特征值后面的内容保存起来，所以将给定特征值给去掉了
            reducedFeatVec.extend(featVec[axis + 1:])
            # 添加到返回列表中
            retDataSet.append(reducedFeatVec)

    return retDataSet

# 计算值连续特征的信息增益
def calcInfoGainForSeries(dataSet, i, baseEntropy):

    # 记录最大的信息增益
    maxInfoGain = 0

    # 最好的划分点
    bestMid = -1

    # 得到数据集中所有的当前特征值列表
    featList = [example[i] for example in dataSet]

    # 得到分类列表
    classList = [example[-1] for example in dataSet]

    dictList = dict(zip(featList, classList))

    # 将其从小到大排序，按照连续值的大小排列
    sortedFeatList = sorted(dictList.items(), key=operator.itemgetter(0))


    # 计算连续值有多少个
    numberForFeatList = len(sortedFeatList)

    # 计算划分点，保留三位小数
    midFeatList = [round((sortedFeatList[i][0] + sortedFeatList[i+1][0])/2.0, 3)for i in range(numberForFeatList - 1)]

    # 计算出各个划分点信息增益
    for mid in midFeatList:
        # 将连续值划分为不大于当前划分点和大于当前划分点两部分
        eltDataSet, gtDataSet = splitDataSetForSeries(dataSet, i, mid)

        # 计算两部分的特征值熵和权重的乘积之和
        newEntropy = len(eltDataSet)/len(featList)*calcShannonEnt(eltDataSet) + len(gtDataSet)/len(featList)*calcShannonEnt(gtDataSet)
        # 计算出信息增益
        infoGain = baseEntropy - newEntropy

        print(infoGain)
        # print('当前划分值为：' + str(mid) + '，此时的信息增益为：' + str(infoGain))
        if infoGain > maxInfoGain:
            bestMid = mid
            maxInfoGain = infoGain
    print(maxInfoGain)
    return maxInfoGain, bestMid

# 计算离散特征的信息增益
def calcInfoGain(dataSet ,featList, i, baseEntropy):

    # 将当前特征唯一化，也就是说当前特征值中共有多少种
    uniqueVals = set(featList)

    # 新的熵，代表当前特征值的熵
    newEntropy = 0.0

    # 遍历现在有的特征的可能性
    for value in uniqueVals:
        # 在全部数据集的当前特征位置上，找到该特征值等于当前值的集合
        subDataSet = splitDataSet(dataSet=dataSet, axis=i, value=value)
        # 计算出权重
        prob = len(subDataSet) / float(len(dataSet))
        # 计算出当前特征值的熵
        newEntropy += prob * calcShannonEnt(subDataSet)

    # 计算出“信息增益”
    infoGain = baseEntropy - newEntropy

    return infoGain

# 选择最好的特征来进行划分
def chooseBestFeatureToSplit(dataSet, labels):

    # 得到数据的特征值总数
    numFeatures = len(dataSet[0]) - 1

    # 计算出基础信息熵
    baseEntropy = calcShannonEnt(dataSet)

    # 基础信息增益为0.0
    bestInfoGain = 0.0

    # 最好的特征值
    bestFeature = -1

    # 标记当前最好的特征值是不是连续值
    flagSeries = 0

    # 如果是连续值的话，用来记录连续值的划分点
    bestSeriesMid = 0.0

    # 对每个特征值进行求信息熵
    for i in range(numFeatures):

        # 得到数据集中所有的当前特征值列表
        featList = [example[i] for example in dataSet]

        if isinstance(featList[0], str):
            infoGain = calcInfoGain(dataSet, featList, i, baseEntropy)
        else:
            print('当前划分属性为：' + str(labels[i]))
            infoGain, bestMid = calcInfoGainForSeries(dataSet, i, baseEntropy)

        print('当前特征值为：' + labels[i] + '，对应的信息增益值为：' + str(infoGain))
        # 如果当前的信息增益比原来的大
        if infoGain > bestInfoGain:
            # 最好的信息增益
            bestInfoGain = infoGain
            # 新的最好的用来划分的特征值
            bestFeature = i
            flagSeries = 1
            bestSeriesMid = bestMid

    print('信息增益最大的特征为：' + labels[bestFeature])

    if flagSeries:
        print(bestSeriesMid)
        return bestFeature, bestSeriesMid
    else:
        return bestFeature


# 找到次数最多的类别标签
def majorityCnt(classList):

    # 用来统计标签的票数
    classCount = collections.defaultdict(int)

    # 遍历所有的标签类别
    for vote in classList:
        classCount[vote] += 1

    # 从大到小排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    # 返回次数最多的标签
    return sortedClassCount[0][0]


# 创建决策树
def createTree(dataSet, labels):

    # 拿到所有数据集的分类标签
    classList = [example[-1] for example in dataSet]

    # 统计第一个标签出现的次数，与总标签个数比较，如果相等则说明当前列表中全部都是一种标签，此时停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 计算第一行有多少个数据，如果只有一个的话说明所有的特征属性都遍历完了，剩下的一个就是类别标签
    if len(dataSet[0]) == 1:
        # 返回剩下标签中出现次数较多的那个
        return majorityCnt(classList)

    # 选择最好的划分特征，得到该特征的下标
    bestFeat = chooseBestFeatureToSplit(dataSet=dataSet, labels=labels)

    # 得到最好特征的名称
    bestFeatLabel = ''

    # 记录此刻是连续值还是离散值,1连续，2离散
    flagSeries = 0

    # 如果是连续值，记录连续值的划分点
    midSeries = 0.0

    # 如果是元组的话，说明此时是连续值
    if isinstance(bestFeat, tuple):
        # 重新修改分叉点信息
        bestFeatLabel = str(labels[bestFeat[0]]) + '<' + str(bestFeat[1]) + '?'
        # 得到当前的划分点
        midSeries = bestFeat[1]
        # 得到下标值
        bestFeat = bestFeat[0]
        # 连续值标志
        flagSeries = 1
    else:
        # 得到分叉点信息
        bestFeatLabel = labels[bestFeat]
        # 离散值标志
        flagSeries = 0

    # 使用一个字典来存储树结构，分叉处为划分的特征名称
    myTree = {bestFeatLabel: {}}

    # 得到当前特征标签的所有可能值
    featValues = [example[bestFeat] for example in dataSet]

    # 连续值处理
    if flagSeries:
        # 将连续值划分为不大于当前划分点和大于当前划分点两部分
        eltDataSet, gtDataSet = splitDataSetForSeries(dataSet, bestFeat, midSeries)
        # 得到剩下的特征标签
        subLabels = labels[:]
        # 递归处理小于划分点的子树
        subTree = createTree(eltDataSet, subLabels)
        myTree[bestFeatLabel]['Y'] = subTree

        # 递归处理大于当前划分点的子树
        subTree = createTree(gtDataSet, subLabels)
        myTree[bestFeatLabel]['N'] = subTree

        return myTree

    # 离散值处理
    else:
        # 将本次划分的特征值从列表中删除掉
        del (labels[bestFeat])
        # 唯一化，去掉重复的特征值
        uniqueVals = set(featValues)
        # 遍历所有的特征值
        for value in uniqueVals:
            # 得到剩下的特征标签
            subLabels = labels[:]
            # 递归调用，将数据集中该特征等于当前特征值的所有数据划分到当前节点下，递归调用时需要先将当前的特征去除掉
            subTree = createTree(splitDataSet(dataSet=dataSet, axis=bestFeat, value=value), subLabels)
            # 将子树归到分叉处下
            myTree[bestFeatLabel][value] = subTree
        return myTree


# 对单一例子使用决策树的分类函数,这里默认特征都是连续值
def classify(inputTree,labels,testVec):

    firstStr=list(inputTree.keys())[0]
    secondDict=inputTree[firstStr]
    featIndex=labels.index(firstStr[0:2])
    if testVec[featIndex]<float(firstStr[3:-1]):
        key=list(secondDict.keys())[0]
    else:
        key = list(secondDict.keys())[1]
    if type(secondDict[key]).__name__ == 'dict':
        classlabel = classify(secondDict[key], labels, testVec)
    else:
        classlabel = secondDict[key]

    return classlabel


# 求整个测试数据集对分类准确率
def TestAccuracy(inputTree,labels,DataSet):

    # 用来记录准确分类个数
    count = 0
    # 遍历数据集中所有对测试用例
    for feat in DataSet:
        predict_label = classify(inputTree, labels, feat)
        if predict_label == feat[-1]:
            count += 1
    # 计算测试准确率
    accuracy = float(count) / len(DataSet)
    print("accuracy:" + str(accuracy * 100) + '%')


if __name__ == '__main__':
    """
    处理连续值时候的决策树
    """
    dataSet = loadDataset("traindata.txt")
    labels = ['f1', 'f2', 'f3', 'f4']
    # 特征对应的所有可能的情况
    labels_full = {}
    for i in range(len(labels)):
        labelList = [example[i] for example in dataSet]
        uniqueLabel = set(labelList)
        labels_full[labels[i]] = uniqueLabel
    myTree = createTree(dataSet, labels)
    treePlotter.createPlot(myTree)
    testdataSet = loadDataset("testdata.txt")
    TestAccuracy(myTree,labels,testdataSet)

