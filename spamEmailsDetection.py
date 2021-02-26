import numpy as np
import jieba
import re
import os
import json
import datetime
import random

def scu_stopwords():
    with open('./停用词.txt', encoding='utf-8', errors='ignore') as f:
        stopwordList = f.readlines()
    return stopwordList


def textParse(path, stopwordList):
    '''
    解析每一封邮件的内容为分词
    :param path: 邮件路径
    :return: 分词list
    '''
    character = re.compile('[\u4e00-\u9fff]+')
    seg_list = []
    with open(path, 'r', encoding='gb2312', errors='ignore') as f:
        mail = f.read()
        head, context = mail.split('\n\n', 1)
        seg_temp_list = list(jieba.cut(context))
        for i in seg_temp_list.copy():
            if len(i) > 1 and character.match(i) and i not in stopwordList:
                seg_list.append(i)
    return seg_list


def createVocabList(dataSet):
    '''
    创建一个包含在所有文档中出现的不重复词的列表
    :param dataSet: 词
    :return: vocabset: 词汇表
    '''
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 并集
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    '''
    词集模型
    输入邮件的分词与词汇表对照,出现的标为1
    :param vocabList: 词汇表
    :param inputSet: 某个文档
    :return: returnVec: 文档向量，向量的每一元素为1或0
    '''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('词: ' + word + ' 不在字典中')
    return returnVec


def bagOfWords2VecMN(vocabList, inputSet):
    '''
    词袋模型
    输入邮件的分词与词汇表对照,出现的标为出现次数
    :param vocabList: 词汇表
    :param inputSet: 某个文档
    :return: returnVec: 文档向量
    '''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    '''
    计算两个类别的概率向量和是垃圾邮件的概率
    :param trainMatrix: 文档矩阵
    :param trainCategory: 类别矩阵
    :return:
    '''
    numTrainDocs = len(trainMatrix)  # 训练数量
    numWords = len(trainMatrix[0])  # 字典数量
    pSpam = sum(trainCategory) / float(numTrainDocs)  # 垃圾邮件概率
    # 拉普拉斯平滑
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pSpam


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''
    :param vec2Classify: 词向量
    :param p0Vec: 词在正常中的概率向量
    :param p1Vec: 词在垃圾中的概率向量
    :param pClass1: 垃圾邮件概率
    :return: 是否为垃圾邮件 1-垃圾邮件 0-正常邮件
    '''
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def spamTest(trainNum, testNum):
    character = re.compile('[\u4e00-\u9fff]+')
    num = re.compile('[0|1]')
    stopwordList = scu_stopwords()
    if os.path.exists('./model/{}'.format(trainNum)):
        print('已有{}个训练样本的模型'.format(trainNum))
        # 读取训练模型
        p0V = np.load('./model/{}/{}-p0V.npy'.format(trainNum, trainNum))
        p1V = np.load('./model/{}/{}-p1V.npy'.format(trainNum, trainNum))
        pSpam = np.load('./model/{}/{}-pSpam.npy'.format(trainNum, trainNum))
        vocabList = list(np.load('./model/{}/{}-vocabList.npy'.format(trainNum, trainNum), allow_pickle=True))
        docList = list(np.load('./model/{}/{}-docList.npy'.format(trainNum, trainNum), allow_pickle=True))
        classList = list(np.load('./model/{}/{}-classList.npy'.format(trainNum, trainNum), allow_pickle=True))

    else:
        print('没有{}个训练样本的模型'.format(trainNum))
        docList = []
        classList = []
        fullText = []
        with open('./trec06c/full/index', 'r') as f:
            index = f.readlines()
        for i in index[:trainNum]:
            type, path = i.split(' ')
            path = path.replace('../', './trec06c/')
            path = path.replace('\n', '')
            wordList = textParse(path, stopwordList)
            docList.append(wordList)
            fullText.append(wordList)
            if type == 'spam':
                classList.append(1)  # 1代表垃圾邮件
            else:
                classList.append(0)
        vocabList = createVocabList(docList)  # 创建词典
        trainingSet = range(trainNum)
        trainMat = []
        trainClasses = []
        for docIndex in trainingSet:
            trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
            trainClasses.append(classList[docIndex])

        p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
        # 保存训练模型
        if os.path.exists('./model')==False:
            os.mkdir('./model')
        os.mkdir('./model/{}'.format(trainNum))
        np.save('./model/{}/{}-p0V.npy'.format(trainNum, trainNum), p0V)
        np.save('./model/{}/{}-p1V.npy'.format(trainNum, trainNum), p1V)
        np.save('./model/{}/{}-pSpam.npy'.format(trainNum, trainNum), pSpam)
        np.save('./model/{}/{}-vocabList.npy'.format(trainNum, trainNum), np.array(vocabList))
        np.save('./model/{}/{}-docList.npy'.format(trainNum, trainNum), np.array(docList))
        np.save('./model/{}/{}-classList.npy'.format(trainNum, trainNum), np.array(classList))


    # 判断正确率 准确率 召回率
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    result = {}
    testSet = []
    while len(testSet) < testNum:
        testSet.append(random.randrange(trainNum,trainNum+30000))
    with open('./trec06c/full/index', 'r') as f:
        index = f.readlines()
    for n in testSet: #将testSet的分词加入docList和类加入classList
        i = index[n]
        type, path = i.split(' ')
        path = path.replace('../', './trec06c/')
        path = path.replace('\n', '')
        wordList = textParse(path, stopwordList)
        docList.append(wordList)
        if type == 'spam':
            classList.append(1)  # 1代表垃圾邮件
        else:
            classList.append(0)
    for docIndex in range(trainNum,trainNum+testNum):
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) == 1 and classList[docIndex] == 1:
            TP += 1
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) == 1 and classList[docIndex] == 0:
            FP += 1
            print("分类错误(FP)", docList[docIndex][:8])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) == 0 and classList[docIndex] == 1:
            FN += 1
            print("分类错误(FN)", docList[docIndex][:8])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) == 0 and classList[docIndex] == 0:
            TN += 1
    ACC = (TP + TN) / (TP + TN + FP + FN)
    precisonRate = TP / (TP + FP)
    recallRate = TP / (TP + FN)
    print(('TP:'+str(TP)+' FP:'+str(FP)+' FN:'+str(FN)+' TN:'+str(TN)))
    # 记录结果到json文件中
    if not os.path.exists('result.json'):
        with open('result.json', 'w') as f:
            info = {'trainNum': trainNum, 'testNum': testNum, 'ACC': ACC, 'precisonRate': precisonRate,
                    'recallRate': recallRate,'datatime':datetime.datetime.now().timestamp()}
            resultList=[]
            resultList.append(info)
            result[str(trainNum)] = resultList
            f.write((json.dumps(result, indent=4, ensure_ascii=False)))
    else:
        with open('result.json', 'r') as f:
            result = json.loads(f.read())
            info = {'trainNum': trainNum, 'testNum': testNum, 'ACC': ACC, 'precisonRate': precisonRate,
                    'recallRate': recallRate, 'datatime': datetime.datetime.now().timestamp()}
            if str(trainNum) in result.keys():
                resultList = result[str(trainNum)]
            else:
                resultList = []
            resultList.append(info)
            result[str(trainNum)] = resultList
        with open('result.json', 'w') as f:
            f.write((json.dumps(result, indent=4, ensure_ascii=False)))
    return ACC, precisonRate, recallRate


if __name__ == '__main__':
    ACC, precisonRate, recallRate = spamTest(1000, 200)
    print('正确率:' + str(ACC) + '准确率:' + str(precisonRate) + '召回率' + str(recallRate))
