import matplotlib.pyplot as plt
import json

def getAverage(list):
    ACCSum = 0
    precSum = 0
    recallSum = 0
    for i in list:
        ACCSum += i['ACC']
        precSum += i['precisonRate']
        recallSum += i['recallRate']
    return ACCSum/len(list),precSum/len(list),recallSum/len(list)

def readInfo():
    with open('./result.json','r') as f:
        result = json.loads(f.read())
    ACCDic = {}
    precDic = {}
    recallDic = {}
    for i in result:
        ACCDic[i],precDic[i],recallDic[i]= getAverage(result[i])

    return ACCDic,precDic,recallDic

def draw(ACCDic,precDic,recallDic):
    plt.figure()
    plt.plot(list(ACCDic.keys()),list(ACCDic.values()),color='red',label='ACC')
    plt.plot(list(precDic.keys()), list(precDic.values()), color='blue',label='precison')
    plt.plot(list(recallDic.keys()), list(recallDic.values()), color='pink',label='recall')
    plt.legend(loc='upper right')
    plt.show()
if __name__ == '__main__':
    ACCDic,precDic,recallDic = readInfo()
    draw(ACCDic, precDic, recallDic)
