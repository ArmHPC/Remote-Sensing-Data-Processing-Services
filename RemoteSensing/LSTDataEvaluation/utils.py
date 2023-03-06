import os, math
import numpy as np

def read_lst_from_img(Img_Data, x, y):
    try:
        # int(Img_Data[y, x]*10)/10 if Img_Data[y, x]==Img_Data[y, x] else -1 
        Y, X = Img_Data.shape
        if y < 0 or y >= Y : return float('nan')
        if x < 0 or x >= X : return float('nan')
        temp_kelvin = float(Img_Data[y, x]).__round__(1)
        return temp_kelvin if temp_kelvin > 0 else float('nan')
    except:
        return float('nan')

def getTifFiles(dir, tifFiles):
    filesInDir = os.listdir(dir)
    for file in filesInDir:
        filePath = dir + "/" + file
        if os.path.isdir(filePath):
            getTifFiles(filePath,tifFiles)
        else:
            if file.endswith('.tif'):
                tifFiles.append(filePath)

def calcRMSE(y_actual,y_predicted): 
    MSE = np.square(np.subtract(y_actual,y_predicted)).mean() 
    RMSE = math.sqrt(MSE)
    return RMSE

def cov(y_actual,y_predicted): 
    cov = np.sum(np.subtract(y_actual, np.mean(y_actual)) * np.subtract(y_predicted, np.mean(y_predicted))) / math.sqrt(np.sum(np.square(np.subtract(y_actual, np.mean(y_actual)))) * np.sum(np.square(np.subtract(y_predicted, np.mean(y_predicted)))))
    return cov

def calcBias(y_actual,y_predicted): 
    return np.sum(np.subtract(y_actual, y_predicted)) / len(y_actual)

def calcMean(y_actual):
    return np.mean(y_actual)

def calcSDF(y_actual): 
    mean = calcMean(y_actual)
    return np.sqrt( np.sum(np.square(y_actual - mean)) / len(y_actual))


def calcStatistics(allDF, station, daytime, channel):
    filterDF = allDF[allDF[channel].notnull() & (allDF['name'] == station) & (allDF['daytime'] == daytime)]
    y_actual = filterDF["temp"].values.tolist()
    y_predicted = filterDF[channel].values.tolist()
    rmse = calcRMSE(y_actual, y_predicted)
    r = cov(y_actual, y_predicted)
    bias = calcBias(y_actual, y_predicted)
    return (rmse, r, bias)