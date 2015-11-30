from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import tree
from sklearn.linear_model import LinearRegression as linearR 
from sklearn.metrics import mean_squared_error as msr
from sklearn.linear_model import LogisticRegression as lR
from sklearn.linear_model import ElasticNet as eN
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import accuracy_score as acu
from sklearn import svm
from sklearn.svm import LinearSVC
import numpy as np
import os
import matplotlib.pylab as plt
import funs as fun
import pickle
from sqlparse.filters import OutputFilter
from numpy.core.numeric import extend_all
from bs4 import testing
from _sqlite3 import DatabaseError
from httplib import PROCESSING

def benchmark(winLen,overLap,classifier,nout,maxFreq,loadFile,fast=False,featureSelection=False,topK=10):

#     winLen=4000
#     overLap=20

    path='./intermediateData/'
    resultsPath='./results/'
    
    calculate=True
    
    graphPSD=False
    results=[]
    resultsFile='results.txt'
    print('Starting experiment for winlen : %s - Overlap %s'%(winLen,overLap))
    print(' for nOut : %s - maxFreq %s'%(nout,maxFreq))
    
    if calculate:
        print('Calculating features')
        file=open(path+loadFile,'rb')
        segments=pickle.load(file)
        file.close()
        
        
        
        segments['psd']=[]
        segments['f']=[]
        segments['psdStats']=[]
        segments['norm']=[]
        
        
        for i in range(len(segments['data'])):
            data=segments['data'][i]
            timestamps=segments['timestamp'][i]
            
            #The overlap for psd has to be very small however we
            #control over the window size on the psdStats
            temp,f,norm=fun.psd(data, timestamps,winLen=winLen,overlap=1,nout=nout,maxFreq=maxFreq)
            if temp==[]:
                print('segment of data is too small')
                segments['norm'].append([])
                segments['psd'].append([])
                segments['psdStats'].append([])
                
                continue
            segments['norm'].append(norm)
            segments['psd'].append(temp)
            temp2=fun.winStats(temp, timestamps,winLen=winLen,overlap=overLap)
            segments['psdStats'].append(temp2)
            
        
        segments['f']=f
    
        file=open(path+loadFile,'wb')
        pickle.dump(segments,file)
        file.close()
    else:
        print('loading features')
        file=open(path+'segments2.txt','rb')
        segments=pickle.load(file)
        file.close()
    
    
    
    if graphPSD:
        for i in range(len(segments['psd'])):
            
            print(len(segments['psdStats'][i]))
            print(np.shape(segments['psdStats'][i]))
            temp=np.array(segments['psd'][i])
            temp=np.transpose(temp)
            plt.subplot(2,1,1)
            plt.imshow(temp)
            plt.subplot(2,1,2)
            plt.plot(segments['norm'][i])
            plt.show()
                
        print(segments.keys())
    
    
    #Crossvalidation part
    
    partList=['1','2','3','4','5']
    
    
    for i in partList:
        trainParIds=[ i2 for i2 in partList if i2!=i]
        trainData=[]
        testData=[]
        trainLabels=[]
        testLabels=[]
        trainScores=[]
        testScores=[]
        
        for i2 in range(len(segments['participant'])):
            if segments['participant'][i2][0] in trainParIds:
                temp=segments['psdStats'][i2]
                temp=np.array(temp)
                n=len(segments['psdStats'][i2])
                tempLabels=segments['medicated'][i2][:n]
                tempScores=segments['score'][i2][:n]
                
                
                if temp==[]:
                    print('empty segment')
                    continue
                
                if trainData==[] and temp!=[]:
                    trainData=temp
                    trainLabels=tempLabels
                    trainScores=tempScores
                elif temp!=[]:
                    trainData=np.vstack((trainData,temp))
                    trainLabels=trainLabels+tempLabels
                    trainScores=trainScores+tempScores
                    
                
                
    
            elif segments['participant'][i2][0] == i:
                
                temp=np.array(segments['psdStats'][i2])
                n=len(segments['psdStats'][i2])
                tempLabels=segments['medicated'][i2][:n]
                tempScores=segments['score'][i2][:n]
                
                if temp==[]:
                    print('empty segment')
                    continue
                
                if testData==[] and temp!=[]:
                    testData=temp
                    testLabels=tempLabels
                    testScores=tempScores
                elif temp!=[]:
                    testData=np.vstack((testData,temp))
                    testLabels=testLabels+tempLabels
                    testScores=testScores+tempScores
                    
        #ML section
        
        #Feature selection
        if featureSelection:
            topFeatures= SelectKBest(chi2, k=topK).fit(trainData, trainLabels)
            trainData=topFeatures.transform(trainData)
            testData=topFeatures.transform(testData)
            print(topFeatures.get_support(indices=True))
            
            
        
        
        
        
        print('Starting classification')
        
        #Initialization part
    
        if classifier=='logistic':
            model=lR()
        if classifier=='SVM-1':
            model=svm.SVC(kernel='rbf',degree=3)
        if classifier=='SVM-2':
            model=svm.SVC(kernel='rbf',degree=3,gamma=10)
        if classifier=='regression':
#             model=eN(l1_ratio=0.5)
#             model=linearR(fit_intercept=True)
            model=tree.DecisionTreeRegressor()
            
        if classifier=='multi':
            print(i)
            trainScores=np.array(trainScores)
            
            predictedScores=OneVsOneClassifier(svm.SVC(kernel='rbf',degree=3)).fit(trainData, trainScores).predict(testData)
#             predictedScores=OneVsOneClassifier(LinearSVC(random_state=0)).fit(trainData, trainScores).predict(testData)
            temp=fun.performanceSummary(testScores, predictedScores)
            results.append([temp]+[classifier,winLen,overLap])
            
        if classifier=='multiOvR':
            model=OneVsRestClassifier(svm.SVC(kernel='rbf',degree=3)).fit(trainData, trainScores)
            predictedScores=model.predict(testData)
            temp=fun.performanceSummary(testScores, predictedScores)
            results.append([temp]+[classifier,winLen,overLap])

        #Fit part

        if classifier=='SVM-1' or classifier=='SVM-2':            
            model.fit(trainData,trainLabels)
        elif classifier=='regression':
            model.fit(trainData,trainScores)
        
        
        #Prediction
        if classifier=='SVM-1' or classifier=='SVM-2':
            predictedLabels=model.predict(testData)
        
        #Performance
        
        if classifier=='SVM-1' or classifier=='SVM-2':
            temp=fun.performanceSummary(testLabels, predictedLabels)
        elif classifier=='regression':
            temp=model.score(testData,testScores)
            predictedLabels=model.predict(testData)
#             temp=msr(testScores,predictedLabels)
            print(msr(testScores,predictedLabels),'MSR')
            pass
#             temp=fun.performanceSummary(testScores, predictedLabels)

        if classifier=='SVM-1' or classifier=='SVM-2':
            results.append(temp+[classifier,winLen,overLap])
        elif classifier =='regression':
            results.append([temp]+[classifier,winLen,overLap])
            
        print(results[-1][0])
    
    
    print('done')
    return results


if __name__=='__main__':
    
    
    path='./results/'
    
    #Small winLens
    winLen=[
            5000
            ]
    
    
    #Big winLens
#     winLen=[
#             1000,
#             1500,
#             2000,
#             2500,
#             3000,
#             3500,
#             ]
    
    
    overLap=[
             100,
             ]
    
#     overLap=[20]
    
    results={'data':[],'winLen':[],'overLap':[]}
    
    classifier=[
                'regression'
                ]
    nout=100
    maxFreq=100
    featureSelection=True
    topK=10
    loadFile='segments.txt'
    outputFile='experiment_multiSVM.txt'
    print('results will be stored in %s'%(outputFile))
    
    for iClass in range(len(classifier)):
        for iLap in range(len(overLap)):
            for iWin in range(len(winLen)):
                results['data'].append(benchmark(winLen[iWin], overLap[iLap],classifier[iClass],nout,maxFreq,loadFile=loadFile,topK=topK,featureSelection=featureSelection))
                results['winLen'].append(winLen[iWin])
                results['overLap'].append(overLap[iLap])
                file=open(path+outputFile,'wb')
                pickle.dump(results,file)
                file.close()
        file=open(path+outputFile,'wb')
        pickle.dump(results,file)
        file.close()
    
        
        
    
# This benchmark can be extended so that it can be used by anyone really quickly to do experiments
# however there are a couple of things to extend
# First it needs to include a validation set besides training and testing
# Also the section where regression or different classifiers are selected is kind of mesed up
# the data should be read from a Database
# somehow I need a way to need at any point which features Im processing 
# i thinkI solved this before    
    
    
    
    
