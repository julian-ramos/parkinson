from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import accuracy_score as acu
import scipy.signal as sp2
import os
import dataSearchUtils as dSu
import dataReader as dR
import mysql.connector
import numpy as np
import matplotlib.pylab as plt
from matplotlib.mlab import segments_intersect

def performanceSummary(trueLabels,predictedLabels):
    accuracy=acu(trueLabels,predictedLabels)
    precision,recall,fscore,support=prfs(trueLabels,predictedLabels)
    return [accuracy.tolist()]+precision.tolist()+recall.tolist()+fscore.tolist()+support.tolist()


def winStats(data,timestamp,winLen=2000,overlap=1):
    previous=0
    statsOut=[]
    for i in range(len(data)-overlap):
        if i>0:
            if timestamp[i]-timestamp[previous]>winLen and len(data[previous:i])>10:
                dataWindow=data[previous:i]
                statsOut.append(stats(dataWindow))
#                 timeWindow=timestamp[previous:i]
#                 timeWindow=timeWindow-timeWindow[0]
                previous=previous+overlap
    return statsOut
                
                
def stats(data):
    mean=np.mean(data,0)
    std=np.std(data,0)
    q1=np.percentile(data,25,0)
    q2=np.percentile(data,50,0)
    q3=np.percentile(data,75,0)
    min=np.min(data,0)
    max=np.max(data,0)
    median=np.median(data,0)
    iqr=q3-q1
    
    #Add the IQR
    
    #Add the 10th and 90th quartile
#     return mean,std,q1,q2,q3,min,max
    return np.hstack((mean,std,q1,q2,q3,min,max,median,iqr))
#     return [mean+std+q1+q2+q3+min+max+iqr+median]
    
                
                
            
    

def psd(data,timestamp,winLen=2000,graph=False,overlap=1,nout=20,maxFreq=20):
    psd=[]
    
    data=np.array(data)
    timestamp=np.array(timestamp)
    norm=[]
    f=0
    for i in data:
        norm.append(np.sqrt(i[0]**2+i[1]**2+i[2]**2))
    norm=np.array(norm)
    #Create windows of 2seconds of data
    previous=0
    for i in range(len(timestamp)-overlap):
        if i>0:
            if timestamp[i]-timestamp[previous]>winLen and len(norm[previous:i])>10:
                dataWindow=norm[previous:i]
                timeWindow=timestamp[previous:i]
                timeWindow=(timeWindow-timeWindow[0])/1000
                
                f = np.linspace(0.01, maxFreq, nout)
                tempOut=sp2.lombscargle(timeWindow,dataWindow,f)
                tempOut=np.sqrt(4*tempOut/len(dataWindow))
                previous=previous+overlap
                psd.append(tempOut)
                
                if graph:
                    plt.subplot(2, 1, 1)
                    plt.plot(timeWindow,dataWindow, 'b')
                    plt.subplot(2, 1, 2)
                    plt.plot(f, tempOut)
                    plt.show()
                
    return psd,f,norm        
    
    #First calculate the power spectrum density on the norm
    #then calculate statistics over the resulting
    #have in mind that I should use different frequencies
    #for the power spectrum density estimation
    #Also, remember that the PSD is calculated over a window
    #of data ideally here I also produce the labels
    #for the data points as well
    #Once all this is done I guess the remaining part is just the Crossvalidation
    
    print(norm)
    
    
    
    
    
    
    
    pass

def signalsLoader(path):
    filtering=False
    
    
    
    
    
    listOfFiles=os.listdir(path)        
    inds=dSu.listStrFind(listOfFiles, '.csv')
    listOfFiles=[ listOfFiles[i] for i in inds]
    
    signals={'data':[],'filename':[]}
    
    for filename in listOfFiles:
        filepath='%s%s'%(path,filename)
        signals['data'].append(dR.csvReader(filepath, ',', 0)['data'])
        signals['filename'].append(filename)
        for i in range(len(signals['data'][-1])):
            temp=signals['data'][-1][i]
            for i2 in range(4):
                signals['data'][-1][i][i2]=float(temp[i2])
                
        for i in range(len(signals['data'])):
            signals['data'][i]=np.array(signals['data'][i])
    return signals

def dataNicer(data):
    dataOut=[]
    for i in data:
        dataOut.append(list(i))
    return dataOut

def filterbyDate():
    pass

def connect2DB():
    cnx = mysql.connector.connect(user='julian', password='julianepico',
                                  host='epiwork.hcii.cs.cmu.edu',
                                  database='upmc_parkinson')
     
    
    return cnx 

def query(queryS,cnx):
    cursor=cnx.cursor()
    query = (queryS)
    cursor.execute(query)
    
    data=[i for i in cursor]
    return data


def close(cursor):
    cursor.close()
    
def segmenter(data):
    pass
    
    


    



def segmentsFinder(inds,retBiggest=True):
    previous=-1
    segments=[]
    siz=0
    start=False
    end=False
    starting=-1
    ending=-1
    
    for i in range(len(inds)):
        if inds[i]==previous+1:
            start=True
            if starting==-1:
                starting=i-1
            siz+=1
            
        if inds[i]!=previous+1 and start:
            end=True
            ending=i-1
            
        previous=inds[i]
        if start and end:
            start=False
            end=False
            segments.append([starting,ending,ending-starting])
            starting=-1
            ending=-1
            
        if start==True and i==len(inds)-1:
            segments.append([starting,i,i-starting])
    if segments==[]:
        return None
            
    if retBiggest:
        temp=np.array(segments)
        ind=np.argmax(temp[:,-1])
        return segments[ind]
    
    return segments

def filter(data, window=20, stdVar=1):
    inds=[]
    data=np.array(data)
    data=(data-np.mean(data))/np.std(data)
    
    std=[]
    for i in range(len(data)-window):
        std.append(np.std(data[i:window+i]))
        
        
    meanStd=np.mean(std)
    stdStd=np.std(std)
    
    #Calculating a 95% confidence interval one side
    upperBound=meanStd-stdStd*stdVar
    
    for i in range(len(std)):
        if std[i]<upperBound:
            inds.append(i)
        if std[i]<upperBound and i==len(std)-1 :
            for i2 in range(i,i+window+1):
                inds.append(i2)
    return inds

def superFilter(temp,graph=False):
    #Filtering out sections of the segment that are useless
    inds=[[],[],[]]
    fInds=[]
    for i in range(3):
        inds[i]=(filter(temp[:,i+2],stdVar=0.3))
    #Selecting the intersection of indices
    for i in inds[0]:
        if i in inds[1] and i in inds[2]:
            fInds.append(i)
            
    #Selecting the biggest segment
    segment=segmentsFinder(fInds)
    if segment==None:
        return None
    
    fInds=range(segment[0],segment[1]+1)
            
    if fInds!=[]:        
        if graph:
            zeros=np.zeros(len(temp[:,2]))
            zeros[fInds]=1
            
            plt.plot(temp[:,2])
            plt.plot(temp[:,2]*zeros,'x')
            plt.plot(temp[:,3])
            plt.plot(temp[:,3]*zeros,'x')
            plt.plot(temp[:,4])
            plt.plot(temp[:,4]*zeros,'x')
        
            plt.show()
        
        return fInds
    else:
        return None
    

        
if __name__=='__main__':
    
#     Test for segmentsFinder
#     testData=[0,1,2,3,4,7,8,9,21,25,26,40,42,43]
#     inds=segmentsFinder(testData)
#     for i in inds:
#         print(testData[i[0]:i[1]+1])
#     print(inds)
    pass
