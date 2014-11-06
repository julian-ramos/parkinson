import matplotlib.pylab as plt
import dataReader as dR
from dtw import dtw
import dataSearchUtils as dSu
import os
import datetime
import mysql.connector
import numpy as np
import pickle

path='/home/julian/eclipseWorkspace/parkinson/sampleData/'
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

if os.path.isfile(path+'segments-temp.txt'):
    file=open(path+'segments-temp.txt','rb')
    segments=pickle.load(file)
    
    file=open(path+'data-temp.txt','rb')
    data=pickle.load(file)
    
    file=open(path+'timeElapsed-temp.txt','rb')
    timeElapsed=pickle.load(file)
    
    file=open(path+'inds.txt','rb')
    inds=pickle.load(file)
    
    file=open(path+'dates.txt','rb')
    dates=pickle.load(file)      
    
    
else:
    cnx = mysql.connector.connect(user='julian', password='julianepico',
                                  host='epiwork.hcii.cs.cmu.edu',
                                  database='upmc_parkinson')
     
    cursor = cnx.cursor()
    query = ("SELECT * from watch_24")
     
    cursor.execute(query)
    data=[]
    inds1S=[]
    inds5S=[]
    previous=-1
    ind=-1
    #Filter out data from the 24th only
    dayFilter='24'
    monthFilter='10'
    segments=[]
     
    for (data_retrieved) in cursor:
        temp=list(data_retrieved)
        day=datetime.datetime.fromtimestamp(int(temp[1])/1000.0).strftime('%d')
        month=datetime.datetime.fromtimestamp(int(temp[1])/1000.0).strftime('%m')
        if day==dayFilter and month==monthFilter:
            data.append(temp[0:2]+temp[3:6]) 
     
    for row in range(len(data)):
        if row>=1:
            previous=data[row-1][1]
            #Finding the gaps higher than 1 sec
            if data[row][1]-previous > 1000:
                inds1S.append(row-1)
             
                 
    #Measuring time passed between breaks
    ind=-1
    timeElapsed=[]
    dates=[]
    for i in range(len(inds1S)):
        ind+=1
         
        if ind>0:
            timeElapsed.append((data[inds1S[i]][1]-data[inds1S[i-1]+1][1])/1000.0)
            temp=np.array(data[inds1S[i-1]:inds1S[i]])
            segments.append(temp[:,2:])
            
            end=datetime.datetime.fromtimestamp(data[inds1S[i]][1]/1000.0).strftime('%Y-%m-%d %H:%M:%S')
            start=datetime.datetime.fromtimestamp(data[inds1S[i-1]][1]/1000.0).strftime('%Y-%m-%d %H:%M:%S')
            
            dates.append('%s-%s'%(start,end))
        else:
            timeElapsed.append((data[inds1S[i]][1]-data[0][1])/1000.0)
            temp=np.array(data[0:inds1S[i]])
            segments.append(temp[:,2:])
            
    #Storing data from database
#     file=open(path+'segments-temp.txt','wb')
#     pickle.dump(segments, file)
    file=open(path+'data-temp.txt','wb')
    pickle.dump(data,file)
    file=open(path+'timeElapsed-temp.txt','wb')
    pickle.dump(timeElapsed,file)
    
    file=open(path+'inds.txt','wb')
    pickle.dump(inds1S,file)
    
    file=open(path+'dates.txt','wb')
    pickle.dump(dates,file)      
         

# dynamic time warping section
costs=[]
for segInd in range(len(segments)):
    temp=[[0,0,0] for ix in range(7)]
    for sigInd in range(len(signals)):
        for i in range(3):
             
            x=signals['data'][sigInd][:,i+1]
            y=segments[segInd][:,i]
            
            x=np.array(x)
            y=np.array(y)
            
            x=(x-np.mean(x))/np.std(x)
            y=(y-np.mean(y))/np.std(y)
            
            temp2=dtw(x,y)
            temp[sigInd][i]=temp2[0]
            plt.subplot(2,1,1)
            plt.plot(x,'b')
            plt.title(signals['filename'][sigInd]+'-'+dates[segInd])
               
            plt.subplot(2,1,2)
            plt.plot(y,'r')
            plt.show()
             
    dists=np.array(temp)
            
    print(temp2[0])
             
    
       
       
       
 
        
# # plots
# costs=[]
# for segInd in range(len(segments)):
#         
#         for i in range(3):
#             y=segments[segInd][:,i]
#             plt.subplot(3,1,i+1)
#             plt.plot(y)
#             plt.title('segment'+str(segInd)+' '+dates[segInd])
#         plt.show()


 
print 'done'
 
cnx.close()