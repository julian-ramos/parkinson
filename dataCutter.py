import funs as fun
import matplotlib.pylab as plt
import matplotlib.pyplot as pyplt
import dataReader as dR
from dtw import dtw
import dataSearchUtils as dSu
import os
import datetime
import mysql.connector
import numpy as np
import pickle

# path='/home/julian/eclipseWorkspace/parkinson/sampleData/'

path='./intermediateData/'
calculate=True
taskNum=False
graph=False
filtering=False

if taskNum:
    filename='segments_task%d.txt'%(taskNum)
else:
    filename='segments.txt'






if os.path.isfile(path+'timeElapsed.txt')and not(calculate):
    print('loading data')
    file=open(path+'segments.txt','rb')
    segments=pickle.load(file)
    
    file=open(path+'data.txt','rb')
    data=pickle.load(file)
    
    file=open(path+'timeElapsed.txt','rb')
    timeElapsed=pickle.load(file)
    
    file=open(path+'inds.txt','rb')
    inds=pickle.load(file)
    
    file=open(path+'dates.txt','rb')
    dates=pickle.load(file)      
    
    
else:
    print('Retrieving data from the database')
    cnx = mysql.connector.connect(user='julian', password='julianepico',
                                  host='epiwork.hcii.cs.cmu.edu',
                                  database='upmc_parkinson')
     
    cursor = cnx.cursor()
    

    if taskNum:
        # Query for different tasks
        query = ("SELECT * from watch_24 where task=%d"%(taskNum))
        print(query)
        print('Executing query for task number %d'%(taskNum))
    else:
        # Query for on vs OFF
        query = ("SELECT * from watch_24")
        print('Executing query for all tasks')
     
    cursor.execute(query)
    data=[]
    timestamp=[]
    participant=[]
    score=[]
    medicated=[]
    task=[]
    
    
    inds1S=[]
    inds5S=[]
    previous=-1
    ind=-1
    #Filter out data from the 24th only
    dayFilter='24'
    monthFilter='10'
    segments={'data':[],
              'timestamp':[],
              'participant':[],
              'medicated':[],
              'score':[],
              'task':[]
              }

    print('Processing data from the database')
    for (data_retrieved) in cursor:
        temp=list(data_retrieved)
        day=datetime.datetime.fromtimestamp(int(temp[1])/1000.0).strftime('%d')
        month=datetime.datetime.fromtimestamp(int(temp[1])/1000.0).strftime('%m')
        if day==dayFilter and month==monthFilter:
            data.append(temp[0:2]+temp[3:6])
            participant.append(temp[7])
            medicated.append(temp[8])
            task.append(temp[9]) 
            score.append(temp[10])
            timestamp.append(temp[1])
            pass
            
            
     
    for row in range(len(data)):
        if row>=1:
            previous=data[row-1][1]
            #Finding the gaps higher than 1 sec
            if data[row][1]-previous > 2000:
                inds1S.append(row-1)
             
                 
    #Measuring time passed between breaks
    ind=-1
    timeElapsed=[]
    dates=[]
    for i in range(len(inds1S)):
        ind+=1
         
        if ind>0:
            timeElapsed.append((data[inds1S[i]][1]-data[inds1S[i-1]+1][1])/1000.0)
            temp=np.array(data[inds1S[i-1]+1:inds1S[i]])
            
            if temp.tolist()==[]:
                print('found empty segment - omitting')
                continue
                        
            if graph:
                plt.plot(temp[:,2])
                plt.plot(temp[:,3])
                plt.plot(temp[:,4])
                if filtering:
                    inds=fun.superFilter(temp)
                    if inds!=None:
                        temp=np.delete(temp,inds,0)
                     
                plt.plot(temp[:,2])
                plt.plot(temp[:,3])
                plt.plot(temp[:,4])
                plt.show()
                
            

            
            segments['data'].append(temp[:,2:])
            segments['participant'].append(participant[inds1S[i-1]+1:inds1S[i]])
            segments['medicated'].append(medicated[inds1S[i-1]+1:inds1S[i]])
            segments['task'].append(task[inds1S[i-1]+1:inds1S[i]]) 
            segments['score'].append(score[inds1S[i-1]+1:inds1S[i]])
            segments['timestamp'].append(timestamp[inds1S[i-1]+1:inds1S[i]])
            
            end=datetime.datetime.fromtimestamp(data[inds1S[i]][1]/1000.0).strftime('%Y-%m-%d %H:%M:%S')
            start=datetime.datetime.fromtimestamp(data[inds1S[i-1]][1]/1000.0).strftime('%Y-%m-%d %H:%M:%S')
            
            dates.append('%s-%s'%(start,end))
        else:
            timeElapsed.append((data[inds1S[i]][1]-data[0][1])/1000.0)
            
            temp=np.array(data[0:inds1S[i]])
            
            if graph:
                plt.plot(temp[:,2])
                plt.plot(temp[:,3])
                plt.plot(temp[:,4])

                if filtering:
                    inds=fun.superFilter(temp)
                    if inds!=None:
                        temp=np.delete(temp,inds,0)
                plt.plot(temp[:,2])
                plt.plot(temp[:,3])
                plt.plot(temp[:,4])
                plt.show()
            
            segments['data'].append(temp[:,2:])
            segments['participant'].append(participant[0:inds1S[i]])
            segments['medicated'].append(medicated[0:inds1S[i]])
            segments['task'].append(task[0:inds1S[i]]) 
            segments['score'].append(score[0:inds1S[i]])
            segments['timestamp'].append(timestamp[0:inds1S[i]])
            
    #Storing data from database
    print('Writing to file %s'%(filename))
    file=open(path+filename,'wb')
    pickle.dump(segments, file)
    file=open(path+'data.txt','wb')
    pickle.dump(data,file)
    file=open(path+'timeElapsed.txt','wb')
    pickle.dump(timeElapsed,file)
    file=open(path+'inds.txt','wb')
    pickle.dump(inds1S,file)
    file=open(path+'dates.txt','wb')
    pickle.dump(dates,file)      

print 'done'
 
cnx.close()