import funs as fun
import matplotlib.pylab as plt
import dataReader as dR
from dtw import dtw
import dataSearchUtils as dSu
import os
import datetime
import mysql.connector
import numpy as np
import pickle

# path='/home/julian/eclipseWorkspace/parkinson/sampleData/'
path='./sampleData/'
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
        
cnx = mysql.connector.connect(user='julian', password='julianepico',
                                  host='epiwork.hcii.cs.cmu.edu',
                                  database='upmc_parkinson')
     
     
     
     
cursor = cnx.cursor()

query=("show tables")
cursor.execute(query)
tables_list=[ i[0] for i in cursor]
inds=dSu.listStrFind(tables_list,'24_')
tables_list=[tables_list[i] for i in inds]


for i in tables_list:
    query=("select * from %s"%(i))
    cursor.execute(query)
    data=[ i2 for i2 in cursor]
    print('here')


    
# query = ("SELECT * from watch_24")
 
# cursor.execute(query)
cnx.close()
