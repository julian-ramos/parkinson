import dataReader as dR
import matplotlib.pylab as plt
import numpy as np


filename='watch_24.csv'
# dR.csvReader(filename,delimiter=','', headerLines=0)
data=dR.csvReader(filename, delimiter=',', headerLines=0)
data=data['data']

data=[ [ float(i2) for i2 in i]for i in data]
data=np.array(data)
plt.plot(data[:,0])
plt.plot(data[:,1])
plt.plot(data[:,2])
plt.show()