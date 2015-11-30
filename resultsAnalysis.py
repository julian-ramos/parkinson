import matplotlib.pylab as plt
import pickle
import numpy as np




file=open('./results/experiment_reg1l105.txt','rb')
results=pickle.load(file)
file.close()
file=open('./results/results_expTask22.csv','wb')

header='meanAcu,stdAcu,'+\
        'meanPrec0,stdPrec0,'+\
        'meanPrec1,stdPrec1,'+\
        'meanRec0,stdRec0,'+\
        'meanRec1,stdRec1,'+\
        'meanF0,stdF0,'+\
        'meanF1,stdF1,'+\
        'meanSup0,stdSup0,'+\
        'meanSup1,stdSup1,'+\
        'classifier,'+\
        'winLen,'+\
        'overLap'+\
        '\n'
        
file.write(header)
for cvIter in results['data']:
    #Results for every fold
    temp=[]
    for fold in cvIter:
        if len(fold)!=12:
            print('crossvalidation fold %d produced insufficient results omitting'%(len(temp)))
            continue
        temp.append(fold[:-3])
    means=np.mean(np.array(temp),0)
    stds=np.std(np.array(temp),0)
    line=''
    for i in range(len(means)):
        line+='%.4f,%.4f,'%(means[i],stds[i])
    file.write(line+'%s,%d,%d\n'%(fold[-3],fold[-2],fold[-1]))

file.close()
    


print('done')