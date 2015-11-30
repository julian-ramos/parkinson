# import matplotlib.pylab as plt
# import numpy as np
# import funs as fun
#  
#  
# cnx=fun.connect2DB()
# queryS=('select score from watch_24')
# data=fun.query(queryS, cnx)
# # print(np.shape(data))
#  
# inds=np.random.randint(0,len(data),(1000))
# data=np.array(data)
# plt.hist(data[inds])
# plt.show()
# import numpy as np
# from sklearn import datasets
# from sklearn.multiclass import OneVsOneClassifier
# from sklearn.svm import LinearSVC
# iris = datasets.load_iris()
# X, y = iris.data, iris.target
# y=y.tolist()
# print(OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X))
# print(np.shape(X))
# print(np.shape(y))

import numpy as np
a=np.array([True,False,True,False,True])
print(np.where(a))
