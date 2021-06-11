print(__doc__)

from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import pandas
import numpy as np
S = np.zeros((3,2), dtype=float)
y1 =  np.zeros((2), dtype=float)
iris = pandas.read_csv('/home/mahima/Untitled Document.csv')
#print(iris)
S[:,0] = iris.A
S[:,1] = iris.B
y1 = iris.Class
#print(S)
#print(y1)
# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=1)
#clf3 = SVC(gamma=.1, kernel='rbf', probability=True)
#eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),
                                    #('svc', clf3)],
                        #voting='soft', weights=[2, 1, 2])

clf1.fit(S, y1)
clf2.fit(S, y1)
#clf3.fit(S, y1)
#eclf.fit(S, y1)

# Plotting decision regions
x_min, x_max = S[:, 0].min() - 1, S[:, 0].max() + 1
y_min, y_max = S[:, 1].min() - 1, S[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        [clf1, clf2],
                        ['Decision Tree (depth=4)', 'Mahalanobis Distance',
                         'Kernel SVM', 'Soft Voting']):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(S[:, 0], S[:, 1], c=y1,
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)

plt.show()
