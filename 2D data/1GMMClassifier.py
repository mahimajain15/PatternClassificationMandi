# Program to classify 2D data using Unimodal Gaussian and GMM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn import mixture
import sklearn
from sklearn.metrics import classification_report
cmap_1 = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']) # To be used in plots
cmap_2 = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
cmap_3= ListedColormap(['#FFFF00', '#80FF0F', '#0000FF'])
cmap_4=ListedColormap(['red', 'blue', 'yellow'])


Nc=3 # Number of class
NGMM_comp= 10  # Number of components for GMM

d=input('Enter choice for data: 1)Linearly Separable 2)Non-Linearly Separable 3)Image \n')
print("d: "+ repr(d))
if d == 1:
    path = "/home/mahima/LSdata/"
elif d == 2:
    path="/home/mahima/2D data/"
elif d == 3:
    path="/home/mahima/image/"  
#print(path)
def classify_data(X, classifiers):
    y_pred = np.argmax([classifiers[j].score_samples(X) for j in range(0,Nc)],0)
    return y_pred


y=[]
Y=np.empty([0,1])
X=np.empty([0,2])


NGMM_comp=input('Enter the number of mixture components to be used in each Gaussian: \n')

Cov_type=input('Enter the Covariance type to be used:\n 1)full=each component has its own general covariance matrix\n 2)tied=all components share the same general covariance matrix \n 3)diag=each component has its own diagonal covariance m,atrix \n 4)spherical=each component has its own single variance \n')

if Cov_type==1:
    C='full'
elif Cov_type==2:
    C='tied'
elif Cov_type==3:
    C='diag'
else:
    C='spherical'
classifiers = [[] for i in range(0,Nc)]
for i in range(0,Nc):
    classifiers[i]=mixture.GaussianMixture(n_components=NGMM_comp,covariance_type=C)
    #'full'=each component has its own general covariance matrix
    #'tied'=all components share the same general covariance matrix
    #'diag'=each component has its own diagonal covariance m,atrix
    #'spherical'=each component has its own single variance

    #path + "class" + str(i+1) + 
    Xx=pd.read_csv(path + "Class " +str(i+1) + "_train.txt", header = None, delimiter=',', usecols=(0, 23))
    Xx=np.array(Xx)
    classifiers[i].fit(Xx)
    
    m,n=Xx.shape
    y=i*np.ones((m,1))    
    X=np.concatenate((X,Xx))
    Y=np.concatenate((Y,y))
    
h = 0.05 # Grid size
'''
# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = classify_data(np.c_[xx.ravel(), yy.ravel()],classifiers)
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.pcolormesh(xx, yy, Z, cmap=cmap_1)
plt.scatter(X[:,0],X[:,1], c=Y.ravel(), cmap='rainbow')
plt.title('Decision boundary plot using GMM')
'''
    
###################################################
# For Accuracy and CM
x_test=np.empty((0,2))
y_test=np.empty((0,1))

for i in range(0,Nc):
    
    temp=pd.read_csv(path+"Class "+str(i+1)+"_test.txt",header = None, delimiter=',', usecols=(0, 23))
    temp=np.array(temp)
    m,n=temp.shape
    ytemp=i*np.ones((m,1))
    x_test=np.concatenate((x_test,temp))
    y_test=np.concatenate((y_test,ytemp))
    
y_pred = classify_data(x_test, classifiers) 
conf_mat = sklearn.metrics.confusion_matrix(y_test, y_pred)
accuracy = sklearn.metrics.accuracy_score(y_test, y_pred) 
print("The confusion Matrix is")
print(conf_mat)
print('Classification report')
print(classification_report(y_test, y_pred))
print("The accuracy is ")
print(accuracy*100)

###################################################
'''
# Plotting the test data.
x1=x_test[:,0]
y1=x_test[:,1]
plt.figure(2)
plt.pcolormesh(xx, yy, Z, cmap=cmap_1)
plt.scatter(x1,y1,c=y_test.ravel(), cmap='rainbow')
plt.title('Test data with Decision region')
plt.show()
    
'''   
from mpl_toolkits.mplot3d import axes3d
import scipy.stats as st
x = X[:, 0]
y = X[:, 1]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = st.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)

fig = plt.figure(figsize=(13, 7))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('PDF')
ax.set_title('Surface plot of Gaussian 2D KDE')
fig.colorbar(surf, shrink=0.5, aspect=5) # add color bar indicating the PDF
ax.view_init(60, 35)




