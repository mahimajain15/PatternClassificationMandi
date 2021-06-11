'''
Principal Component Analysis (PCA) is useful for visualizing high-dimensional datasets, as it can compress it down to 2 dimensions. It's also useful for reducing the dimensionality of high-dimensional datasets, which require exponentially more data as the number of dimensions increase, but we didn't need to do that in this case because the dataset was rather small.
'''
from matplotlib import pyplot as plt
import numpy as np
import math
import pandas as pd
url = "/home/mahima/image/var_im_train.txt"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'CLASS'])

from sklearn.preprocessing import StandardScaler
features = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']

# Separating out the features
x = df.loc[:, features].values

# Separating out the target
y = df.loc[:,['CLASS']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

#principalDf = pd.DataFrame(data = principalComponents
#             , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5', 'principal component 6', 'principal component 7', 'principal component 8', 'principal component 9', 'principal component 10', 'principal component 11', 'principal component 12'])

#finalDf = pd.concat([principalDf, df[['CLASS']]], axis = 1)

print('Original number of features:', x.shape[1])
print('Reduced number of features:', principalComponents.shape[1])
#print(x)
#print(len(principalComponents))
#print(finalDf)

with open('/home/mahima/pca1/Class 1_train.txt', 'w') as csvfile:    
    for i in range(50):
        csvfile.write(str(principalComponents[i]))
with open('/home/mahima/pca1/Class 2_train.txt', 'w') as csvfile:    
    for i in range(50):
        csvfile.write(str(principalComponents[i+50]))
with open('/home/mahima/pca1/Class 3_train.txt', 'w') as csvfile:    
    for i in range(50):
        csvfile.write(str(principalComponents[i+100]))

#Fitting the PCA algorithm with our Data
pca = PCA().fit(x)#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Image Dataset Explained Variance')
plt.show()

