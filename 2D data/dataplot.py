import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
from IPython.display import display, HTML

# figures inline in notebook %matplotlib inline

np.set_printoptions(suppress=True)

DISPLAY_MAX_ROWS = 20  # number of max rows to print for a DataFrame
pd.set_option('display.max_rows', DISPLAY_MAX_ROWS)

data = pd.read_csv("/home/mahima/IANNECJ_MAHIMA/synthetic/synthetic_data/linearlySeparableData/group1/final.txt", header=None)
data.columns = ["V"+str(i) for i in range(1, len(data.columns)+1)]  # rename column names to be similar to R naming convention
data.V1 = data.V1.astype(float)
#print(type(data.V1))
X = data.loc[:, "V2":]  # independent variables data
y = data.V1  # dependednt variable data

sns.lmplot("V2", "V3", data, hue="V1", fit_reg=False);
plt.tight_layout()
plt.show()

