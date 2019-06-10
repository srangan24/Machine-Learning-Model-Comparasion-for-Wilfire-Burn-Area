import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import seaborn as sns 


column_names = ['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH', 'wind', 'rain', 'area'] 
dataset = pd.read_csv('forestfires.csv', names=column_names)

print(dataset.head())
dataset['Log-area'] = np.log10(dataset['area']+1)
print(dataset.describe())

### Univariate plots ###
#box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(5,3), sharex=False, sharey=False)
#histogram
dataset.hist(bins = 50)

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg

axs[0].hist(dataset['area'], bins=40)

axs[0].set_title("Area")
axs[1].hist(dataset['Log-area'], bins=40)
plt.title("Log(Area + 1)")

### Multivariate plots ###
scatter_matrix(dataset)
sns.pairplot(dataset[["Log-area", "temp", "wind", "rain"]], diag_kind="kde")
plt.show()



for i in dataset.describe().columns[:-2]:
    dataset.plot.scatter(i, 'Log-area', grid=True)

plt.show()
plt.close()

