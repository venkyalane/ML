import numpy as np
import pandas as pd

# Read data from csv.
dataset = pd.read_csv('Country_data.csv')
print(dataset)

# split dependant and independent data
x = dataset[['Country','Age','Salary']].values
y = dataset[['Purchased']].values
print("x_data: \n",x)
print("y_data: \n",y)

# missing values replace using SimpleImputer
from sklearn.impute import SimpleImputer
x[:,1:3] = SimpleImputer(missing_values = np.NaN, strategy = 'mean').fit(x[:,1:3]).fit_transform(x[:,1:3])
print("after missing values replaced: \n",x)

# label encoding
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])
print("After label encoder(x): \n",x)

# Dummy encoding
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
x = onehotencoder.fit_transform(dataset.Country.values.reshape(-1,1)).toarray()
print("After Dummy encoding(x): \n",x)

# label encoder for Y data
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print("After label encoder(y): \n",y)

# split train data and test data(80%,20%)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print("x_train data: \n",x_train)
print("x_test data: \n",x_test)
print("y_train data: \n",y_train)
print("y_test data: \n",x_test)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
print("after feature scaling x_train: \n",x_train)
print("after feature scaling x_test: \n",x_test)