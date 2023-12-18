import pandas as pd
import seaborn as sns
from sklearn import linear_model

df = pd.read_csv(r'C:\Users\MOHAN\ML\train_data.csv')

sns.lmplot(x='age',y='premium',data=df)

reg = linear_model.LinearRegression()

reg.fit(df[['age']],df['premium'])

print(reg.predict([[21]]))

#validating our model
#linear equation: y = mx + c
#ex. premium = m * age + c
age = int(input("Enter age for finding premium: ")
premium_age = reg.coef_ * age + reg.intercept_
print(f"premium_of_age_{age}: ",premium_age)