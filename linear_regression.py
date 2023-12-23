import pandas as pd
import seaborn as sns
from sklearn import linear_model

# read  data from csv file and convert into dataframe
df = pd.read_csv(r'C:\Users\MOHAN\ML\train_data.csv')

# make stastical graphics
sns.lmplot(x='age',y='premium',data=df)

# Train our model
reg = linear_model.LinearRegression()
reg.fit(df[['age']],df['premium'])

age = int(input("Enter age for finding premium: "))

# test our model
predict_premium = reg.predict([[age]])
print(f"predicted premium for age {age}: ",predict_premium)

#validating our model
#linear equation: y = mx + c
#ex. premium = m * age + c
validating_premium = reg.coef_ * age + reg.intercept_
print(f"premium_of_age_{age} after validating our model: ",validating_premium)

try:
    if predict_premium == validating_premium:
        print("model perfectlly work....wel done...")

except:
    print("something went wrong in training model, pls check model!!!!")