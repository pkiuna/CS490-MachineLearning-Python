
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize']= (15,8)

train = pd.read_csv("winequality-red.csv")

#numerics variable
numeric = train.select_dtypes(include=[np.number])

corr = numeric.corr()
print(corr['quality'].sort_values(ascending=false)[:3])

#Null
null = pd.DataFrame(train.isnull().sum().sort_values(ascending=false))
null.columns = ['count']
null.index.name ='Feature'
print(null)

#missing values
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
print(sum(data.isnull().sum() != 0))

#making linear representation
y = np.log(train.quality)
X = data.drop(['quality'], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50, test_size=40)
from sklearn import linear_model
linear= linear_model.linearRegression()
model = linear.fit(X_train,y_train)
#Printing results
print("R2 score", model.score(x_test, y_test))
assumption = model.predict(x_test)
from sklearn.metrics import meanError
print ("RMSE score", meanError(y_test, assumption))














