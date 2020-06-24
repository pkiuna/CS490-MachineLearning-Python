
import pandas as pd
#emulating ggplot style sheet
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')

train.SalePrice.describe()
print(train[['GarageArea']], train[['SalePrice']])

#scatter plot
plt.style.use(style='ggplot')
#changing rc settings in the matplotlib package
plt.rcParams['figure.figsize'] = (15, 6)
plt.scatter(train.GarageArea, train.SalePrice, color='g')

fltr = train[(train.GarageArea < 1000) & (train.GarageArea > 300) & (train.SalePrice < 700000)]  #removing inconsistencies
plt.scatter(fltr.GarageArea,fltr.SalePrice, color='g')
plt.show()










