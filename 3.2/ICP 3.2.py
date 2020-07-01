
import pandas as pd
from sklearn.decomposition import PCA  #PCA Library
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("CC.csv") #reading file

print(data["Longing"].value_counts())


data.loc[(data['Minimum amount of payments'].isnull()==True),'Minimum amount of payments']=data['Minimum amount of payments'].mean()
data.loc[(data['Credit limit'].isnull()==True),'Credit limit']=data['Credit limit'].mean()

x = data.loc[:,2:-2]  #using python.DataFrame
y = data.loc[:,-2]
print(x.shape,y.shape)

#using elbow method to figure out number of clusters
wcss = []
for i in range(1,5):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=200,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1,5),wcss)
plt.title('This is the elbow graph')
plt.xlabel('This is the number of Clusters')
plt.ylabel('Wcss')
plt.show()  #show the following outputs from kMeans algorithm

#number of clusters in kMeans
km = KMeans(n_clusters=5)
km.fit(x)
y_kmeans= km.predict(x)
from sklearn import metrics
score = metrics.silhouette_score(x, y_kmeans)
print(score)

# setting regularity
scalering = StandardScaler()
# training set
scalering.fit(x)

#change both sets
Xscaler = scaler.transform(x)
pca = PCA(5)
x_pca = pca.fit_transform(Xscaler)
data1= pd.DataFrame(data=x_pca)
finalData= pd.concat([data1,data[['LONGING']]],axis=1)
print(finalData)

# KMeans after being formalized

km = KMeans(n_clusters=3)
km.fit(x_pca)
y_kmeans= km.predict(x_pca)
from sklearn import metrics
score = metrics.silhouette_score(x_pca, y_kmeans)
print(score) #printing out score from calculation













