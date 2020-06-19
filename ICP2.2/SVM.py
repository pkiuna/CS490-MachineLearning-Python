
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm  import LinearSVC

#load data set
glass = pd.read_csv('glass.csv')
x = glass[['RI','Na','Mg','AI','Si','K','Ca','Ba','Fe']]
y = glass['Type']  #naming the target

#evaluating with train/test split to form training and testing
X_train, X_test, y_train, y_test= train_test_split(X_train,Y_train, test_size=0.4, random_state=0)

print("Training data")
print(X_train.shape,y_train.shape)

print("Training data")
print(X_test.shape,y_test.shape)

#implementing SVM method

svm = LinearSVC(random_state=0, tol=1e-5)
svm.fit(X_train, y_train)
print('The accuracy of SVM on training set: '.format(svm.score(X_train,y_train)))
#test accuracy on testing set
print('The accuracy of SVM on testing set'.format(svm.score(X_test,y_test)))







