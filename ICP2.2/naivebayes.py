
import pandas as pd
from sklearn.naive_bayes import GaussaianNB
from sklearn.model_selection import train_test_split

#load data set
glass = pd.read_csv('glass.csv')
x = glass[['RI','Na','Mg','AI','Si','K','Ca','Ba','Fe']]
y = glass['Type']  #naming the target

#evaluating with train/test split to form training and testing
X_train, X_test, y_train, y_test= train_test_split(X_train, Y_train, test_size=0.4, random_state=0)

print("Training data")
print(X_train.shape,y_train.shape)
print("Training data")
print(X_test.shape,y_test.shape)

clf = GaussaianNB()
clf.fit(X_train, y_train)
#test accuracy on training set
print('The accuracy on Naive Bayes on training set: {:.3f}'.format(clf.score(X_train,y_train)))
#test accuracy on testing set
print('The accuracy on Naive Bayes on testing set {:.3f}'.format(clf.score(X_test,y_test)))






