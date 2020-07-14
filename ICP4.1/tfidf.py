
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vector = TfidfVectorizer()
tfidf_Vector1 = TfidfVectorizer(ngram_range=(1, 2))
tfidf_Vector2 = TfidfVectorizer(stop_words='english')

X_train_tfidf = tfidf_Vector.fit_transform(twenty_train.data)
X_train_tfidf1 = tfidf_Vector1.fit_transform(twenty_train.data)
X_train_tfidf2 = tfidf_Vector2.fit_transform(twenty_train.data)

clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)
clfOne = MultinomialNB()
clfOne.fit(X_train_tfidf1, twenty_train.target)
clfTwo = MultinomialNB()
clfTwo.fit(X_train_tfidf2, twenty_train.target)

X_test_tfidf = tfidf_Vect.transform(twenty_test.data)
prediction = clf.predict(X_test_tfidf)
score = round(metrics.accuracy_score(twenty_test.target, prediction), 4)
print("MultinomialNB : ", score)

predictionOne = clf1.predict(X_test_tfidf1)
scoreOne = round(metrics.accuracy_score(twenty_test1.target, predictionOne), 4)
print("MultinomialNB accuracy when using bigram is: ", scoreOne)
#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train_tfidf, twenty_train.target)
accuracyKNN = round(knn.score(X_train_tfidf, twenty_train.target) * 100, 2)
print ("Accuracy for KNN is: ", accuracyKNN /100)







