import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

words = sentence = open('input.txt', encoding="utf8").read()

#Tokenization
aTokens = nltk.sent_tokenize(words)
bTokens = nltk.word_tokenize(words)

print("Tokenizing for words")
print(bTokens)
print("Tokenizing for sentence")
print(aTokens)
print("Stemming")

from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

portStemmer = PorterStemmer()
lancStemmer = LancasterStemmer()
stop_words = SnowballStemmer('english')

a1 = 0
for a in bTokens:
   a1 = a1 + 1
   if a1 < 4:

       print(portStemmer.stem(t), lancStemmer.stem(t), stop_words(t))

print("Lemmatilization")
#POS
#Lemmentization

from nltk.stem import WordNetLemmatizer
lemmatization = WordNetLemmatizer
a1 = 0
for a in bTokens:
   a1 = a1 + 1
   if a1 < 6:
       print("Lemmatizer:", lemmatizer.lemmatize(t), ",    With POS=a:", lemmatizer.lemmatize(t, pos="a"))

    #trigram
from nltk.util import ngrams
token = nltk.word_tokenize(words)
a = 0
for b in aTokens:
     a = a + 1
     if a < 2:
         token = nltk.word_tokenize(b)
         biGrams = list(ngrams(token, 2))
         triGrams = list(ngrams(token, 3))
         print("Text is:", b, "\nword_tokenize:", token, "\nbiGrams:", biGrams, "\ntriGrams", triGrams)
#Name Equity Recognition
print("Name Equity Recognition")
from nltk import word_tokenize, pos_tag, ne_chunk
a = 0
for b in aTokens:
    n = n + 1
    if n < 2:
        print(ne_chunk(pos_tag(word_tokenize(s))))




















