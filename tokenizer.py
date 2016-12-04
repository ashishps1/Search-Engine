import nltk
import os
import pickle
from nltk.stem.porter import *
import math
fileNames = os.listdir('brown')  #to list the name of all the docs inside the brown corpus

stemmer = PorterStemmer()   #object for porter stemmer

dict = {}       #to store term frequency of each file
d = set()       #to store all the distinct words after normalization
d1 = set()      #to store all the distinct words before normalization


for files in fileNames:
    fileName = 'brown/'+files
    #print files
    dict[str(files)]={}
    with open(fileName) as f:       #Reading all the files word by word
        content = f.read()
        for word in content.split():
            ind = word.find('/')
            if word.find(',')==-1 and word.find("'")==-1 and word.find('(')==-1:
                word = word[:ind]
                word=word.lower()           #Converting each word to small case
                d1.add(str(word))
                word = stemmer.stem(word)   #Applying porter stemmer to each word
                d.add(str(word))
                #print word
                if dict[files].has_key(word):
                    k=dict[files][word]
                    dict[files][word]=k+1
                else:
                    dict[files][word]=1    

print len(d1)

invertedIndex = {}      #to store inverted index for each term

for term in d:          #Constructing the inverted index
    invertedIndex[term] = []
    for file in fileNames:
        if dict[file].has_key(term):
            invertedIndex[term].append(file)
           

n = len(fileNames)

tf_idf = {}

for files in fileNames:
    fileName = 'brown/'+files
    #print files
    tf_idf[str(files)]={}
    for key in dict[str(files)]:        #Finding the tf-idf value for each doc
        tf_idf[files][key] = (1 + math.log(dict[files][key],10.0) ) * (math.log(n/(1.0 * len(invertedIndex[key]) ), 10.0))

#Dumping all the data structures to be used while querying
pickle.dump( d1 , open( "words.p", "wb" ) )     
pickle.dump( dict , open( "dict.p", "wb" ) )
pickle.dump( tf_idf , open( "tf_idf.p", "wb" ) )
pickle.dump( invertedIndex , open( "invertedIndex.p", "wb" ) )
