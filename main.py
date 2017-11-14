from bayes_best import *

Bayes = Bayes_Classifier()

testFile = Bayes.loadFile('./movies_reviews/movies-1-168.txt')
#testFile = Bayes.loadFile('./test.txt')
#print(testFile)
#print(Bayes.countTokens(Bayes.tokenize(testFile)))

#load
# try:
#     os.remove('./negSingleDictionary.txt')
#     os.remove('./negTupleDictionary.txt')
#     os.remove('./negCapsDictionary.txt')
# except OSError:
#     pass
# try:
#     os.remove('./posSingleDictionary.txt')
#     os.remove('./posTupleDictionary.txt')
#     os.remove('./posCapsDictionary.txt')
# except OSError:
#     pass
# Bayes.train()

#load files if there ?
negSingleDictionary = Bayes.load('./negSingleDictionary.txt')
posSingleDictionary = Bayes.load('./posSingleDictionary.txt')
negTupleDictionary = Bayes.load('./negTupleDictionary.txt')
posTupleDictionary = Bayes.load('./posTupleDictionary.txt')
negCapsDictionary = Bayes.load('./negCapsDictionary.txt')
posCapsDictionary = Bayes.load('./posCapsDictionary.txt')
#classify
#Bayes.classify(posSingleDictionary,negSingleDictionary,testFile)
Bayes.classify(posSingleDictionary,negSingleDictionary,posTupleDictionary,negTupleDictionary,posCapsDictionary,negCapsDictionary,testFile)
#print(posDictionary)
