from bayes import *

Bayes = Bayes_Classifier()

testFile = Bayes.loadFile('./movies_reviews/movies-5-2.txt')
#testFile = Bayes.loadFile('./test.txt')
#print(testFile)
#print(Bayes.countTokens(Bayes.tokenize(testFile)))
# try:
#     os.remove('./negDictionary.txt')
# except OSError:
#     pass
# try:
#     os.remove('./posDictionary.txt')
# except OSError:
#     pass
#Bayes.train()
negDictionary = Bayes.load('./negDictionary.txt')
posDictionary = Bayes.load('./posDictionary.txt')

Bayes.classify(posDictionary,negDictionary,testFile)
#print(posDictionary)
