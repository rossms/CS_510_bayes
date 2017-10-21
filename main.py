from bayes import *

Bayes = Bayes_Classifier()

#testFile = Bayes.loadFile('./movies_reviews/movies-1-10044.txt')
testFile = Bayes.loadFile('./test.txt')
print(testFile)
print(Bayes.countTokens(Bayes.tokenize(testFile)))
