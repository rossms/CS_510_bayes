from bayesbest import *

Bayes = Bayes_Classifier()

lFileList = []
for fFileObj in os.walk("./evaluation_docs/"):
    lFileList = fFileObj[2]
    break
posFileList = []
negFileList = []
for fileName in lFileList:
    if re.search(r'\-1\-',fileName):
        negFileList.append(fileName)
    elif re.search(r'\-5\-',fileName):
        posFileList.append(fileName)
# negDictionary = dict()
# posDictionary = dict()
for fileName in negFileList:
    fileText = Bayes.loadFile('./evaluation_docs/'+fileName)
    print(fileName+' '+Bayes.classify(fileText))
for fileName in posFileList:
    fileText = Bayes.loadFile('./evaluation_docs/'+fileName)
    print(fileName+' '+Bayes.classify(fileText))

# testFile = Bayes.loadFile('./evaluation_docs/movies-5-2.txt')
# #tokens = Bayes.tokenize(testFile)
# c = Bayes.classify(testFile)
# print(c)
#testFile = Bayes.loadFile('./test.txt')
#print(testFile)
#print(Bayes.countTokens(Bayes.tokenize(testFile)))

#load
# try:
#     os.remove('./negDictionary.txt')
#     #os.remove('./negSingleDictionary.txt')
#     #os.remove('./negTupleDictionary.txt')
#     #os.remove('./negCapsDictionary.txt')
# except OSError:
#     pass
# try:
#     os.remove('./posDictionary.txt')
#     #os.remove('./posSingleDictionary.txt')
#     #os.remove('./posTupleDictionary.txt')
#     #os.remove('./posCapsDictionary.txt')
# except OSError:
#     pass
# Bayes.train()

#load files if there ?
#negDictionary = Bayes.load('./negDictionary.txt')
#posDictionary = Bayes.load('./posDictionary.txt')
#negSingleDictionary = Bayes.load('./negSingleDictionary.txt')
#posSingleDictionary = Bayes.load('./posSingleDictionary.txt')
#negTupleDictionary = Bayes.load('./negTupleDictionary.txt')
#posTupleDictionary = Bayes.load('./posTupleDictionary.txt')
#negCapsDictionary = Bayes.load('./negCapsDictionary.txt')
#posCapsDictionary = Bayes.load('./posCapsDictionary.txt')
#classify
#classification = Bayes.classify(tokens)
#Bayes.classify(posSingleDictionary,negSingleDictionary,posTupleDictionary,negTupleDictionary,posCapsDictionary,negCapsDictionary,testFile)
#print(posDictionary)
#print(classification)
