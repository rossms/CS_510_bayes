from __future__ import print_function
from __future__ import division
import math, os, pickle, re
from decimal import *

class Bayes_Classifier:
    negDictionary = dict()
    posDictionary = dict()
    def __init__(self):
    	'''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
    	cache of a trained classifier has been stored, it loads this cache.  Otherwise,
    	the system will proceed through training.  After running this method, the classifier
    	is ready to classify input text.'''
        global negDictionary
        global posDictionary
        if (os.path.isfile('./negDictionary.txt') & os.path.isfile('./posDictionary.txt')):
            negDictionary = self.load('./negDictionary.txt')
            posDictionary = self.load('./posDictionary.txt')
        else:
            try:
                os.remove('./negDictionary.txt')
                os.remove('./posDictionary.txt')
            except OSError:
                pass
            self.train()
            negDictionary = self.load('./negDictionary.txt')
            posDictionary = self.load('./posDictionary.txt')

    def train(self):
        '''Trains the Naive Bayes Sentiment Classifier.'''
        lFileList = []
        for fFileObj in os.walk("./movies_reviews/"):
            lFileList = fFileObj[2]
            break
        posFileList = []
        negFileList = []
        for fileName in lFileList:
            if re.search(r'\-1\-',fileName):
                negFileList.append(fileName)
            elif re.search(r'\-5\-',fileName):
                posFileList.append(fileName)
        #print(len(posFileList))
        #print(len(negFileList))
        negDictionary = dict()
        posDictionary = dict()
        for fileName in negFileList:
            fileText = self.loadFile('./movies_reviews/'+fileName)
            self.countTokens(self.tokenize(fileText),negDictionary)
        for fileName in posFileList:
            fileText = self.loadFile('./movies_reviews/'+fileName)
            self.countTokens(self.tokenize(fileText),posDictionary)
        #print(len(negDictionary))
        #print(len(posDictionary))
        self.save(negDictionary, './negDictionary.txt')
        self.save(posDictionary, './posDictionary.txt')

    def classify(self, sText):
        '''Given a target string sText, this function returns the most likely document
        class to which the target string belongs. This function should return one of three
        strings: "positive", "negative" or "neutral".
        '''
        global posDictionary
        global negDictionary
        sPosDictionary = posDictionary
        sNegDictionary = negDictionary
        lFileList = []
        for fFileObj in os.walk("./movies_reviews/"):
            lFileList = fFileObj[2]
            break
        posFileList = []
        negFileList = []
        for fileName in lFileList:
            if re.search(r'\-1\-',fileName):
                negFileList.append(fileName)
            elif re.search(r'\-5\-',fileName):
                posFileList.append(fileName)
        posDocsLen = int(len(posFileList))
        negDocsLen = int(len(negFileList))
        totalDocsLen = int(len(lFileList))
        getcontext.prec = 100
        priorPositive = Decimal(posDocsLen/totalDocsLen)
        priorNegative = Decimal(negDocsLen/totalDocsLen)

        sumPosDictionaryVals = sum(sPosDictionary.values())
        sumNegDictionaryVals = sum(sNegDictionary.values())

        classifyTokens = self.tokenize(sText)
        posFeatureProb = Decimal(0.0)
        negFeatureProb = Decimal(0.0)
        for word in classifyTokens:
            freqWordPos = Decimal((sPosDictionary.get(word,1) + 1) / (sumPosDictionaryVals + posDocsLen))
            posFeatureProb += Decimal(math.log10(freqWordPos))
            freqWordNeg = Decimal((sNegDictionary.get(word,1) + 1)/ (sumNegDictionaryVals + negDocsLen))
            negFeatureProb += Decimal(math.log10(freqWordNeg))

        probDocPos = Decimal(Decimal(math.log10(priorPositive)) + posFeatureProb)
        probDocNeg = Decimal(Decimal(math.log10(priorNegative)) + negFeatureProb)
        #print(probDocPos)
        #print(probDocNeg)

        if(int(probDocPos) == int(probDocNeg)):
            #print('neutral')
            return 'neutral'
        elif(probDocPos > probDocNeg):
            #print('positive')
            return 'positive'
        else:
            #print('negative')
            return 'negative'


    def loadFile(self, sFilename):
		'''Given a file name, return the contents of the file as a string.'''

		f = open(sFilename, "r")
		sTxt = f.read()
		f.close()
		return sTxt

    def save(self, dObj, sFilename):
		'''Given an object and a file name, write the object to the file using pickle.'''

		f = open(sFilename, "w")
		p = pickle.Pickler(f)
		p.dump(dObj)
		f.close()

    def load(self, sFilename):
		'''Given a file name, load and return the object stored in the file.'''

		f = open(sFilename, "r")
		u = pickle.Unpickler(f)
		dObj = u.load()
		f.close()
		return dObj

    def tokenize(self, sText):
		'''Given a string of text sText, returns a list of the individual tokens that
		occur in that string (in order).'''

		lTokens = []
		sToken = ""
		for c in sText:
			if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\'" or c == "_" or c == '-':
				sToken += c
			else:
				if sToken != "":
					lTokens.append(sToken)
					sToken = ""
				if c.strip() != "":
					lTokens.append(str(c.strip()))

		if sToken != "":
			lTokens.append(sToken)

		return lTokens

    def countTokens(self, lTokens, dictionary):
		#dictionary = dict()
		for word in lTokens:
			 if word in dictionary:
					currCount = int(dictionary.get(word))
					currCount += 1
					dictionary.update({word:currCount})
			 else:
				  dictionary.update({word:1})
		return dictionary
