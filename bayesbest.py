from __future__ import print_function
from __future__ import division
import math, os, pickle, re, string
from decimal import *

class Bayes_Classifier:
    negSingleDictionary = dict()
    posSingleDictionary = dict()
    negTupleDictionary = dict()
    posTupleDictionary = dict()
    negCapsDictionary = dict()
    posCapsDictionary = dict()
    def __init__(self):
    	'''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
    	cache of a trained classifier has been stored, it loads this cache.  Otherwise,
    	the system will proceed through training.  After running this method, the classifier
    	is ready to classify input text.'''
        global negSingleDictionary
        global posSingleDictionary
        global negTupleDictionary
        global posTupleDictionary
        global negCapsDictionary
        global posCapsDictionary
        if (os.path.isfile('./negSingleDictionary.txt') & os.path.isfile('./posSingleDictionary.txt') & os.path.isfile('./negCapsDictionary.txt') & os.path.isfile('./posCapsDictionary.txt')& os.path.isfile('./negTupleDictionary.txt') & os.path.isfile('./posTupleDictionary.txt')):
            negSingleDictionary = self.load('./negSingleDictionary.txt')
            posSingleDictionary = self.load('./posSingleDictionary.txt')
            negCapsDictionary = self.load('./negCapsDictionary.txt')
            posCapsDictionary = self.load('./posCapsDictionary.txt')
            negTupleDictionary = self.load('./negTupleDictionary.txt')
            posTupleDictionary = self.load('./posTupleDictionary.txt')
        else:
            try:
                os.remove('./negSingleDictionary.txt')
                os.remove('./posSingleDictionary.txt')
                os.remove('./negCapsDictionary.txt')
                os.remove('./posCapsDictionary.txt')
                os.remove('./negTupleDictionary.txt')
                os.remove('./posTupleDictionary.txt')
            except OSError:
                pass
            self.train()
            negSingleDictionary = self.load('./negSingleDictionary.txt')
            posSingleDictionary = self.load('./posSingleDictionary.txt')
            negCapsDictionary = self.load('./negCapsDictionary.txt')
            posCapsDictionary = self.load('./posCapsDictionary.txt')
            negTupleDictionary = self.load('./negTupleDictionary.txt')
            posTupleDictionary = self.load('./posTupleDictionary.txt')

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
        negSingleDictionary = dict()
        posSingleDictionary = dict()
        negTupleDictionary = dict()
        posTupleDictionary = dict()
        negCapsDictionary = dict()
        posCapsDictionary = dict()

        # create single neg and pos dictionaries
        for fileName in negFileList:
            fileText = self.loadFile('./movies_reviews/'+fileName)
            self.countTokens(self.tokenize(fileText),negSingleDictionary)
        for fileName in posFileList:
            fileText = self.loadFile('./movies_reviews/'+fileName)
            self.countTokens(self.tokenize(fileText),posSingleDictionary)
        self.save(negSingleDictionary, './negSingleDictionary.txt')
        self.save(posSingleDictionary, './posSingleDictionary.txt')

        # create tuple (word1, word2), (wordn-1, wordn) neg and pos dictionaries

        for fileName in negFileList:
            fileText = self.loadFile('./movies_reviews/'+fileName)
            self.countTokens(self.tuplize(self.tokenizeCaps(fileText)),negTupleDictionary)
        for fileName in posFileList:
            fileText = self.loadFile('./movies_reviews/'+fileName)
            self.countTokens(self.tuplize(self.tokenizeCaps(fileText)),posTupleDictionary)
        self.save(negTupleDictionary, './negTupleDictionary.txt')
        self.save(posTupleDictionary, './posTupleDictionary.txt')

        # create UPPERCASE neg and pos dictionaries

        for fileName in negFileList:
            fileText = self.loadFile('./movies_reviews/'+fileName)
            self.countTokens(self.tokenizeCaps(fileText),negCapsDictionary)
        for fileName in posFileList:
            fileText = self.loadFile('./movies_reviews/'+fileName)
            self.countTokens(self.tokenizeCaps(fileText),posCapsDictionary)
        self.save(negCapsDictionary, './negCapsDictionary.txt')
        self.save(posCapsDictionary, './posCapsDictionary.txt')


    def classify(self, sText):
        '''Given a target string sText, this function returns the most likely document
        class to which the target string belongs. This function should return one of three
        strings: "positive", "negative" or "neutral".
        '''
        global negSingleDictionary
        global posSingleDictionary
        global negCapsDictionary
        global posCapsDictionary
        global negTupleDictionary
        global posTupleDictionary
        sPosDictionary = posSingleDictionary
        sNegDictionary = negSingleDictionary
        sPosCapsDictionary = posCapsDictionary
        sNegCapsDictionary = negCapsDictionary
        sPosTupleDictionary = posTupleDictionary
        sNegTupleDictionary = negTupleDictionary
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

        ###################
        # take the probablility of single features (each word is its own feature)

        classifyTokens = self.tokenize(sText)
        calculatedProb = self.calculateProbability(classifyTokens, sPosDictionary, sNegDictionary, posDocsLen, negDocsLen, priorPositive, priorNegative)
        #print(calculatedProb[0])
        #print(calculatedProb[1])
        classifyTupleTokens = self.tuplize(self.tokenizeCaps(sText))
        tupleCalculatedProb = self.calculateProbability(classifyTupleTokens, sPosTupleDictionary, sNegTupleDictionary, posDocsLen, negDocsLen, priorPositive, priorNegative)
        #print(tupleCalculatedProb[0])
        #print(tupleCalculatedProb[1])

        classifUpperTokens = self.tokenizeCaps(sText)
        calculatedCapsProb = self.calculateProbability(classifUpperTokens, sPosCapsDictionary, sNegCapsDictionary, posDocsLen, negDocsLen, priorPositive, priorNegative)
        #print(calculatedCapsProb[0])
        #print(calculatedCapsProb[1])
        #probSingleDocPos = math.fabs(calculatedProb[0]/100)
        #probSingleDocNeg = math.fabs(calculatedProb[1]/100)

        #probTupleDocPos = math.fabs(tupleCalculatedProb[0]/100)
        #probTupleDocNeg = math.fabs(tupleCalculatedProb[1]/100)

        probDocPos = round(((calculatedProb[0] * Decimal(0.4)) + (calculatedCapsProb[0] * Decimal(0.4)) + (tupleCalculatedProb[0] * Decimal(0.2))),1)
        probDocNeg = round(((calculatedProb[1] * Decimal(0.4)) + (calculatedCapsProb[1] * Decimal(0.4)) + (tupleCalculatedProb[1] * Decimal(0.2))),1)
        if(calculatedProb[0] > calculatedProb[1]):
            singleProb = 1
        else:
            singleProb = -1
        if(calculatedCapsProb[0] > calculatedCapsProb[1]):
            capsProb = 1
        else:
            capsProb = -1
        if(tupleCalculatedProb[0] > tupleCalculatedProb[1]):
            tupleProb = 1
        else:
            tupleProb = -1

        prob = singleProb + capsProb + tupleProb
        #print(prob)
        if(prob > 0):
            return 'positive'
        else:
            return 'negative'
        #print(probDocPos)
        #print(probDocNeg)
        #sVal
        #if (probSingleDocPos < 0):

        #probDocPos = math.log10(probSingleDocPos) + math.log10(probTupleDocPos)
        #probDocNeg = math.log10(probSingleDocNeg) + math.log10(probTupleDocNeg)
        # probDocPos = probSingleDocPos * probTupleDocPos
        # probDocNeg = probSingleDocNeg * probTupleDocNeg
        # probDocPos = probTupleDocPos
        # probDocNeg = probTupleDocNeg

        # if(int(probDocPos) == int(probDocNeg)):
        #     return 'neutral'
        # elif(probDocPos > probDocNeg):
        # if(probDocPos > probDocNeg):
        #     return 'positive'
        # else:
        #     return 'negative'

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
		sTextWordsOnly = sText.translate(None,string.punctuation)
		for c in sTextWordsOnly:
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
		for word in lTokens:
			 if word in dictionary:
					currCount = int(dictionary.get(word))
					currCount += 1
					dictionary.update({word:currCount})
			 else:
				  dictionary.update({word:1})
		return dictionary

    def tuplize(self, lTokens):
        inTokens = lTokens
        tupleTokens = []
        if len(inTokens) % 2 != 0:
            tupleTokens.append(inTokens[0])
            del inTokens[0]
        while inTokens:
            tupleTokens.append(inTokens[0]+inTokens[1])
            del inTokens[0:2]
        return tupleTokens

    def calculateProbability(self, classifyTokens, sPosDictionary, sNegDictionary, posDocsLen, negDocsLen, priorPositive, priorNegative):
        sumPosDictionaryVals = sum(sPosDictionary.values())
        sumNegDictionaryVals = sum(sNegDictionary.values())
        posFeatureProb = Decimal(0.0)
        negFeatureProb = Decimal(0.0)
        for word in classifyTokens:
            freqWordPos = Decimal((sPosDictionary.get(word,1) + 1) / (sumPosDictionaryVals + posDocsLen))
            posFeatureProb += Decimal(math.log10(freqWordPos))
            freqWordNeg = Decimal((sNegDictionary.get(word,1) + 1)/ (sumNegDictionaryVals + negDocsLen))
            negFeatureProb += Decimal(math.log10(freqWordNeg))

        probDocPos = Decimal(Decimal(math.log10(priorPositive)) + posFeatureProb)
        probDocNeg = Decimal(Decimal(math.log10(priorNegative)) + negFeatureProb)
        return (probDocPos,probDocNeg)

    def tokenizeCaps(self, sText):
        '''Given a string of text sText, returns a list of the individual tokens that
        occur in that string (in order).'''
        lTokens = []
        sToken = ""
        sTextWordsOnly = sText.translate(None,string.punctuation)
        for c in sTextWordsOnly:
            u = c.upper()
            if re.match("[a-zA-Z0-9]", str(u)) != None or u == "\'" or u == "_" or u == '-':
                sToken += u
            else:
                if sToken != "":
                    lTokens.append(sToken)
                    sToken = ""
                if u.strip() != "":
                    lTokens.append(str(u.strip()))
        if sToken != "":
            lTokens.append(sToken)

        return lTokens
