from __future__ import print_function
from __future__ import division
import math, os, pickle, re, string
from decimal import *

class Bayes_Classifier:

    def __init__(self, trainDirectory = "movie_reviews/"):
		'''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
		cache of a trained classifier has been stored, it loads this cache.  Otherwise,
		the system will proceed through training.  After running this method, the classifier
		is ready to classify input text.'''

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
        #print(len(negSingleDictionary))
        #print(len(posSingleDictionary))
        self.save(negSingleDictionary, './negSingleDictionary.txt')
        self.save(posSingleDictionary, './posSingleDictionary.txt')

        # create tuple (word1, word2), (wordn-1, wordn) neg and pos dictionaries

        for fileName in negFileList:
            fileText = self.loadFile('./movies_reviews/'+fileName)
            self.countTokens(self.tuplize(self.tokenize(fileText)),negTupleDictionary)
        for fileName in posFileList:
            fileText = self.loadFile('./movies_reviews/'+fileName)
            self.countTokens(self.tuplize(self.tokenize(fileText)),posTupleDictionary)
        #print(len(negTupleDictionary))
        #print(len(posTupleDictionary))
        self.save(negTupleDictionary, './negTupleDictionary.txt')
        self.save(posTupleDictionary, './posTupleDictionary.txt')

        # create UPPERCASE neg and pos dictionaries

        for fileName in negFileList:
            fileText = self.loadFile('./movies_reviews/'+fileName)
            self.countTokens(self.tokenizeCaps(fileText),negCapsDictionary)
        for fileName in posFileList:
            fileText = self.loadFile('./movies_reviews/'+fileName)
            self.countTokens(self.tokenizeCaps(fileText),posCapsDictionary)
        #print(len(negCapsDictionary))
        #print(len(posCapsDictionary))
        self.save(negCapsDictionary, './negCapsDictionary.txt')
        self.save(posCapsDictionary, './posCapsDictionary.txt')


    def classify(self, sPosDictionary, sNegDictionary, sPosTupleDictionary, sNegTupleDictionary, sPosCapsDictionary, sNegCapsDictionary,  sText):
        '''Given a target string sText, this function returns the most likely document
        class to which the target string belongs. This function should return one of three
        strings: "positive", "negative" or "neutral".
        '''
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
        #print('pos docs:',posDocsLen)
        #print('neg docs:',negDocsLen)
        #print('tot docs:',totalDocsLen)
        getcontext.prec = 100
        priorPositive = Decimal(posDocsLen/totalDocsLen)
        priorNegative = Decimal(negDocsLen/totalDocsLen)
        #print(priorPositive)
        #print(priorNegative)

        ###################
        # take the probablility of single features (each word is its own feature)


        classifyTokens = self.tokenize(sText)
        calculatedProb = self.calculateProbability(classifyTokens, sPosDictionary, sNegDictionary, posDocsLen, negDocsLen, priorPositive, priorNegative)

        #classifyTupleTokens = self.tuplize(self.tokenize(sText))
        #tupleCalculatedProb = self.calculateProbability(classifyTupleTokens, sPosTupleDictionary, sNegTupleDictionary, posDocsLen, negDocsLen, priorPositive, priorNegative)
        classifUpperTokens = self.tokenizeCaps(sText)
        calculatedCapsProb = self.calculateProbability(classifUpperTokens, sPosCapsDictionary, sNegCapsDictionary, posDocsLen, negDocsLen, priorPositive, priorNegative)

        #probSingleDocPos = math.fabs(calculatedProb[0]/100)
        #probSingleDocNeg = math.fabs(calculatedProb[1]/100)

        #probTupleDocPos = math.fabs(tupleCalculatedProb[0]/100)
        #probTupleDocNeg = math.fabs(tupleCalculatedProb[1]/100)

        probDocPos = round((calculatedProb[0] + calculatedCapsProb[0]) / 2,1)
        probDocNeg = round((calculatedProb[1] + calculatedCapsProb[1]) / 2,1)
        #print(calculatedProb)
        #print(tupleCalculatedProb)

        #sVal
        #if (probSingleDocPos < 0):

        #print(probSingleDocPos)
        #print(probSingleDocNeg)

        #probDocPos = math.log10(probSingleDocPos) + math.log10(probTupleDocPos)
        #probDocNeg = math.log10(probSingleDocNeg) + math.log10(probTupleDocNeg)
        # probDocPos = probSingleDocPos * probTupleDocPos
        # probDocNeg = probSingleDocNeg * probTupleDocNeg
        # probDocPos = probTupleDocPos
        # probDocNeg = probTupleDocNeg
        #
        #print(probDocPos)
        #print(probDocNeg)
        #
        if(probDocPos == probDocNeg):
            print('neutral')
        elif(probDocPos > probDocNeg):
            print('positive')
        else:
            print('negative')




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
		sTextWordsOnly = sText.strip(string.punctuation)
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
		#dictionary = dict()
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
        #print(sumPosDictionaryVals)
        #print(sumNegDictionaryVals)
        posFeatureProb = Decimal(0.0)
        negFeatureProb = Decimal(0.0)
        for word in classifyTokens:
            freqWordPos = Decimal((sPosDictionary.get(word,1) + 1) / (sumPosDictionaryVals + posDocsLen))
            #print(freqWordPos)
            #print(math.log10(freqWordPos))
            posFeatureProb += Decimal(math.log10(freqWordPos))
            #print(posFeatureProb)
            freqWordNeg = Decimal((sNegDictionary.get(word,1) + 1)/ (sumNegDictionaryVals + negDocsLen))
            negFeatureProb += Decimal(math.log10(freqWordNeg))

        probDocPos = Decimal(Decimal(math.log10(priorPositive)) + posFeatureProb)
        probDocNeg = Decimal(Decimal(math.log10(priorNegative)) + negFeatureProb)
        #print(probDocPos)
        #print(probDocNeg)
        return (probDocPos,probDocNeg)

    def tokenizeCaps(self, sText):
        '''Given a string of text sText, returns a list of the individual tokens that
        occur in that string (in order).'''
        lTokens = []
        sToken = ""
        for c in sText:
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