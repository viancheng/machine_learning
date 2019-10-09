import numpy as np
from collections import Counter

def get_vocablist(dataset):
	outset=set([])
	for i in dataset:
		outset = outset | set(i)
	return list(outset)

def bagOfword2vec(vocablist,vec):
	outvec=[0]*len(vocablist)
	for i in vec:
		if i in vocablist:
			outvec[vocablist.index(i)] += 1
	return outvec

def trainNB(trainMatrix,trainCategory):
	trainNum=len(trainMatrix)
	wordNum=len(trainMatrix[0])
	classprob={}
	classid=Counter(trainCategory).keys()
	for k,v in Counter(trainCategory).iteritems():
		classprob[k]=float(v)/len(trainCategory)
	pNum=np.ones((len(classid),wordNum));pDenom=[2.0]*len(classid)
	for i in range(trainNum):
		for j in range(len(classid)):
			if trainCategory[i] == classid[j]:
				pNum[j] += trainMatrix[i]
				pDenom[j] += sum(trainMatrix[i])
	outVecList=[]
	for j in range(len(classid)):
		outVecList.append(np.log(pNum[j]/pDenom[j]))
	return classid,outVecList,classprob
	
def classify(testVec,classid,wordProblst,classprob,vocablist):
	p=[]
	newtestVec=np.array(bagOfword2vec(vocablist,testVec),dtype=float)
	for i in range(len(classid)):
		p.append(sum(newtestVec*wordProblst[i])+np.log(classprob[classid[i]]))
	return classid[p.index(max(p))]
	
if __name__ == '__main__':
	lst=[['my','dog','has','flea','problems','help','please'],['maybe','not','take','him','to','dog','park','stupid'],['my','dalmation','is','so','cute','I','love','him'],['stop','posting','stupid','worthless','garbage'],['mr','licks','ate','my','steak','how','to','stop','him'],['quit','buying','worthless','dog','food','stupid']]
	classvec=[0,1,0,1,0,1]
	vocablist=get_vocablist(lst)
	trainMat=[]
	for i in lst:
		trainMat.append(bagOfword2vec(vocablist,i))
	a,b,c=trainNB(trainMat,classvec)
	result=classify(["stupid","garbage","dog"],a,b,c,vocablist)
	print(result)
