import os,sys
import numpy as np

def autoNorm(dataset):
	minv=dataset.min(0)
	maxv=dataset.max(0)
	ranges=maxv-minv
	normDataset=np.zeros(np.shape(dataset))
	m=dataset.shape[0]
	normDataset=dataset-np.tile(minv,(m,1))
	normDataset=normDataset/np.tile(ranges,(m,1))
	return normDataset,ranges,minv

def classify(x,files,k):
	"""classify0(x,file,k):
x : your query vector for prediction, len(x) == len(col(file))-1;
file : your dataset for taining, no header, last col is label;
k : choose the best result in k range;
PS : this classfiy is not good for large set!"""
	arr=np.loadtxt(files,str,delimiter='\t')
	dataset,labels=sepdata(arr)
	normDataset,ranges,minv=autoNorm(dataset)
	dataset=normDataset
	datasetSize=dataset.shape[0]
	diffmat=np.tile(x,(datasetSize,1))-dataset
	sqdiffmat=diffmat**2
	sqsum=sqdiffmat.sum(axis=1)
	distance=sqsum**0.5
	sortDistance=np.argsort(distance)
	classcount={}
	for i in range(k):
		vote=labels[sortDistance[i]]
		classcount[vote]=classcount.get(vote,0)+1
	sortclass=sorted(classcount.iteritems(),key=lambda item:item[1],reverse=True)
	return sortclass[0][0]

def sepdata(datasets):
	dataset=datasets[:,0:-1].astype(np.float32)
	labels=datasets[:,-1]
	return dataset,labels

