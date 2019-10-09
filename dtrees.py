from math import log
import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt

def classify(tree,labels,testvec):
	"""using trained tree and labels to predict testvec result"""
	testvec=map(str,testvec)
	str1=tree.keys()[0]
	seconddic=tree[str1]
	labelidx=labels.index(str1)
	classlabel=""
	for k in seconddic.keys():
		if testvec[labelidx] == k:
			if type(seconddic[k]).__name__ == 'dict':
				classlabel=classify(seconddic[k],labels,testvec)
			else:
				classlabel=seconddic[k]
	return classlabel

def get_depth(tree):
	"""leaf number is x-axis of tree plot, depth is y-axis if tree plot"""
	maxdepth=0
	first=tree.keys()[0]
	second=tree[first]
	for k in second.keys():
		if type(second[k]).__name__ == 'dict':
			thisdepth=1+get_depth(second[k])
		else:
			thisdepth=1
		if thisdepth > maxdepth:
			maxdepth=thisdepth
	return maxdepth

def get_leafnum(tree):
	"""leaf number is x-axis of tree plot, depth is y-axis if tree plot"""
	leafnum=0
	first=tree.keys()[0]
	second=tree[first]
	for k in second.keys():
		if type(second[k]).__name__ == 'dict':
			leafnum+=get_leafnum(second[k])
		else:
			leafnum+=1
	return leafnum

def plotNode(text,centerpos,parentpos,nodetype,arrow):
	createPlot.ax1.annotate(text,xy=parentpos,xycoords='axes fraction',\
	xytext=centerpos,textcoords='axes fraction',\
	va="center",ha="center",bbox=nodetype,arrowprops=arrow)

def plotMidtext(centerpos,parentpos,text):
	x=(float(parentpos[0])-float(centerpos[0]))/2.0+centerpos[0]
	y=(float(parentpos[1])-float(centerpos[1]))/2.0+centerpos[1]
	createPlot.ax1.text(x,y,text)

def plotTree(tree,parentpos,nodetext):
	dnode=dict(boxstyle="sawtooth",fc=[1,0,0])
	leafnode=dict(boxstyle="round4",fc=[1,1,0])
	arrow=dict(arrowstyle="<-")
	
	x=float(get_leafnum(tree))
	y=float(get_depth(tree))
	str1=tree.keys()[0]
	pos1=(plotTree.xOff+(1.0+x)/2.0/plotTree.totalW,plotTree.yOff)
	plotMidtext(pos1,parentpos,nodetext)
	plotNode(str1,pos1,parentpos,dnode,arrow)
	seconddic=tree[str1]
	plotTree.yOff=float(plotTree.yOff-1.0/plotTree.totalD)
	for k in seconddic.keys():
		if type(seconddic[k]).__name__ == 'dict':
			plotTree(seconddic[k],pos1,str(k))
		else:
			plotTree.xOff+=1.0/plotTree.totalW
			plotNode(seconddic[k],(plotTree.xOff,plotTree.yOff),pos1,leafnode,arrow)
			plotMidtext((plotTree.xOff,plotTree.yOff),pos1,str(k))
	plotTree.yOff+=1.0/plotTree.totalD

def createPlot(tree):
	fig=plt.figure(1,facecolor='white')
	fig.clf()
	axprops=dict(xticks=[],yticks=[])
	createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
	plotTree.totalW=float(get_leafnum(tree))
	plotTree.totalD=float(get_depth(tree))
	plotTree.xOff=-0.5/plotTree.totalW; plotTree.yOff=1.0;
	plotTree(tree,(0.5,1.0),"")
	plt.show()

def cal_shang(dataset):
	samp_num=len(dataset)
	shang=0.0
	labels=dataset[:,-1]
	labelcount={}
	for label in labels:
		labelcount[label]=labelcount.get(label,0)+1
	for label in labelcount:
		prob = float(labelcount[label])/samp_num
		shang -= prob*log(prob,2)
	return shang

def splitdataset(dataset,axis,value):
	feature=dataset[:,axis].tolist()
	redataset=[]
	for i,f in enumerate(feature):
		if f == value:
			lst=dataset[i,:axis].tolist()
			lst.extend(dataset[i,axis+1:].tolist())
			redataset.append(lst)
	return np.array(redataset)

def find_best_feature(dataset):
	"""return the best feature index"""
	feature_num=len(dataset[0])-1
	shang1=cal_shang(dataset)
	bestgain=0.0
	bestfeature=-1
	for i in range(feature_num):
		val=[ex[i] for ex in dataset]
		unival=list(set(val))
		shang2=0.0
		for j in unival:
			subset=splitdataset(dataset,i,j)
			prob=len(subset)/float(len(dataset))
			shang2 += prob * cal_shang(subset)
		gain=shang1-shang2
		if (gain > bestgain):
			bestgain = gain
			bestfeature = i
	return bestfeature

def choosebest(lst):
	dic={}
	for i in lst:
		dic[i]=dic.get(i,0)+1
	sortdic=sorted(dic.iteritems(),key=operator.itemgetter(1),reverse=True)
	return sortdic[0][0]

def creatTree(dataset,labels):
	classlist=[ex[-1] for ex in dataset]
	if len(set(classlist)) == 1:
		return classlist[0]
	if len(dataset[0]) == 1:
		return choosebest(classlist)
	best_feature=find_best_feature(dataset)
	best_featlabel=labels[best_feature]
	mytree={best_featlabel:{}}
	del(labels[best_feature])
	feature_value=[ex[best_feature] for ex in dataset]
	uniq_featval=set(feature_value)
	for v in uniq_featval:
		sublabel=labels[:]
		mytree[best_featlabel][v]=creatTree(splitdataset(dataset,best_feature,v),sublabel)
	return mytree

def storeTree(tree,filename):
	import pickle
	fw=open(filename,'wb')
	pickle.dump(tree,fw)
	fw.close()

def grepTree(filename):
	import pickle
	fr=open(filename,'rb')
	return pickle.load(fr)
	

if __name__ == '__main__':
	dataset=np.loadtxt("dtreedata.txt",str,delimiter='\t')
	mytree=creatTree(dataset,['flippers','surface'])
	print(mytree)
	storeTree(mytree,"treeStore.txt")
	mytree=grepTree("treeStore.txt")
	myresult=classify(mytree,['flippers','surface'],[1,0])
	print("when input is [1,0], the result of prediction is: "+myresult)
#	creatPlot(mytree)
	
