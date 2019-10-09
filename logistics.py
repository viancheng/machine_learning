import numpy as np

def sigmoid(x):
	return 1.0/(1+exp(-x))

def gradAscent(dataMat,labels,iterNum=150):
	m,n=np.shape(dataMat)
	weight=np.ones(n)
	for j in range(iterNum):
		dataIndex=range(m)
		for i in range(m):
			alpha=4/(1.0+j+i)+0.01
			randIndex=int(np.random.uniform(0,len(dataIndex)))
			h=sigmoid(np.sum(dataMat[randIndex]*weight))
			error=labels[randIndex]-h
			weight=weight+alpha*error*dataMat[randIndex]
			del (dataIndex[randIndex])
	return weight

def plotBestFit(dataMat,labels,weight):
	import matplotlib.pyplot as plt
	dataArr=np.array(dataMat)
	n=np.shape(dataArr)[0]
	x1=[];y1=[]
	x2=[];y2=[]
	for i in range(n):
		if int(labels[i]) == 1:
			x1.append(dataArr[i,1]);y1.append(dataArr[i,2])
		else:
			x2.append(dataArr[i,1]);y2.append(dataArr[i,2])
	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.scatter(x1,y1,s=30,c='red',marker='s')
	ax.scatter(x2,y2,s=30,c='green')
	x=np.arange(-3.0,3.0,0.1)
	y=(-weight[0]-weight[1]*x)/weight[2]
	ax.plot(x,y)
	plt.xlabel('x1');plt,ylabel('x2')
	plt.show()

def classify(inputlst,weight):
	prob=sigmod(np.sum(inputlst*weight))
	if prob > 0.5:
		return 1.0
	else:
		return 0.0
