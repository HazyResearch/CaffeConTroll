#!/usr/bin/env python



__author__ = 'igor1'
#run: mpirun -np 2 python sim1.py : -np 1 ./caffe-ct 200 3 32 32 6 3 2 1 : -np 1 ./caffe-ct 200 6 16 16 24 3 2 1 : -np 3 python sim1.py | grep error
#run: mpirun -np 7 python sim1.py | grep error
from mpi4py import MPI
import time
import sys
import numpy as np
from collections import namedtuple
from operator import attrgetter
from matplotlib import pyplot as plt
import random





#Header0=namedtuple('Header0',['rank', 'request','batchID','dataType','requestNum','responseNum','nextHeader'],verbose=False)
#Header0.__new__.__defaults__ = (0,0,0) #set default requestNum to 0
#Header1=namedtuple('Header0',['C_start', 'C_end','R_start','R_end','D_start','D_end','M_start','M_end','evict','nextHeader'],verbose=False)
#Header1.__new__.__defaults__ = (,0) #set default requestNum to 0


#class Message(object):
#    def __init__(self,rank,request,batchID,dataType,requestNum=0,responseNum=0,nextHeader=0):
#        self.header0=Header0(rank=rank,request=request,batchID=batchID,dataType=dataType,requestNum=requestNum,responseNum=responseNum,nextHeader=nextHeader)
#        self.foo="goo"
#    def __iter__(self):
#        return iter(list(self.header0))
#    def __str__(self):
#        return "rank="+str(self.header0.rank)+" request="+requestStr[self.header0.request]+" batchID="+str(self.header0.batchID)+" dataType="+dataTypeStr[self.header0.dataType]

#    def __getattr__(self, name): #hack to allow access to msg.header.someAttribute using the incorrect syntax "msg.someAttribute" . Might be dangerous
#        if hasattr(self.header0,name):
#            return getattr(self.header0, name)
#        elif hasattr(self,'header1') and hasattr(self.header1,name):
#            return getattr(self.header1, name)
#        else:
#            raise Exception("Message has no attribute "+name)



Message= namedtuple('Message',['rank', 'request','batchID','dataType','requestNum','nextHeader'],verbose=False)
Message.__new__.__defaults__ = (0,0) #set default requestNum to 0

#  class Message(_Message):


# {rank, requestType, batchID, dataType, metaData}
#rank - in natruals
#dataType -label, data, grad, weight
# {rank, requestType, batchID, dataType, metaData}
#list(Message): array of 6 ints
#rank - int
#requests: int:
[EVICT,FEED,SYNC,KILL]=[0,1,2,3 ]
requestStr=['EVICT','FEED','SYNC']
#batchID - int
#dataType - int:
[LABELS,FEATURES,GRADS,MODEL,BIAS]=[0,1,2,3,4]
dataTypeStr=['LABELS','FEATURES','GRADS','MODEL','BIAS']
#metaData - int
SIZE=1024
MSG_SIZE=6


#########################################################################################################################

class Cube(object):
    def __init__(self,shape,dataType,batchID=-1):
        self.shape=shape
        self.dataType=dataType
        #print shape
        self.data=np.zeros(shape).astype('<f32')
        self.batchID=batchID

#TODO: Make Cube a subclass of np.array that also has a dataType field  and a normalized random init method, and is always float32


    def __str__(self):
        return dataTypeStr[self.dataType]+":"+str(self.shape)
#########################################################################################################################

class ReactiveBridge(object): # All dict implementation
    def __init__(self,comm,rank,batchSize,inDims,outDims,cacheSize=1):
        self.comm=comm
        self.rank=rank
        self.cacheSize=cacheSize
        self.inFeatures=Cube([batchSize]+list(inDims),FEATURES)
        self.inGrads=Cube([batchSize]+list(outDims),FEATURES)
        self.outFeaturesDict={}
        self.outGradsDict={}
        self.cacheDict={}
        for i in range(cacheSize):
            msg=Message(rank=self.rank,request=FEED,batchID=-1,dataType=FEATURES)
            send(msg,dest=0, tag=1)
        self.requestNum=0

    def runLoop(self):
        while(1):
            msg= recv(source=0,tag=1)
            #print self.rank,":got message",msg
            if msg.request==FEED and msg.dataType==FEATURES:
                self.feedFeaturesToRemote(msg)
            elif msg.request==FEED and msg.dataType==GRADS:
                self.feedGradsToRemote(msg)
            elif msg.request==EVICT and msg.dataType==FEATURES:
                self.evictFeaturesFromRemote(msg)
            elif msg.request==EVICT and msg.dataType==GRADS:
                self.evictGradsFromRemote(msg)
            elif msg.request=='kill':
                exit()
            else:
                print self.rank,": Unknown request",msg

    def evictFeaturesFromRemote(self,msg):
        batchID=msg.batchID
        src=msg.rank
        cube=self.inFeatures
        self.comm.Recv(cube.data,source=src,tag=2)
        outFeatures,cache=self.forward(cube.data)
        #print self.rank,": outFeatures.shape=",outFeatures.shape
        self.cacheDict[batchID]=cache
        self.outFeaturesDict[batchID]=outFeatures
        msg=Message(rank=self.rank,request=EVICT,batchID=batchID,dataType=FEATURES)
        send(msg,dest=0, tag=1)

    def feedFeaturesToRemote(self,msg):
        batchID=msg.batchID
        dst=msg.rank
        data=self.outFeaturesDict.pop(batchID)
        self.comm.Send(data,dest=dst,tag=2)
        msg=Message(rank=self.rank,request=FEED,batchID=batchID,dataType=GRADS)
        send(msg,dest=0, tag=1)

    def evictGradsFromRemote(self,msg):
        batchID=msg.batchID
        src=msg.rank
        cube=self.inGrads
        self.comm.Recv(cube.data,source=src,tag=2)
        cache=self.cacheDict.pop(batchID)
        #print self.rank,": self.inGrads.data.shape=",self.inGrads.data.shape
        outGrads=self.backward(cube.data,cache)
        self.outGradsDict[batchID]=outGrads
        #print self.rank,":self. outGradsDict.keys()=",self.outGradsDict.keys()
        msg=Message(rank=self.rank,request=EVICT,batchID=batchID,dataType=GRADS)
        send(msg,dest=0, tag=1)

    def feedGradsToRemote(self,msg):
        batchID=msg.batchID
        dst=msg.rank
        data=self.outGradsDict.pop(batchID)
        self.comm.Send(data,dest=dst,tag=2)
        msg=Message(rank=self.rank,request=FEED,batchID=-1,dataType=FEATURES)
        send(msg,dest=0, tag=1)

    def forward(self,X):
        Exception("Not implemented")

    def forward(self,dY,cache):
        Exception("Not implemented")

#########################################################################################################################

class ReactiveFCBridge(ReactiveBridge):

    def __init__(self,comm,rank,batchSize,inDims,outDims,cacheSize=1):
        assert (len(outDims)==1)
        super(ReactiveFCBridge, self).__init__(comm,rank,batchSize,inDims,outDims,cacheSize)
        self.WCube=Cube([np.product(inDims)]+outDims,'weights')
        self.bCube=Cube([1]+outDims,'bias')
        self.WCube.data=np.random.random(list(inDims)+list(outDims)).astype('float32')
        self.WCube.data/=np.sum(self.WCube.data)
        self.counter=0
        self.lr=5e-7

    def forward(self,X):
        #time.sleep(0.1)
        x=X.copy().reshape([X.shape[0],-1])
        Y=x.dot(self.WCube.data)+self.bCube.data  ##TODO: Finish this
        return Y,x

    def backward(self,dY,x):
        #time.sleep(0.1)
        reg= 5e4

        #print self.rank,":dY.shape=",dY.shape
        #print self.rank,":x.shape=",x.shape
        self.counter+=1
        if self.counter%1000==0 and self.counter>0:
            self.lr*=0.1

        dX=dY.dot(self.WCube.data.T)
        dx=self.inGrads.data.dot(self.WCube.data.T)
        dW=x.T.dot(self.inGrads.data) +reg*self.WCube.data
        db=np.sum(self.inGrads.data,axis=0)

        self.WCube.data-=self.lr*dW
        self.bCube.data-=self.lr*db
        return dY

#########################################################################################################################

class ReactiveConvReluPoolBridge(ReactiveBridge):

    def __init__(self,comm,rank,batchSize,inDims,outDims,cacheSize=1):
        super(ReactiveFCBridge, self).__init__(comm,rank,batchSize,inDims,outDims,cacheSize)
        self.WCube=Cube([np.product(inDims)]+outDims,'weights')
        self.bCube=Cube([1]+outDims,'bias')
        self.WCube.data=np.random.random(list(inDims)+list(outDims)).astype('float32')
        self.WCube.data/=np.sum(self.WCube.data)


    def forward(self,X):
        #time.sleep(0.1)
        x=X.copy().reshape([X.shape[0],-1])
        Y=x.dot(self.WCube.data)+self.bCube.data  ##TODO: Finish this
        return Y,x

    def backward(self,dY,x):
        #time.sleep(0.1)
        reg= 5e4
        lr=5e-7
        #print self.rank,":dY.shape=",dY.shape
        #print self.rank,":x.shape=",x.shape

        dX=dY.dot(self.WCube.data.T)
        dx=self.inGrads.data.dot(self.WCube.data.T)
        dW=x.T.dot(self.inGrads.data) +reg*self.WCube.data
        db=np.sum(self.inGrads.data,axis=0)

        self.WCube.data-=lr*dW
        self.bCube.data-=lr*db
        return dY

#########################################################################################################################
class ReactiveDataBridge(ReactiveBridge):

    def __init__(self,comm,rank,batchSize,inDims,outDims,X,y,cacheSize=1):
        #print "ReactiveDataBridge: X.shape[1]=",X.shape[1]," outDims[0]=",outDims[0]
        assert(X.shape[1]==outDims[0])
        self.comm=comm
        self.rank=rank
        self.cacheSize=cacheSize
        self.inFeatures=Cube([batchSize]+list(inDims),FEATURES)
        self.inGrads=Cube([batchSize]+list(outDims),FEATURES)
        self.batchID=0
        self.totalBatches=int(X .shape[1]/batchSize)
        self.requestNum=0
        self.labelsDict={}
        self.batchSize=batchSize
        self.X=X
        self.y=y
        for i in range(cacheSize):
            msg=Message(rank=self.rank,request=EVICT,batchID=self.batchID,dataType=FEATURES)
            send(msg,dest=0, tag=1)
            self.batchID=(self.batchID+1)% self.totalBatches




    def feedFeaturesToRemote(self,msg):
        batchID=msg.batchID
        dst=msg.rank
        offset=batchID % (self.totalBatches)
        #print "offset=",offset,"self.batchSize=",self.batchSize,"range=",offset*self.batchSize,(offset+1)*self.batchSize
        indexes=range(offset*self.batchSize,(offset+1)*self.batchSize)
        labels=self.y[indexes].reshape([-1,1]).astype('float32').copy()
        self.labelsDict[batchID]=labels
        outFeatures=self.X[indexes,:].astype('float32').copy()
        self.comm.Send(outFeatures,dest=dst,tag=2)
        msg=Message(rank=self.rank,request=FEED,batchID=batchID,dataType=GRADS)
        send(msg,dest=0, tag=1)
        msg=Message(rank=self.rank,request=EVICT,batchID=batchID,dataType=LABELS)
        send(msg,dest=0, tag=1)

    def feedLabelsToRemote(self,msg):
        batchID=msg.batchID
        dst=msg.rank
        offset=batchID % (self.totalBatches)
        indexes=range(offset*self.batchSize,(offset+1)*self.batchSize)
        labels=self.y[indexes].reshape([-1,1]).astype('float32').copy()
        self.labelsDict[batchID]=labels
        labels=self.labelsDict.pop(batchID)
        self.comm.Send(labels,dest=dst,tag=2)

    def evictGradsFromRemote(self,msg):
        batchID=msg.batchID
        src=msg.rank
        cube=self.inGrads
        self.comm.Recv(cube.data,source=src,tag=2)
        self.batchID=(self.batchID+1)
        msg=Message(rank=self.rank,request=EVICT,batchID=self.batchID,dataType=FEATURES)
        send(msg,dest=0, tag=1)


    def runLoop(self):
        while(1):
            msg= recv(source=0,tag=1)
            if msg.request==FEED and msg.dataType==FEATURES:
                self.feedFeaturesToRemote(msg)
            elif msg.request==FEED and msg.dataType==LABELS:
                self.feedLabelsToRemote(msg)
            elif msg.request==EVICT and msg.dataType==GRADS:
                self.evictGradsFromRemote(msg)
            elif msg.request=='kill':
                exit()
            else:
                print self.rank,": Unknown request",msg

#########################################################################################################################
class Bridge(object):

    def __init__(self,comm,rank,batchSize,inDims,outDims):
        self.comm=comm
        self.rank=rank
        self.inCube=Cube([batchSize]+list(inDims),FEATURES)
        self.outCube=Cube([batchSize]+list(outDims), FEATURES)
        self.inGrads=Cube([batchSize]+list(outDims), GRADS) #can probably point to the same array as outCube...
        self.outGrads=Cube([batchSize]+list(inDims), GRADS)
        self.batchID=-1
        self.cache={}

    def runLoop(self):
        while(1):
            self.feed(self.inCube)
            #time.sleep(1)
            self.forward()
            self.evict(self.outCube,self.batchID)
            self.feed(self.inGrads)
            #time.sleep(1)
            self.backward()
            self.evict(self.outGrads,self.batchID)


    def feed(self,cube,batchID=-1):
        msg=Message(rank=self.rank,request=FEED,batchID=batchID,dataType=cube.dataType)
        send(msg,dest=0, tag=1)
        msg= recv(source=0,tag=1)
        #print self.rank,"got message", msg
        if msg.request=="kill":
            exit()
        src=msg.rank
        self.batchID=msg.batchID
        #print self.rank,"feeding data of size",cube.data.shape
        comm.Recv(cube.data,source=src,tag=2)


    def evict(self,cube,batchID):
        msg=Message(rank=self.rank,request=EVICT,batchID=batchID,dataType=cube.dataType)
        send(msg,dest=0, tag=1)
        msg= recv(source=0,tag=1)
        #print self.rank,"got message", msg
        if msg.request=="kill":
            exit()
        dst=msg.rank
        #print self.rank,"evicting data of size",cube.data.shape
        comm.Send(cube.data,dest=dst,tag=2)

    def forward(self):
        print "====Not Implemented====="

    def backward(self):
        print "====Not Implemented====="


#########################################################################################################################
def plotWeights(w):
    from matplotlib import pyplot as plt
    w= w.T.reshape(10,3, 32, 32)
    w=w.transpose(0,2,3,1)
    print "w.shape=",w.shape
    w_min, w_max = np.min(w), np.max(w)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.figure()
    for i in xrange(10):
        plt.subplot(2, 5, i + 1)
        wimg = 255.0 * ((w[i].squeeze() - w_min) / (w_max - w_min))
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])
    plt.show()
#########################################################################################################################

class FCBridge(Bridge):

    def __init__(self,comm,rank,batchSize,inDims,outDims):
        super(FCBridge, self).__init__(comm,rank,batchSize,inDims,outDims,cacheSize=1)
        assert (len(outDims)==1)
        self.WCube=Cube([np.product(inDims)]+outDims,'weights')
        self.bCube=Cube([1]+outDims,'bias')
        self.WCube.data=np.random.random(list(inDims)+list(outDims)).astype('float32')
        self.WCube.data/=np.sum(self.WCube.data)
        self.cache={}


    def forward(self):
        x=self.inCube.data.copy().reshape([self.inCube.shape[0],-1])
        self.cache[self.batchID]=x
        self.outCube.data=x.dot(self.WCube.data)+self.bCube.data
        #if self.batchID%500==0: plotWeights(self.WCube.data.copy())

        #print self.outCube.data

    def backward(self):
        reg= 5e4
        lr=5e-7

        x=self.cache.pop(self.batchID)
        dx=self.inGrads.data.dot(self.WCube.data.T)
        dW=x.T.dot(self.inGrads.data) +reg*self.WCube.data
        db=np.sum(self.inGrads.data,axis=0)

        self.WCube.data-=lr*dW
        self.bCube.data-=lr*db
        self.outGrads.data=np.reshape(dx,self.outGrads.data.shape)

        pass




#########################################################################################################################

class SoftmaxLossBridge(Bridge):

    def __init__(self,comm,rank,batchSize,inDims,outDims):
        super(SoftmaxLossBridge, self).__init__(comm,rank,batchSize,inDims,outDims)
        self.labelsCube=Cube([batchSize]+[1],LABELS)
        print "softmax labels:",self.labelsCube

    def runLoop(self):
        while(1):
            self.feed(self.labelsCube)
            self.feed(self.inCube)

            x=self.inCube.data
            y=np.round(self.labelsCube.data.reshape(-1)).astype('int')
            #print y
            #print self.labelsCube.data
            probs = np.exp(x - np.max(x, axis=1, keepdims=True))
            probs /= np.sum(probs, axis=1, keepdims=True)
            errRate=np.mean(np.abs(1*(np.argmax(probs,axis=1)!=y)))
            N = x.shape[0]
            #print "============"
            #print "N=",N
            #print "============"
            loss = -np.sum(np.log(probs[np.arange(N), y])) / N
            print "batch",self.batchID,"loss:",loss,"error rate=",errRate
            dx = probs.copy()
            dx[np.arange(N), y] -= 1
            dx /= N
            self.outGrads.data=dx.copy()
            self.evict(self.outGrads,self.batchID)




#########################################################################################################################


class SrcBridge(Bridge):
    def __init__(self,comm,rank,batchSize,inDims,outDims):
        super(SrcBridge, self).__init__(comm,rank,batchSize,inDims,outDims)
        self.labelsCube=Cube([batchSize]+[1],LABELS)



    def runLoop(self):
        for ii in range(iterations):
            self.labelsCube.data=np.random.randint(0,2,size=self.labelsCube.data.shape).astype('float32')
            self.outCube.data=(np.random.random(self.outCube.data.shape)+self.labelsCube.data).astype('float32')
            self.evict(self.outCube,batchID=ii)
            self.evict(self.labelsCube,batchID=ii)
            self.feed(self.inGrads)

#########################################################################################################################

class DataBridge(Bridge):
    def __init__(self,comm,rank,batchSize,inDims,outDims,X,y):
        super(DataBridge, self).__init__(comm,rank,batchSize,inDims,outDims)
        self.labelsCube=Cube([batchSize]+[1],LABELS)
        self.batchSize=batchSize
        assert(X.shape[1]==outDims[0])
        self.X=X
        self.y=y
        self.totalBatches=int(X .shape[1]/batchSize)

    def runLoop(self):
        for ii in range(iterations):
            batchNum=ii % self.totalBatches
            indexes=range(batchNum*self.batchSize,(batchNum+1)*self.batchSize)
            self.labelsCube.data=self.y[indexes].reshape([-1,1]).astype('float32').copy()
            self.outCube.data=self.X[indexes,:].astype('float32').copy()
            self.evict(self.outCube,batchID=ii)
            self.evict(self.labelsCube,batchID=ii)
            self.feed(self.inGrads)




#########################################################################################################################
#########################################################################################################################





comm = MPI.COMM_WORLD
rank = comm.Get_rank()

iterations=10000
batchSize=200

#inDims=[10]
#outDims=[2]
#outDims=1
dimensions=[[3072],[1536],[1536],[100],[10]]
numRanks=len(dimensions)+2
#This is code for setting the order of bridges:
nextRank={i:i+1 for i in range(numRanks)}
prevRank={i:i-1 for i in range(numRanks+1)}


def recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG):
    msgArray=np.zeros(MSG_SIZE,dtype='<i4')
    comm.Recv([msgArray, MSG_SIZE,MPI.INT],source=source,tag=tag)
    msg= Message(*list(msgArray))
    #print comm.Get_rank(),"received",msg
    return msg

def send(msg,dest,tag):
    #print comm.Get_rank(),"sending",msg,"to",dest
    #msg=msg._replace(metaData=0)
    msgArray=np.array(list(msg),dtype='<i4')

    comm.Send([msgArray, MSG_SIZE,MPI.INT],dest=dest,tag=tag)



def findMatch(msg,requests):
    if msg.dataType==FEATURES :
        wantedRank= nextRank[msg.rank] if msg.request==EVICT else prevRank[msg.rank]
    elif msg.dataType==GRADS:
        wantedRank= prevRank[msg.rank] if msg.request==EVICT else nextRank[msg.rank]
    elif msg.dataType==LABELS:
        if msg.request==FEED:
            wantedRank=1
        else:
            wantedRank=numRanks-1
    else:
        Exception ("unknown datatype " + msg.dataType)

    #print "wantedRank is ",wantedRank
    match=next((x for x in requests if (x.rank==wantedRank and (x.batchID==msg.batchID or x.batchID == -1 or msg.batchID==-1)\
                                        and x.dataType==msg.dataType)),None)  #TODO: check that dimensions match
    #print "match is",match
    return match

if rank == 0:
    print "rank",rank,": I am the scheduler"
    evictReqs=[]
    feedReqs=[]
    for i in range(iterations*100):

        #msg = comm.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG)
        msg=recv()

        #print "========================"
        #print "Sched received:",msg
        #print "evictReqs=",evictReqs
        #   print "feedReqs=",feedReqs
        if msg.request==FEED:
            feedMsg=msg
            evictMsg=findMatch(feedMsg,evictReqs)
            if evictMsg is not None:
                evictReqs.remove(evictMsg)
                #print "matched with msg:",evictMsg
            else:
                feedReqs+=[feedMsg]
        elif msg.request==EVICT:
            evictMsg=msg
            feedMsg=findMatch(evictMsg,feedReqs)
            if feedMsg is not None:
                feedReqs.remove(feedMsg)
                #print "matched with msg:",feedMsg
            else:
                evictReqs+=[evictMsg]
                evictReqs=sorted(evictReqs, key=attrgetter('batchID'))
        else:
            print "Warning! Unknown message type:",msg
        if feedMsg is not None and evictMsg is not None:

            print evictMsg.rank,"-----("+dataTypeStr[evictMsg.dataType]+"_"+str(evictMsg.batchID)+")---->",feedMsg.rank
            #print "evictMsg=",evictMsg
            send(evictMsg, dest=feedMsg.rank,tag=1)
            #print "feedMsg=",feedMsg
            send(feedMsg._replace(batchID=evictMsg.batchID),dest=evictMsg.rank,tag=1)


elif rank ==1:
    from data_utils import get_CIFAR10_data_rows
    X_train, y_train, _, _, _, _=get_CIFAR10_data_rows()
    #plotWeigts(X_train[:10,...])
    print "rank",rank,": I am the ReactiveDataBridge"
    bridge=ReactiveDataBridge(comm,rank,batchSize=batchSize,inDims=[0],outDims=dimensions[0],X=X_train,y=y_train,cacheSize=1)
    bridge.runLoop()
elif rank ==numRanks-1:
    print "rank",rank,": I am the SoftmaxLossBridge"
    bridge=SoftmaxLossBridge(comm,rank,batchSize=batchSize,inDims=dimensions[-1],outDims=[0])
    bridge.runLoop()
elif rank >1 and rank <numRanks-1:
    print "rank",rank,": I am the ReactiveFCBridge"
    bridge=ReactiveFCBridge(comm,rank,batchSize=batchSize,inDims=dimensions[rank-2],outDims=dimensions[rank-1],cacheSize=1)
    bridge.runLoop()

