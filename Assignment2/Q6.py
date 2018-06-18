import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as LA


np.random.seed(1000)
st1=np.sqrt(.5)
sample1 = pd.DataFrame([(np.random.normal(-2,st1),np.random.normal(-2,st1)) for j in range(1000)])
sample1[2]=np.ones(1000)
sample1[3]=np.ones(1000)*-1

st2=np.sqrt(.25)
sample2= pd.DataFrame([(np.random.normal(-10,st2),np.random.normal(-10,st2)) for j in range(1000)])
sample2[2]=np.ones(1000)
sample2[3]=np.ones(1000)


SampleAll= pd.concat([sample1,sample2])
SampleAll.index = np.arange(2000)
np.random.shuffle(SampleAll.values)


def Perceptron():
    W = []
    mistakes=0
    W.append(np.zeros(len(SampleAll.columns)-1))
    for t in range(len(SampleAll)):
        x_t = SampleAll.ix[t].values
        prod=np.inner(x_t[:-1],W[t])
        if prod >= 0:
            p_t=1
        else:
            p_t= -1
        if x_t[-1]*prod <=0:            
            W.append(W[t] + x_t[-1]*x_t[:-1])
            mistakes+=1
        else:
            W.append(W[t])

    W_star=W[-1]
    Margin=np.zeros(len(SampleAll))

    for j in range(len(SampleAll)):
        x=SampleAll.ix[j].values
        Margin[j] = (np.inner(W_star,x[:-1]))*x[-1]
    Gamma = np.min(Margin)

    A=np.arange(3)
    X = SampleAll[A]
    R = np.max(LA.norm(X,axis=1))
    W_norm =  LA.norm(W_star[:-1])

    MBound = ((R**2)*(W_norm**2))/(Gamma**2)
    print("Estimated margin :",Gamma)
    print("Mistake by Perceptron Algorithm :", mistakes)
    print('\n')
    print("Mistake Bound using Formula for Perceptron:", MBound)

   

def Winnow():
    etaVec=np.arange(0.05,.51,.05)
    mistakes = np.zeros(len(etaVec))
    for i in range(len(etaVec)):
        W = []
        m=0
        W.append((np.ones(len(SampleAll.columns)-1))/(len(SampleAll.columns)-1))
        for t in range(len(SampleAll)):
           x_t = SampleAll.ix[t].values
           prod=np.inner(x_t[:-1],W[t])

           if prod >=0:
               p_t=1
           else:
               p_t= -1

           if x_t[-1]*prod<=0:
               Z_t= np.sum(W[t]*np.exp(etaVec[i]*x_t[-1]*x_t[:-1]))
               A= (W[t]*np.exp(etaVec[i]*x_t[-1]*x_t[:-1]))/Z_t
               W.append(A)
               m+=1
           else:
               W.append(W[t])
        mistakes[i]=m
    A=np.arange(3)
    X = SampleAll[A]      
    etaVec=np.arange(0.05,.5,.05)
    W_star = W[-1]
    Margin=np.zeros(len(SampleAll))
    WsNorm = LA.norm(W_star,ord=1)
    for j in range(len(SampleAll)):
        x=SampleAll.ix[j].values
        Margin[j] = (np.inner(W_star,x[:-1]))*x[-1]
    Gamma = np.min(Margin)
    R = np.max(LA.norm(X,axis=1))
    print('\n')
    print(" No. of Mistakes by Winnow Algorithm ")
    for i in range(len(etaVec)):
        print('eta Value: '+str(etaVec[i])+"  Mistakes :", str(mistakes[i]))
    
    Mbound = (WsNorm*np.log(len(W_star)))/(etaVec*Gamma - WsNorm*np.log((np.exp(etaVec*R))-np.exp(-etaVec*R)/2))


Perceptron()
Winnow()


