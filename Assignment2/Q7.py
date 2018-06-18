import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rnd

np.random.seed(500)

dataset1 = pd.DataFrame([np.random.normal(-2,0.5,500) for j in range(50000)])
dataset1[500]=np.ones(50000)
dataset1[501]=np.ones(50000)*-1
 
dataset2= pd.DataFrame([np.random.normal(-10,0.25,500) for j in range(50000)])
dataset2[500]=np.ones(50000)
dataset2[501]=np.ones(50000)

SampleAll= pd.concat([dataset1,dataset2],ignore_index=True)
np.random.shuffle(SampleAll.values)
#print(SampleAll.head())
T = np.arange(1,len(SampleAll)+1)

def OnlineDescent():
    W = []
    W.append(np.zeros(len(SampleAll.columns)-1))
    R = np.zeros(len(SampleAll))
    
    for t in range(len(SampleAll)):
        A = SampleAll.ix[t].values
        x_t = A[:-1]
        y_t=A[-1]
        prod=np.inner(W[t],x_t)
        cost= max(0,1 - y_t*prod)
        
        if t==0:
            R[t]=cost
        else:
            R[t]=R[t-1]+cost
        if prod >=0:
            p=1
        else:
            p=-1
        
        if p != y_t:
            W.append(W[t] + y_t*x_t)
        else:
            W.append(W[t])
    return (R)

def OnlineMirror():
    W = []
    eta=.25
    W.append(np.zeros(len(SampleAll.columns)-1))
    R = np.zeros(len(SampleAll))
   
    for t in range(len(SampleAll)):
        A = SampleAll.ix[t].values
        x_t = A[:-1]
        y_t=A[-1]
        prod=np.inner(W[t],x_t)
        cost= max(0,1 - y_t*prod)
        if t==0:
            R[t]=cost
        else:
            R[t]=R[t-1]+cost
        if prod >=0:
            p=1
        else:
            p=-1
        
        if p != y_t:
            W.append(W[t]*np.exp(eta*y_t*x_t))
        else:
            W.append(W[t])
    return (R)
    
    
    
    
    


   
R_OMD=OnlineMirror()
R_OGD=OnlineDescent()
#T=np.log(T)
#R_OMD=np.log(R_OMD)
#R_OGD=np.log(R_OGD)
plt.plot(T,R_OGD,label='OGD')
plt.plot(T,R_OMD,label='OMD')
plt.title('Plot of Regret vs T for OGD and OMD')
plt.xlabel('T')
plt.ylabel('Cumulative Regret')
plt.legend(loc='center right', numpoints=1)
plt.savefig('Q7')




           



