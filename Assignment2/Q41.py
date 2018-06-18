import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA


np.random.seed(1000)

skin_df =  pd.read_csv('skin.csv',header=None)
T=np.arange(len(skin_df))

def Skin_percept():
    W = []
    P_diffnorm = np.zeros(len(skin_df))
    W.append(np.zeros(3))
    for t in range(len(skin_df)):
        x_t = skin_df.ix[t].values
        prod=np.inner(x_t[:-1],W[t])
        if prod >=0:
            p_t= 1
        else:
            p_t= -1
        if x_t[-1]*prod<=0:
            W.append(W[t] + x_t[-1]*x_t[:-1])
        else:
            W.append(W[t])
      
        if LA.norm(W[t]) and LA.norm(W[t+1])!=0:
            
            A= W[t+1]/LA.norm(W[t+1])
            B= W[t]/LA.norm(W[t])
            P_diffnorm[t]= (LA.norm((A-B)))
    return(P_diffnorm)

def Skin_Winnow():
    eta=.25
    W = []
    W.append((np.ones(len(skin_df.columns)-1))/(len(skin_df.columns)-1))
    W_diffNorm=np.zeros(len(skin_df))
    
    for t in range(len(skin_df)):
        
        x_t = skin_df.ix[t].values
        prod=np.inner(x_t[:-1],W[t])
        if prod >=0:
            p_t=1
        else:
            p_t= -1
            
        if x_t[-1]*prod<=0:
            Z_t= np.sum(W[t]*np.exp(eta*x_t[-1]*x_t[:-1]))
            A= (W[t]*np.exp(eta*x_t[-1]*x_t[:-1]))/Z_t
            W.append(A)
        else:
            W.append(W[t])
        
        if LA.norm(W[t])!=0 and LA.norm(W[t+1])!=0:
            A= W[t+1]/LA.norm(W[t+1])
            B= W[t]/LA.norm(W[t])
            W_diffNorm[t]=(LA.norm((A-B)))
        
        
    return (W_diffNorm)           

PDiff=Skin_percept()
WDiff=Skin_Winnow()

fig = plt.figure()
plot1=fig.add_subplot(1,2,1,facecolor='white')
plot2=fig.add_subplot(1,2,2,facecolor='white')
plot1.plot(T,PDiff,'r',label='Perceptron Skin')
plot2.plot(T,WDiff,'b',label='Winnow Skin')

plot1.set_title("Plot for Perceptron")
plot1.set_xlabel("No. of Skin data Example")
plot1.set_ylabel("Norm variation")
    
plot2.set_title("Plot Winnow")
plot2.set_xlabel("No. of Skin data Example")
plot2.set_ylabel("Norm variation")

plt.savefig('Q4Skin')
#plt.show()
                    
                   
                   
    
    
