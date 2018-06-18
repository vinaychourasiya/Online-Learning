import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA

news_df =  pd.read_csv('news.csv',header=None)
N =news_df.ix[:,:len(news_df.columns)-2]
newsN_df = (N - np.min(N))/(np.max(N)-np.min(N))
T=np.arange(len(news_df))


def News_percept():
    W = []
    W.append(np.zeros(len(news_df.columns)-1))
    P_diffnorm = np.zeros(len(news_df))
    P_index=np.arange(len(news_df))
    for t in range(len(news_df)):
        x_t = news_df.ix[t].values
        y_t=x_t[-1]
        prod=np.inner(x_t[:-1],W[t])
        if prod >=0:
            p_t= 1
        else:
            p_t= -1
        if y_t*prod<=0:
            W.append(W[t] + y_t*x_t[:-1])
        else:
            W.append(W[t])
      
        if LA.norm(W[t]) and LA.norm(W[t+1])!=0:
            
            A= W[t+1]/LA.norm(W[t+1])
            B= W[t]/LA.norm(W[t])
            P_diffnorm[t]= LA.norm((A-B))
    return(P_diffnorm)
    

def News_Winnow():
    eta=.25
    W = []
    W.append((np.ones(len(news_df.columns)-1))/(len(news_df.columns)-1))
    W_index=[]
    W_diffNorm=[]
    
    for t in range(len(news_df)):
        
        x_t = newsN_df.ix[t].values
        a= news_df.ix[t].values
        y_t=a[-1]
        prod=np.inner(x_t,W[t])
        if prod >=0:
            p_t= 1
        else:
            p_t= -1
            
        if y_t*prod <=0:
            Z_t= np.sum(W[t]*np.exp(eta*y_t*x_t))
            A= W[t]*np.exp(eta*y_t*x_t)/Z_t
            W.append(A)
        else:
            W.append(W[t])
        
        if LA.norm(W[t]) and LA.norm(W[t+1]) !=0:
            A= W[t+1]/LA.norm(W[t+1])
            B= W[t]/LA.norm(W[t])
            W_diffNorm.append(LA.norm((A-B)))
            W_index.append(t)
        
    return (W_diffNorm)
            
    
 
perceptronW = News_percept()
WinnowW= News_Winnow()
fig = plt.figure()
plot1=fig.add_subplot(1,2,1,facecolor='white')
plot2=fig.add_subplot(1,2,2,facecolor='white')
plot1.plot(T,perceptronW,'r',label='Perceptron News')
plot2.plot(T,WinnowW,'b',label='Winnow News')

plot1.set_title("Plot for Perceptron")
plot1.set_xlabel("No. of Sample(Example)")
plot1.set_ylabel("Norm variation")
    
plot2.set_title("Plot Winnow")
plot2.set_xlabel("No. of Sample(Example)")
plot2.set_ylabel("Norm variation")

plt.savefig('Q4News')
plt.close()


