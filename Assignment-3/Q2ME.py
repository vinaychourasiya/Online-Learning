import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

def ME(K):
    ArmMeans = [3/4]
    for i in range(2,K+1):
        ArmMeans.append(3/4-i/40)
    
    d=0.1/2
    arms=np.arange(K)
    e=.01/4
    S=arms
    BlockSample=[] #list for collection the each block sample(log(K))
    while (len(S)>1):
        Samples=[] 
        SampleSize = int((1/((e/2)**2))*np.log(3/d)) # sample size (n)
        BlockSample.append(len(S)*SampleSize)
        for i in S:
            x = np.random.binomial(1,ArmMeans[i],SampleSize)
            Samples.append(x)
        Mean=np.mean(Samples,axis=1)
        Median=np.median(Mean)
        NewVec= list(zip(Mean,S)) #vector of tuple of  Mean and arm 
        temp=[] # temp list for updating the arm accounding the median
        for m,arm in NewVec:
            if m>=Median:
                temp.append(arm)
        S=temp
        e=(3/4)*e
        d=d/2
    
    return (sum(BlockSample))

M=[]
K=[5,10,15,20,25]
for k in K:
    M.append(ME(k))
#sns.set()
plt.figure(dpi=100)
plt.plot(K,M,marker='o',label='Median Elimination (ME)')
plt.title(" Plot of sample complexity v/s number of arms for ME")
plt.ylabel("Sample Complexity")
plt.xlabel("number of arms")
plt.legend(loc='upper left', numpoints=1)
plt.savefig('ME')

    
                
                
            
            
            
            
            
        
