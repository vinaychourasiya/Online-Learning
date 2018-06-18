import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

d=10
T=10**5           
np.random.seed(123456)
def WeightedMajority(d,T,etha):
    n=20
    WeightedMajority_regret = []
    for i in range(n):
        loss=0
        weights = np.ones((d))
        lossMatrix=np.empty((T,d))
        for t in range(T):
            # probability of experts 
            prob_weights = weights/(np.sum(weights))
            # given distribution parameters
            ExpertsDist = np.array([.5,.5,.5,.5,.5,.5,.5,.5,.4,.6])
            if t>T/2:
                ExpertsDist[-1]=0.3
            
            # loss vector Bernoulii distribution
            CostVector = np.random.binomial(1,p=ExpertsDist)
            lossMatrix[t]=CostVector   
            
            k=np.random.choice(CostVector,p=prob_weights) 
            loss+=k

            # update weights 
            weights=weights*np.exp(-etha*CostVector)
        # min through one expert
        MinExprtLoss=min(np.sum(lossMatrix, axis=0))
        WeightedMajority_regret.append(loss-MinExprtLoss)


    print("Regret",np.mean(WeightedMajority_regret))
    WeightedMajority_regret.append(etha)
    
    return (WeightedMajority_regret)


c=np.arange(.1,2.2,.2)      
ethaVector=c*(np.sqrt((2*np.log(d))/T))
W_regret=[]

for k in range(len(ethaVector)):
    R=WeightedMajority(d, T, ethaVector[k])
    W_regret.append(R)

eta = []
regret_mean = []
regret_err = []
freedom_degree = len(W_regret[0]) - 2
for regret in W_regret:
    eta.append(regret[-1])
    regret_mean.append(np.mean(regret[:-1]))
    regret_err.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[:-1]))


colors = list("rgbcmyk")
shape = ['--^', '--d', '--v']
plt.errorbar(eta, regret_mean, regret_err, color=colors[0])
plt.plot(eta, regret_mean, colors[0] + shape[0], label='Weighted_Majority')

plt.legend(loc='upper right', numpoints=1)
plt.title("Pseudo Regret vs Learning Rate for T = 10^5 and 20 Sample paths")
plt.xlabel("Learning Rate")
plt.ylabel("Pseudo Regret")
plt.savefig("WeightedMajority.png", bbox_inches='tight')
plt.close()





