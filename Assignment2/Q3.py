import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss


d=10
T=10**5        
np.random.seed(123)
eta_star =np.sqrt((2*np.log(d))/T)
N=np.arange(T)
def FoReL(eta):
    TotalRegret=np.zeros((20,T))
    for i in  range(20):
        W=[]
        W.append(np.ones(10)/10)
        CumLoss = 0
        R=np.zeros(T)
        for t in range(T):
            Parameters = np.array([.5,.5,.5,.5,.5,.5,.5,.5,.4,.6])
            if t>T/2:
                Parameters[-1]=0.3
            
            ExpertLoss = np.random.binomial(1,p=Parameters)
            #print(ExpertLoss)
            cost = np.inner(W[t],ExpertLoss)
            #print(cost)
            CumLoss+=cost
            R[t]=CumLoss
            #update Weights
            W.append(W[t]*np.exp(-eta*ExpertLoss))
        TotalRegret[i] = R
    AvgRegret = np.mean(TotalRegret,axis=0)

    regret_mean = np.zeros(T)
    regret_err = np.zeros(T)
    freedom_degree = len(TotalRegret) - 1
    for i in range(T):
        regret= np.zeros(20)
        for j in range(20):
            regret[j]=(TotalRegret[j][i])
        regret_mean[i]=(np.mean(regret))
        regret_err[i]=(ss.t.ppf(0.95, freedom_degree) * ss.sem(regret))
    
    
    return(AvgRegret,regret_mean,regret_err)      
   
    
colors = list("rgbcmykrgbc")
shape = ['--d', '--v','--^', '--d', '--v','--d', '--v','--^', '--d', '--v','--^']
c = np.arange(.1,2.2,.2)  
EtaVec= c*eta_star
RC=np.zeros(len(c))
for i in range(len(c)):
    AvgRegret,regret_mean,regret_err=FoReL(EtaVec[i])
    plt.errorbar(N, regret_mean, regret_err)
    plt.plot(N, AvgRegret , colors[i] + shape[i], label='c: '+str(c[i]))


plt.legend(loc='upper left', numpoints=1)
plt.title("Cumulative Pseudo Regret vs T(10^5) for different of C values[.1,2.2]")
plt.xlabel("T")
plt.ylabel("Cumulative Pseudo Regret")
plt.savefig("Q3.png", bbox_inches='tight')
plt.close()









