import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss


d=10
T=10**5          
#np.random.seed(123456)
N=np.arange(T)
def FoReL():
    
    TotalRegret=np.zeros((20,T))
    eta =np.sqrt((2*np.log(d))/T)
    for i in  range(20):
        #weights = np.ones((d))
        W=[]
        W.append(np.ones(10)/10)
        CumLoss = 0
        R=np.zeros(T)
        for t in range(T):
            #W_t= weights/(np.sum(weights))
            Parameters = np.array([.5,.5,.5,.5,.5,.5,.5,.5,.4,.6])
            if t>T/2:
                Parameters[-1]=0.3
            
            ExpertLoss = np.random.binomial(1,p=Parameters)
            cost = np.inner(W[t],ExpertLoss)
            CumLoss+=cost
            R[t]=CumLoss
            #update Weights
            #weights=weights*np.exp(-eta*ExpertLoss)
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
            
    plt.errorbar(N, regret_mean, regret_err, color='r')
    plt.plot(N, AvgRegret , 'b' + '--^', label='FoRL')
    plt.legend(loc='center right', numpoints=1)
    plt.title("Cumulative Pseudo Regret vs T(10^5) for 20 Sample paths")
    plt.xlabel("T")
    plt.ylabel("Cumulative Pseudo Regret")
    plt.savefig("FoReL.png", bbox_inches='tight')
    plt.close()


Experts=10
T=10**5
N=np.arange(1,T+1)
logN=np.log(N)


def FTL():
    TotalRegret=np.zeros((20,T))
    for i in  range(20):
        W=np.zeros(10)
        CumLoss = np.zeros(10)
        Regret=np.zeros(T)
        MinLossArm=0  #np.zeros(10)
        Loss =0
        for t in range(T):
            Parameters = np.array([.5,.5,.5,.5,.5,.5,.5,.5,.4,.6])
            if t>T/2:
                Parameters[-1]=0.3
            
            ExpertLoss = np.random.binomial(1,p=Parameters)
            A = np.argmin((CumLoss))
            CumLoss+=ExpertLoss
            Loss += ExpertLoss[A]
            I = np.argmin(CumLoss)
            MinLossArm += ExpertLoss[I]
            R = Loss - MinLossArm
            Regret[t]=R
            
        TotalRegret[i] = Regret
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
            
    plt.errorbar(logN, regret_mean, regret_err, color='r')
    plt.plot(logN, AvgRegret , 'b' + '--v', label='FTL')
    plt.legend(loc='upper left', numpoints=1)
    plt.title("Cumulative Pseudo Regret vs T(10^5) for 20 Sample paths")
    plt.xlabel("log(T)")
    plt.ylabel("Cumulative Pseudo Regret")
    plt.savefig("FTL.png", bbox_inches='tight')
    plt.close()
    #plt.show()

 
    
FTL()    
FoReL()
        
            
            

