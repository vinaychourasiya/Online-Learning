import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

K=20
MeanVec = np.array([.15,.12,.10,.05,.05,.05,.05,.05,.05,.05,.05
                    ,.05,.03,.03,.03,.03,.03,.03,.03,.03])
L=3
T = 20000
Optimal = np.sum(MeanVec[:3])
def MP_TS(K,L):
    """
    "MS_TS" function- define for Multi-player MAB setting
    using Thompson Sampling
    K = # arms
    L = # of arms play in each round
    """
    TotalRegret=[]
    for i in range(25):
        A = np.ones(K)
        B= np.ones(K)
        S=0
        R=[]
        I=[]
        for t in range(T):
            Theta = np.random.beta(A,B)
            Sort_theta = np.sort(Theta)
            I_t = Sort_theta[-L:]
            Index=[]
            for v in I_t:
                p=Theta.tolist().index(v)
                Index.append(p)

            I.append(Index)
            r = np.random.binomial(1,MeanVec)
            
            for i in Index:
                if  r[i]==1:
                    A[i]+=1
                else:
                    B[i]+=1
            S = S + np.sum(MeanVec[Index]) 

            R.append((t+1)*Optimal -S)
        TotalRegret.append(R)
    TotalRegret=np.array(TotalRegret)
    AvgRegret = np.mean(TotalRegret,axis=0)
    N=np.arange(len(AvgRegret))
    regret_mean = np.zeros(len(AvgRegret))
    regret_err = np.zeros(len(AvgRegret))
    freedom_degree = len(AvgRegret) - 1
    for i in range(len(AvgRegret)):
            regret=np.zeros(25)
            for j in range(25):
                    regret[j]=TotalRegret[j][i]
            regret_mean[i]=np.mean(regret)
            regret_err[i]=ss.t.ppf(0.95, freedom_degree) * ss.sem(regret)
    return(AvgRegret,regret_mean,regret_err,N)




def CUCB(K,L):
    TotalRegret=[]
    for i in range(25):
        N = np.ones(K)
        r = np.random.binomial(1,MeanVec)
        X = r
        Mu_cap = X/N  
        A=0
        R=[]
        for t in range(K,T+1):
            Mu_bar = Mu_cap + np.sqrt(3*np.log(t)/(2*N))

            S1 = np.sort(Mu_bar)
            S = S1[-L:]
            Index=[]
            for v in S:
                p=Mu_bar.tolist().index(v)
                Index.append(p)
            r = np.random.binomial(1,MeanVec)

            for i in Index:
                N[i]=N[i]+1
                X[i]=X[i]+r[i]
            Mu_cap = X/N
            A = A + np.sum(MeanVec[Index]) 
            R.append(((t+1-K)*Optimal) -A)
        TotalRegret.append(R)
    TotalRegret=np.array(TotalRegret)
    AvgRegret = np.mean(TotalRegret,axis=0)
    N=np.arange(len(AvgRegret))
    regret_mean = np.zeros(len(AvgRegret))
    regret_err = np.zeros(len(AvgRegret))
    freedom_degree = len(AvgRegret) - 1
    for i in range(len(AvgRegret)):
            regret=np.zeros(25)
            for j in range(25):
                    regret[j]=TotalRegret[j][i]
            regret_mean[i]=np.mean(regret)
            regret_err[i]=ss.t.ppf(0.95, freedom_degree) * ss.sem(regret)
    return(AvgRegret,regret_mean,regret_err,N)



AvgRegretTS,regret_meanTS,regret_errTS,NTS=MP_TS(K,L)

plt.errorbar(NTS, regret_meanTS, regret_errTS,color='c')
plt.plot(NTS, AvgRegretTS ,'b^', label='MP-TS')


AvgRegretC,regret_meanC,regret_errC,NC=CUCB(K,L)

plt.errorbar(NC, regret_meanC, regret_errC,color='g')
plt.plot(NC, AvgRegretC ,'ro', label='CUCB')

plt.legend(loc='upper left', numpoints=1)
#plt.title("Cumulative Pseudo Regret vs T(20000)")
plt.xlabel("T")
plt.ylabel("Cumulative Pseudo Regret")
plt.savefig("MP.png", bbox_inches='tight')
plt.close()



