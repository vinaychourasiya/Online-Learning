import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.optimize import bisect
from scipy.stats import entropy

"""
array description
MeanVec = mean of all arm given in the Qus
R = Regret vector that collect cumulative regret for all round(T=25000)
M /(AlgoM)= Mean arm that algorithm gives at each round (use for calculation of regret)   
TotalRegret = matrix that collect Vector R for 20 times (used in Error bar)
"""


T=25000
K=10
MeanVec=[.5]
for j in range(2,K+1):
    MeanVec.append(1/2-j/70)
MeanVec=np.array(MeanVec)

def Thompson(K):
    TotalRegret=[]
    for n in range(20):
        R=np.zeros(T)
        S=np.zeros(K)
        F=np.zeros(K)
        M=0
        for t in range(T):
            theta=np.zeros(K)
            for k in range(K):
                theta[k]=np.random.beta(S[k]+1,F[k]+1)
            I_t=np.argmax(theta)
            r=np.random.binomial(1,p=MeanVec[I_t])
            if r==1:
                S[I_t]+=1
            else:
                F[I_t]+=1
            M+=MeanVec[I_t]
            R[t]=(t+1)*MeanVec[0]-M
        TotalRegret.append(R)
    TotalRegret=np.array(TotalRegret)

    AvgRegret = np.mean(TotalRegret,axis=0)
    N=np.arange(len(AvgRegret))
    regret_mean = np.zeros(len(AvgRegret))
    regret_err = np.zeros(len(AvgRegret))
    freedom_degree = len(AvgRegret) - 1
    for i in range(len(AvgRegret)):
            regret=np.zeros(20)
            for j in range(20):
                    regret[j]=TotalRegret[j][i]
            regret_mean[i]=np.mean(regret)
            regret_err[i]=ss.t.ppf(0.95, freedom_degree) * ss.sem(regret)
    return(AvgRegret,regret_mean,regret_err,N)




def UCB(K):
    TotalRegret=[]
    for n in range(20):
        alpha=1.5
        M=np.zeros(K)
        count=np.ones(K)
        CumMean =np.zeros(K)
        AlgoM=0
        R=[]
        for i in range(K):
            r = np.random.binomial(1,MeanVec[i])
            M[i]=r
        for t in range(1,T):
            v = M/count+ np.sqrt((alpha*np.log(t))/count)
            I_t = np.argmax(v)
            r = np.random.binomial(1,MeanVec[I_t])
            M[I_t]+=r
            count[I_t]+=1
            AlgoM += MeanVec[I_t]
            R.append(t*MeanVec[0] - AlgoM)
        TotalRegret.append(R)
    TotalRegret=np.array(TotalRegret)

    AvgRegret = np.mean(TotalRegret,axis=0)
    N=np.arange(len(AvgRegret))
    regret_mean = np.zeros(len(AvgRegret))
    regret_err = np.zeros(len(AvgRegret))
    freedom_degree = len(AvgRegret) - 1
    for i in range(len(AvgRegret)):
            regret=np.zeros(20)
            for j in range(20):
                    regret[j]=TotalRegret[j][i]
            regret_mean[i]=np.mean(regret)
            regret_err[i]=ss.t.ppf(0.95, freedom_degree) * ss.sem(regret)

    return(AvgRegret,regret_mean,regret_err,N)
            
def Egreedy(K):
    TotalRegret=[]
    for n in range(20):
        arms=np.arange(K)
        M=np.zeros(K)
        count=np.zeros(K)
        prob=np.ones(K)/K
        AlgoM=0
        R=[]
        for t in range(T):
            I_t = np.random.choice(arms,size=1,p=prob)
            r = np.random.binomial(1,MeanVec[I_t])
            M[I_t]+= r
            count[I_t] += 1
            temp=np.zeros(K)
            for i in range(K):
                if count[i]==0:
                    temp[i]=0
                else:
                    temp[i]=M[i]/count[i]
            v = np.argmax(temp)
            # v = np.argmax((M/count))
            epsilon=1/(t+1)
            for j in range(K):
                if j==v:
                    prob[j]=1-epsilon+(epsilon/K)
                else:
                    prob[j]=epsilon/K
            AlgoM += MeanVec[I_t]
            R.append((t+1)*MeanVec[0] - AlgoM)
        TotalRegret.append(R) 
    TotalRegret=np.array(TotalRegret)
    AvgRegret = np.mean(TotalRegret,axis=0)
    N=np.arange(len(AvgRegret))
    regret_mean = np.zeros(len(AvgRegret))
    regret_err = np.zeros(len(AvgRegret))
    freedom_degree = len(AvgRegret) - 1
    for i in range(len(AvgRegret)):
        regret=np.zeros(20)
        for j in range(20):
            regret[j]=TotalRegret[j][i]
        regret_mean[i]=np.mean(regret)
        regret_err[i]=ss.t.ppf(0.95, freedom_degree) * ss.sem(regret)

    return(AvgRegret,regret_mean,regret_err,N)
        
        
def UCB_V(K):
    TotalRegret=[]
    for n in range(20):
        Sample_arm = []
        Mu_t=np.zeros(K)
        Var=np.zeros(K)
        count=np.ones(K)
        AlgoM=0
        R=[]
        for i in range(K):
            r = np.random.binomial(1,MeanVec[i])
            x=[r]
            Sample_arm.append(x)
        for k in range(K):
            Mu_t[k] = np.mean(Sample_arm[k])
            Var[k]=np.var(Sample_arm[k])    
        for t in range(1,T):
            B = Mu_t + np.sqrt((1.2*2*Var*np.log(t))/count)+(1.2*3*np.log(t)/count)
            I_t = np.argmax(B)
            r = np.random.binomial(1,MeanVec[I_t])
            Sample_arm[I_t].append(r)
            count[I_t]+=1
            AlgoM += MeanVec[I_t]
            Mu_t[I_t]=np.mean(Sample_arm[I_t])
            Var[I_t]=np.var(Sample_arm[I_t])
            R.append((t*MeanVec[0])-AlgoM)
        TotalRegret.append(R)
    TotalRegret=np.array(TotalRegret)
    AvgRegret = np.mean(TotalRegret,axis=0)
    N=np.arange(len(AvgRegret))
    regret_mean = np.zeros(len(AvgRegret))
    regret_err = np.zeros(len(AvgRegret))
    freedom_degree = len(AvgRegret) - 1
    for i in range(len(AvgRegret)):
            regret=np.zeros(20)
            for j in range(20):
                    regret[j]=TotalRegret[j][i]
            regret_mean[i]=np.mean(regret)
            regret_err[i]=ss.t.ppf(0.95, freedom_degree) * ss.sem(regret)
    return(AvgRegret,regret_mean,regret_err,N)
                  
def KL_UCB(K):
    TotalRegret=[]
    def check_condition(bound, arm_expected_reward, sample_count, log_val):
        p = [arm_expected_reward, 1 - arm_expected_reward]
        q = [bound, 1 - bound]
        return sample_count * entropy(p, qk=q) - log_val

    # Computing Upper and Lower bound
    def compute_q(arm_expected_reward, sample_count, log_val):
        try:
            upper_bound_val = bisect(check_condition, arm_expected_reward, 1, args=(arm_expected_reward, sample_count, log_val))
        except:
            upper_bound_val = 1
        return upper_bound_val

    for i in range(20):
        R=[]
        S=np.zeros(K)
        N=np.ones(K)
        Mu_hat=np.zeros(K)
        V=np.zeros(K)
        AlgoM=0
        for k in range(K):
            r=np.random.binomial(1,p=MeanVec[k])
            S[k]=r
        for t in range(1,T):
          
            Mu_hat = S/N
            log_val=np.log(t)
            for k in range(K):
                V[k] = compute_q(Mu_hat[k],N[k],log_val)
            a = np.argmax(V)
            r=np.random.binomial(1,MeanVec[a])
            N[a]+=1
            S[a]+=r
            AlgoM+=MeanVec[a]
            R.append((t*MeanVec[0])-AlgoM)
        TotalRegret.append(R)
    TotalRegret=np.array(TotalRegret)

    AvgRegret = np.mean(TotalRegret,axis=0)
    N=np.arange(len(AvgRegret))
    regret_mean = np.zeros(len(AvgRegret))
    regret_err = np.zeros(len(AvgRegret))
    freedom_degree = len(AvgRegret) - 1
    for i in range(len(AvgRegret)):
        regret=np.zeros(20)
        for j in range(20):
            regret[j]=TotalRegret[j][i]
        regret_mean[i]=np.mean(regret)
        regret_err[i]=ss.t.ppf(0.95, freedom_degree) * ss.sem(regret)

    return(AvgRegret,regret_mean,regret_err,N)
            

print("Programme is running...")
colors = list("rgbcy")
shape = ['--d', '--v','--^','--o','--x']

plt.figure(dpi=200)
AvgRegretE,regret_meanE,regret_errE,NE=Egreedy(K)
plt.errorbar(NE, regret_meanE, regret_errE)
plt.plot(NE, AvgRegretE , colors[0] + shape[0], label='E-greedy')

print("E-greedy Done,programme is running")

AvgRegretU,regret_meanU,regret_errU,NU=UCB(K)
plt.errorbar(NU, regret_meanU, regret_errU)
plt.plot(NU, AvgRegretU , colors[1] + shape[1], label='UCB(1)')
print("UCB Done,programme is running")

AvgRegretTS,regret_meanTS,regret_errTS,NTS=Thompson(K)
plt.errorbar(NTS, regret_meanTS, regret_errTS)
plt.plot(NTS, AvgRegretTS , colors[2] + shape[2], label='Thompson')
print("Thompson Done,programme is running")

AvgRegretUV,regret_meanUV,regret_errUV,NUV=UCB_V(K)
plt.errorbar(NUV, regret_meanUV, regret_errUV)
plt.plot(NUV, AvgRegretUV , colors[3] + shape[3], label='UCB-V')
print("UCB-V Done,programme is running")

AvgRegretKL,regret_meanKL,regret_errKL,NKL=KL_UCB(K)
plt.errorbar(NKL, regret_meanKL, regret_errKL)
plt.plot(NKL, AvgRegretKL , colors[4] + shape[4], label='KL-UCB')
print("KL-UCB Done,programme is running")

plt.legend(loc='upper left', numpoints=1)
plt.title("Cumulative Pseudo Regret vs T(25000) for K=10")
plt.xlabel("Number of sample(T)")
plt.ylabel("Cumulative Pseudo Regret")
#plt.show()
plt.savefig("Q1K10.png", bbox_inches='tight')
plt.close()






























         
