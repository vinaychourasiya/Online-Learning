import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.optimize import bisect
from scipy.stats import entropy
import numpy.ma as ma


KVec=[5,10,15,20,25]
print("programe is running...")
def check_condition(bound, arm_expected_reward, sample_count, Beta_val):

    p = [arm_expected_reward, 1 - arm_expected_reward]
    q = [bound, 1 - bound]
    return sample_count * entropy(p, qk=q) - Beta_val

# Computing Upper and Lower bound
def upperbound(arm_expected_reward, sample_count, Beta_val):
    try:
        upper_bound_val = bisect(check_condition, arm_expected_reward, 1, args=(arm_expected_reward, sample_count, Beta_val))
    except:
        upper_bound_val = 1
    return upper_bound_val
   
def lowerbound(arm_expected_reward, sample_count, Beta_val):
    try:
        lower_bound_val = bisect(check_condition, 0,arm_expected_reward, args=(arm_expected_reward, sample_count, Beta_val))
    except:
        lower_bound_val = 0
    return lower_bound_val
def KL_LUCB(K):
    MeanVec = [3/4]
    for i in range(2,K+1):
        MeanVec.append(3/4-i/40)
    TotalSamples=[]
    Mistakes=0
    for n in range(20):
        N=np.ones(K)
        S=np.zeros(K)
        UB=np.zeros(K)
        LB=np.zeros(K)
        for k in range(K):
            r=np.random.binomial(1,p=MeanVec[k])
            S[k]=r
        P_hat=S/N
        t=1
        delta=0.1
        alpha=2
        k1=4*np.exp(1)+4
        B=100
        Beta_val=np.log(k1*K*(t**alpha)/delta)+np.log(np.log(k1*K*(t**alpha)/delta))
        for k in range(K):
            UB[k]=upperbound(P_hat[k],N[k],Beta_val)
            LB[k]=lowerbound(P_hat[k],N[k],Beta_val)
        
        
        while B>0:
            mx=np.zeros(K)
            l_t=np.argmax(P_hat)
            #l_t=LB[maxI]
            mx[l_t]=1
            J_c = ma.masked_array(UB,mask=mx)
            u_t=np.argmax(J_c)
            r1=np.random.binomial(1,p=MeanVec[u_t])
            r2=np.random.binomial(1,p=MeanVec[l_t])
            S[u_t]+=r1;S[l_t]+=r2
            N[u_t]+=1;N[l_t]+=1
            P_hat=S/N
            t+=1
            Beta_val=np.log(k1*K*t**alpha/delta)+np.log(np.log(k1*K*t**alpha/delta))
            for k in range(K):
                UB[k]=upperbound(P_hat[k],N[k],Beta_val)
                LB[k]=lowerbound(P_hat[k],N[k],Beta_val)
            B=UB[u_t]-LB[l_t]
        
        if l_t!=0:
            Mistakes+=1
        TotalSamples.append(2*t)
    
    freedom_degree=len(TotalSamples)-1
    S_mean=np.mean(TotalSamples)
    S_err=ss.t.ppf(0.95, freedom_degree) * ss.sem(TotalSamples)
        
    return(S_mean,S_err,Mistakes)



SMean=[]
SErr=[]
Mistakes=[]
for k in KVec:
    S_mean,S_err,M=KL_LUCB(k)
    SMean.append(S_mean);SErr.append(S_err)
    Mistakes.append(M)
    print("Cpmplete For K value "+str(k)+"...")
print('Completed')
np.savetxt('KLLUCB.csv',SMean,delimiter=',')
plt.figure(dpi=100)
plt.plot(KVec,SMean,marker='d',label='KL-LUCB')
plt.errorbar(KVec,SErr,SMean)
plt.title(" Plot of sample complexity v/s number of arms for KL-LUCB")
plt.ylabel("Sample Complexity")
plt.xlabel("number of arms")
plt.legend(loc='upper left', numpoints=1)
plt.savefig('KLLUCB')
#plt.show()

















