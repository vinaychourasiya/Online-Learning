import numpy as np
import matplotlib.pyplot as plt

K=10
MeanVec = np.array([0.09, 0.55, 0.91, 0.8 , 0.4 ,
                   0.39, 0.11, 0.17, 0.66, 0.53])
CostVec = np.array([0.63, 0.58, 0.98, 0.55, 0.3 ,
                    0.39, 0.67, 0.61, 0.16, 0.25])
R_C_ratio = MeanVec/CostVec
M = np.max(R_C_ratio)
Max_arm = np.argmax(R_C_ratio)

Delta=M-R_C_ratio 

def BTS(B):
    TotalRegret=[]
    for i in range(50):
        Arms = np.arange(K)
        S_r = np.zeros(K)
        F_r =np.zeros(K)
        T = np.zeros(K)
        S_c = np.zeros(K)
        F_c =np.zeros(K)
        #B = 50*1000
        Regret = 0
        while B>0:
            theta_r = np.random.beta(S_r+1,F_r+1)
            theta_c = np.random.beta(S_c+1,F_c+1)

            I_t = np.argmax(theta_r/theta_c)
            r = np.random.binomial(1,MeanVec[I_t])
            c = np.random.binomial(1,CostVec[I_t])
            if r==1:
                S_r[I_t]+=1
            else:
                F_r[I_t]+=1

            if c==1:
                S_c[I_t]+=1
            else:
                F_c[I_t]+=1

            T[I_t]+=1
            B = B  - c
        for i in range(K):
            if i!=Max_arm:
                R=CostVec[i]*Delta[i]*T[i]
                Regret=Regret+R
        TotalRegret.append(Regret)
    TotalRegret= np.array(TotalRegret)
    Avgregret=np.mean(TotalRegret)
    
    return (Avgregret)




def UCB_BV1(B):
    L = min(CostVec)
    T=np.ones(K)
    N=np.ones(K)
    #r = np.random.binomial(1,MeanVec)
    #c = np.random.binomial(1,CostVec)
    X_r = np.ones(K)
    X_c = np.ones(K)
    r_bar  = X_r /N
    c_bar = X_c /N
    t=K
    Regret=0
    while B>=0:
        a = (1+(1/L))*np.sqrt(np.log(t-1)/N)
        b = L - np.sqrt(np.log(t-1)/N)
        D = (r_bar/c_bar) + (a/b)

        I_t = np.argmax(D)
        r = np.random.binomial(1,MeanVec[I_t])
        c = np.random.binomial(1,CostVec[I_t])
        X_c[I_t]+=c
        X_r[I_t]+=r
        N[I_t]+=1
        T[I_t]+=1
        r_bar  = X_r /N
        c_bar = X_c /N
        B=B-c
        t=t+1
    for i in range(K):
        if i!=Max_arm:
            R=CostVec[i]*Delta[i]*T[i]
            Regret=Regret+R
    return Regret
        
        

BudgetVec = []
for i in range(5000,5*10**4+1,5000):
    BudgetVec.append(i)
Re=[]
for B in BudgetVec:
    R=BTS(B)
    Re.append(R)
    print("Done for "+str(B))

BudgetVec[0]=1000
BudgetVec[1]=2000
Re1=[]
for B in BudgetVec:
    R= UCB_BV1(B)
    Re1.append(R)
    print("Done for "+str(B))

plt.plot(BudgetVec,Re,"b^",label='BTS')
plt.plot(BudgetVec,Re1,"ro",label='UCB-BV1')
plt.show()










    
        
