import numpy as np

T=10**5
K=10
def Exp3p(etha,gamma,beta,T,K):
    
    Regret=[]
    
    for i in range(50):
        P=np.array([.1 for i in range(K)])
        Cum_Loss=np.zeros(K)
        ExpectLoss=0
        MinArmLossVector=np.zeros(K)
        for t in range(T):
            Arm_Dist = np.array([.5,.5,.5,.5,.5,.5,.5,.5,.4,.6])
            if t>T/2:
                Arm_Dist[-1]=0.3
            ArmLossVector = np.random.binomial(1,p=Arm_Dist)
            
            arm_index=np.random.choice(K,p=P)
            arm_loss = (ArmLossVector[arm_index]+beta)/P[arm_index]
            Cum_Loss[arm_index]+=arm_loss
            # regret calculation
            k=Arm_Dist[arm_index]
            ExpectLoss += k
            MinArmLossVector += Arm_Dist
            
            #probabilities Update
            P=(np.exp(-etha*Cum_Loss)*(1-gamma))/(np.sum(np.exp(-etha*Cum_Loss)))+gamma/K
            
        Regret.append(ExpectLoss-np.min(MinArmLossVector))
        
    print("Regret..",np.mean(Regret))
    Regret.append(etha)
    return (Regret)

c=np.arange(.1,2.2,.2)      
ethaVector=c*(np.sqrt((2*np.log(K))/(K*T)))
beta=ethaVector
gamma=K*ethaVector
exp3p_regret=[]
for i in range(len(ethaVector)):      
    exp3p_regret[i]=Exp3p(ethaVector[i],gamma[i],beta[i],T,K)
#print(Regret)
""" 
[ 6915.42599996  3595.226       2222.69000001  1702.02200001  1390.86200001
  1219.062       1098.068        991.212       1037.162        955.498
  1021.044     ]
"""
# EXP3.P
eta = []
regret_mean = []
regret_err = []
freedom_degree = len(exp3p_regret[0]) - 2
for regret in exp3p_regret:
    eta.append(regret[-1])
    regret_mean.append(np.mean(regret[:-1]))
    regret_err.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[:-1]))

colors = list("rgbcmyk")
shape = ['--^', '--d', '--v']
plt.errorbar(eta, regret_mean, regret_err, color=colors[1])
plt.plot(eta, regret_mean, colors[1] + shape[1], label='EXP3.P')

plt.legend(loc='upper right', numpoints=1)
plt.title("Pseudo Regret vs Learning Rate for T = 10^5 and 50 Sample paths")
plt.xlabel("Learning Rate")
plt.ylabel("Pseudo Regret")
plt.savefig("Exp3p.png", bbox_inches='tight')
plt.close()


