import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

def Exp3Ix(T,K,eta,gamma):
    
    Regret=[]
    print("Regrets for "+str(eta)  +" eta value(Avg over 50 Sample Path)" )
    for i in range(50):
        weights=np.ones(K)
        Arms=np.arange(K)
        Loss = 0
        MinArmLoss=np.zeros(K)
        for t in range(T):
            prob=weights/(np.sum(weights))
            l_tilda=np.zeros(K)            
            ExpectedArm_Loss = np.array([.5,.5,.5,.5,.5,.5,.5,.5,.4,.6])
            if t>T/2:
                ExpectedArm_Loss[-1]=0.3
            ArmLossVector = np.random.binomial(1,p=ExpectedArm_Loss)
            
            It=np.random.choice(Arms,p=prob)
            Observeloss=ArmLossVector[It]
            l_tilda[It]=Observeloss/(prob[It]+gamma)
            Loss +=ExpectedArm_Loss[It] 
            MinArmLoss+=ExpectedArm_Loss
                          
                
            #update weights
            weights=weights*(np.exp(-eta*l_tilda))
        R=Loss - min(MinArmLoss)
        Regret.append(R)
    print(np.mean(Regret))
    Regret.append(eta)
    return (Regret)

K=10
T=10**5  
c=np.arange(.1,2.2,.2)
etaVector= c*np.sqrt(2*np.log(K)/(K*T))
gammaVec=etaVector/2
exp3ix_regret=[]

for i in range(len(c)):
   R=Exp3Ix(T, K, etaVector[i], gammaVec[i])
   exp3ix_regret.append(R)



# EXP3-IX
eta = []
regret_mean = []
regret_err = []
freedom_degree = len(exp3ix_regret[0]) - 2
for regret in exp3ix_regret:
    eta.append(regret[-1])
    regret_mean.append(np.mean(regret[:-1]))
    regret_err.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[:-1]))
colors = list("rgbcmyk")
shape = ['--^', '--d', '--v']
plt.errorbar(eta, regret_mean, regret_err, color=colors[2])
plt.plot(eta, regret_mean, colors[2] + shape[2], label='EXP3-IX')

plt.legend(loc='upper right', numpoints=1)
plt.title("Pseudo Regret vs Learning Rate for T = 10^5 and 50 Sample paths")
plt.xlabel("Learning Rate")
plt.ylabel("Pseudo Regret")
plt.savefig("exp3ix.png", bbox_inches='tight')
plt.close()

