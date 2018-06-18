import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss


K=10
T=10**5

def Exp3(T,K,etha):
   Regret=[]
   for i in range(50):
        P=np.array([.1,.1,.1,.1,.1,.1,.1,.1,.1,.1])
        Cum_Loss=np.zeros(K)
        
        ExpectLoss=0
        MinArmLossVector=np.zeros(K)
                
        for t in range(T):
            
            ExpectedArm_Loss = np.array([.5,.5,.5,.5,.5,.5,.5,.5,.4,.6])
            if t>T/2:
                ExpectedArm_Loss[-1]=0.3
            ArmLossVector = np.random.binomial(1,p=ExpectedArm_Loss)

            # Choose arm with P
            arm_index=np.random.choice(list(range(K)),p=P)
            arm_loss = (ArmLossVector[arm_index])/P[arm_index]
            Cum_Loss[arm_index] = Cum_Loss[arm_index]+arm_loss
      
            ExpectLoss += ExpectedArm_Loss[arm_index]
            MinArmLossVector += ExpectedArm_Loss
                                      
                   
            #Update Probability vector
            P=(np.exp(-etha*Cum_Loss))/(np.sum(np.exp(-etha*Cum_Loss)))
        
        Regret.append(ExpectLoss-np.min(MinArmLossVector))
      
   print("Regret", np.mean(Regret))
   Regret.append(etha)
   return (Regret)

c=np.arange(.1,2.2,.2)      
ethaVector=c*(np.sqrt((2*np.log(K))/(K*T)))

exp3_regret=[]
for i,etha in enumerate(ethaVector):
   R=Exp3(T, K, etha)
   exp3_regret.append(R)


"""[ 6898.86399996  3564.496       2024.60200001  1132.21600001    73.08400002
  -414.34599999 -1028.08199999 -1620.884      -1694.57       -1937.34800001
 -2230.52200002]
 """

# Plotting Regret vs Eta
# EXP3
eta = []
regret_mean = []
regret_err = []
freedom_degree = len(exp3_regret[0]) - 2
for regret in exp3_regret:
    eta.append(regret[-1])
    regret_mean.append(np.mean(regret[:-1]))
    regret_err.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[:-1]))

colors = list("rgbcmyk")
shape = ['--^', '--d', '--v']
plt.errorbar(eta, regret_mean, regret_err, color=colors[2])
plt.plot(eta, regret_mean, colors[0] + shape[0], label='EXP3')

# Plotting
plt.legend(loc='upper right', numpoints=1)
plt.title("Pseudo Regret vs Learning Rate for T = 10^5 and 50 Sample paths")
plt.xlabel("Learning Rate")
plt.ylabel("Pseudo Regret")
plt.savefig("exp3.png", bbox_inches='tight')
plt.close()

