import numpy as np

import pyximport
pyximport.install() 

from rover_domain_core_gym import RoverDomainGym
from utils.ccea_2 import *

#pri alignment multiagent tumernt(vals)
def make_env(nagents,rand=0):
    vals =np.array([0.8,1.0,0.6,0.3,0.2,0.1])
    
    pos=np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.5],
        [0.0, 0.5],
        [1.0, 0.0]
    ])

    sim = RoverDomainGym(nagents,30,pos,vals)
 
    sim.data["Coupling"]=1
    sim.data['Number of Agents']=nagents
    sim.data['Trains per Episode']=32 #pop size

    obs=sim.reset()
    return sim

env=make_env(4)
initCcea(input_shape=8, num_outputs=2, num_units=20)(env.data)

populationSize=len(env.data['Agent Populations'][0])
pop=env.data['Agent Populations']
#team=self.team[0]

for gen in range(1000):
    Globals=[]
    for worldIndex in range(populationSize):
        env.data["World Index"]=worldIndex
        
        #for agent_idx in range(self.types):
        
        
        state = env.reset() 
        done=False 
        #assignCceaPoliciesHOF(env.data)
        assignCceaPolicies(env.data)
        while not done:
            
            
            policyCol=env.data["Agent Policies"]
            action=[]
            for s,pol in zip(state,policyCol):
    
                a = pol.get_action(s)
                action.append(a)
            action = np.array(action)*2.0
            state, D, done, info = env.step(action)

        #learning happens here for reward functions 

        for pol,d in zip(policyCol,D):
            pol.fitness=d#neural network critic reward function
        G=env.data["Global Reward"]
        Globals.append(G)

    print("Gen :"+str(gen)+"  Best G: "+str(max(Globals)))

    evolveCceaPolicies(env.data)