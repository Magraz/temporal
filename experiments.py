from train import make_env,train
from rewards.align import align
from rewards.attention import attention
from rewards.fitnesscritic import fitnesscritic
from rewards.lstm import lstm
import multiprocessing as mp
import numpy as np
import pickle as pkl
import time
import torch
import torch.multiprocessing as mp

def experiment(n_agents,reward_type,trial,device):
    fname="-".join([str(i) for i in [n_agents,reward_type,trial]])+".pkl"
    env=make_env(n_agents)
    
    if reward_type==0:
        reward_mechanism=align(n_agents,device,loss_f=0)
    elif reward_type==1:
        reward_mechanism=align(n_agents,device,loss_f=2) #alignment
    elif reward_type==2:
        reward_mechanism=attention(n_agents,device,loss_f=0)
    elif reward_type==3:
        reward_mechanism=attention(n_agents,device,loss_f=2) #alignment
    elif reward_type==4:
        reward_mechanism=fitnesscritic(n_agents,device)
    elif reward_type==5:
        reward_mechanism=lstm(n_agents,device)

    R,pos=train(env,reward_mechanism)
    with open("saves/"+fname,"wb") as f:
        pkl.dump([R,pos],f)



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    try:
        mp.set_start_method('spawn')
    except:
        print("no spawn")
    for reward_type in [5]:
        for n_agents in [4,6,8]:
            procs=[]
            for trial in range(10,20):
                p=mp.Process(target=experiment,args=(n_agents,reward_type,trial,device))
                time.sleep(1)
                procs.append(p)
                p.start()
            for p in procs:
                p.join()