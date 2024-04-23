import numpy as np
import torch
from collections import deque
from random import sample

class Net():
    def __init__(self,device,hidden=20*4,lr=5e-4,loss_fn=2):#*4
        learning_rate=lr
        self.device=device

        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(8, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden,1)
        ).to(device)
        
            
        if loss_fn==0:
            self.loss_fn = torch.nn.MSELoss(reduction='sum')
        elif loss_fn==1:
            self.loss_fn = self.alignment_loss
        elif loss_fn ==2:
            self.loss_fn = lambda x,y: self.alignment_loss(x,y) + torch.nn.MSELoss(reduction='sum')(x,y)

        self.sig = torch.nn.Sigmoid()

        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    def feed(self,x):
        x=torch.from_numpy(x.astype(np.float32)).to(self.device)
        pred=self.model(x)
        return pred.cpu().detach().numpy()
        
    
    def train(self,x,y,shaping=False,n=5,verb=0):
        x=torch.from_numpy(x.astype(np.float32)).to(self.device)
        y=torch.from_numpy(y.astype(np.float32)).to(self.device)
        pred=self.model(x)
        
        loss=self.loss_fn(pred,y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.cpu().detach().item()

    def alignment_loss(self,o, t,shaping=False):
        if shaping:
            o=o+t
        ot=torch.transpose(o,0,1)
        tt=torch.transpose(t,0,1)

        O=o-ot
        T=t-tt

        align = torch.mul(O,T)
        #print(align)
        align = self.sig(align)
        loss = -torch.mean(align)
        return loss
class align():
    def __init__(self,nagents,loss_f=0):
        self.nagents=nagents
        #self.nets=[Net(loss_fn=loss_f) for i in range(nagents)]
        self.nets=[LSTM(loss_fn=loss_f) for i in range(nagents)]
        self.hist=[deque(maxlen=30000) for i in range(nagents)]

    def add(self,trajectory,G,agent_index):
        self.hist[agent_index].append([trajectory,G])

    def evaluate(self,trajectory,agent_index):
        input = np.array([trajectory])
        
        npad = ((0, 32-input.shape[0]), (0, 0), (0, 0))
        input = np.pad(input, pad_width=npad, mode='constant', constant_values=0)
        input = torch.from_numpy(input.astype(np.float32))

        feedback = self.nets[agent_index].forward(input)
        return feedback[0,0]
    
        #return self.nets[agent_index].feed(trajectory[-1])

class LSTM(torch.nn.Module):
    def __init__(self, input_size=8, hidden_size=20*4, num_layers=1, lr=2e-3, loss_fn=0):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.learning_rate = lr

        self.lstm = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.output = torch.nn.Linear(self.hidden_size, 1)

        self.sig = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if loss_fn==0:
            self.loss_fn = torch.nn.MSELoss(reduction='sum')
        elif loss_fn==1:
            self.loss_fn = self.alignment_loss
        elif loss_fn ==2:
            self.loss_fn = lambda x,y: self.alignment_loss(x,y) + torch.nn.MSELoss(reduction='sum')(x,y)
    
    def forward(self,x):
        _, (hn, cn) = self.lstm(x)
        out = self.relu(hn)
        out = self.output(out)
        out = self.tanh(out)
        return out
    
    def train(self,x,y,shaping=False,n=5,verb=0):
        y = np.expand_dims(y, axis=0)

        x=torch.from_numpy(x.astype(np.float32))
        y=torch.from_numpy(y.astype(np.float32))

        pred = self.forward(x)
        
        loss=self.loss_fn(pred,y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def alignment_loss(self,o, t,shaping=False):
        if shaping:
            o=o+t

        ot=torch.transpose(o,0,1)
        tt=torch.transpose(t,0,1)

        O=o-ot
        T=t-tt

        align = torch.mul(O,T)
        #print(align)
        align = self.sig(align)
        loss = -torch.mean(align)
        return loss



    def train(self):
        loss = 99999
        for a in range(self.nagents):
            for i in range(100):
                if len(self.hist[a])<24:
                    trajG=self.hist[a]
                else:
                    trajG=sample(self.hist[a],24)
                S,G=[],[]
                for traj,g in trajG:
                    S.append(traj)
                    G.append([g])
                S,G=np.array(S),np.array(G)

                loss = self.nets[a].train(S,G)

            print(f"Agent {a}, Loss {loss}")

if __name__ == "__main__":
    model = LSTM()
    input = np.array([[5,2,3,4,5,1,2,8],[1,2,3,4,5,6,7,8],[2,1,5,4,3,6,7,8]])
    input = np.expand_dims(input, axis=0)

    npad = ((0, 32-input.shape[0]), (0, 0), (0, 0))
    input = np.pad(input, pad_width=npad, mode='constant', constant_values=0)

    print(input.shape)

    print(input)
    input=torch.from_numpy(input.astype(np.float32))
    print(model.forward(input)[0,0])

    # net = Net()
    # input = np.array([5,2,3,4,5,1,2,8])
    # print(type(net.feed(input)))
    # print(net.feed(input))

