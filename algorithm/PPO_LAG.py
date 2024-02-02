import collections
import random
import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cpu")

class PolicyNet(torch.nn.Module):
    def __init__(self, stateDim,actionDim):  # 244 # 8
        super(PolicyNet, self).__init__()
        self.stateDim = stateDim # 244
        self.actionDim = actionDim # 8
        self.S = torch.nn.Linear(self.stateDim, 64).to(device)
        self.A = torch.nn.Linear(self.actionDim,4).to(device)
        self.L1 = torch.nn.Linear(64+4,32).to(device)
        self.L2 = torch.nn.Linear(32,8).to(device)
        self.f = torch.nn.Linear(8, 1).to(device)

    def forward(self, X):
        s = X[:, :self.stateDim]
        a = X[:, -self.actionDim:]
        s1 = F.relu(self.S(s))
        a1 = F.relu(self.A(a))
        y1 = torch.cat((s1, a1), dim=1)
        l1 = F.relu((self.L1(y1)))
        l2 = F.relu((self.L2(l1)))
        return self.f(l2)


# critic
class RewardValueNet(torch.nn.Module):
    def __init__(self, stateDim):
        super(RewardValueNet, self).__init__()
        self.stateDim = stateDim  # 244
        self.S = torch.nn.Linear(stateDim, 64).to(device)
        self.L1 = torch.nn.Linear(64, 32).to(device)
        self.L2 = torch.nn.Linear(32, 8).to(device)
        self.f = torch.nn.Linear(8, 1).to(device)

    def forward(self, X):
        s = X[:, :self.stateDim]
        y1 = F.relu(self.S(s))
        l1 = F.relu((self.L1(y1)))
        l2 = F.relu((self.L2(l1)))
        return F.relu(self.f(l2))



class CostValueNet(torch.nn.Module):
    def __init__(self, stateDim):
        super(CostValueNet, self).__init__()
        self.stateDim = stateDim  # 244
        self.S = torch.nn.Linear(stateDim, 64).to(device)
        self.L1 = torch.nn.Linear(64, 32).to(device)
        self.L2 = torch.nn.Linear(32, 8).to(device)
        self.f = torch.nn.Linear(8, 1).to(device)

    def forward(self, X):
        s = X[:, :self.stateDim]
        y1 = F.relu(self.S(s))
        l1 = F.relu((self.L1(y1)))
        l2 = F.relu((self.L2(l1)))
        return F.relu(self.f(l2))


class ReplayBuffer:
    def __init__(self, capacity,batchSize):
        self.buffer = collections.deque(maxlen=capacity)
        self.batchSize = batchSize

    def add(self,matchingState,state,action,reward,cost,nextState):
        self.buffer.append((matchingState,state,action,reward,cost,nextState))

    def sample(self):
        transitions = random.sample(self.buffer, self.batchSize)
        matchingState,state,action,reward,cost,nextState = zip(*transitions)
        state = list(state)
        state = [x.tolist() for x in state]
        return state, action, reward, np.array(nextState)

    def size(self):
        return len(self.buffer)


class PPOLag:
    def __init__(self, stateDim, actionDim, actorLr, criticLr, gamma, batchSize,device):
        self.actor = PolicyNet(stateDim,actionDim).to(device)
        self.rewardCritic = RewardValueNet(stateDim).to(device)
        self.costCritic = CostValueNet(stateDim).to(device)
        self.actorLr = actorLr
        self.rewardCriticLr = criticLr
        self.costCriticLr = criticLr
        self.gamma = gamma
        self.batchSize = batchSize
        self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actorLr)
        self.rewardCriticOptimizer = torch.optim.Adam(self.rewardCritic.parameters(), lr=self.rewardCriticLr)
        self.costCriticOptimizer = torch.optim.Adam(self.costCritic.parameters(), lr=self.costCriticLr)

    def take_action(self, matchingState):  # 训练
        matchingState = torch.tensor(matchingState, dtype=torch.float).to(device)
        vOutput = self.actor(matchingState)
        vOutput = vOutput.reshape(-1)  # 将二维张量变为一维张量
        actionProb = torch.softmax(vOutput, dim=0)
        a = random.random()
        if a < 1:
            actionDist = torch.distributions.Categorical(actionProb)
            action = actionDist.sample().cpu()
        else:
            action = torch.max(actionProb,0)[1]
        return action.item()  # 对softmax函数求导时的用法

    def update(self, matchingState,state,action,reward,cost,nextState,round):