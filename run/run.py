import pandas as pd
import numpy as np
import math
from simulator.envs import *
from algorithm.PPO_LAG import *

dayIndex = 0  # the current day
maxDay = 50 # Max day
maxTime = 960 # Max dispatch round

minlon = 113.90
maxlon = 114.05
minlat = 22.530
maxlat = 22.670

locRange = [minlon,maxlon,minlat,maxlat]

M = 10
N = 10

memorySize = 100000
batchSize = 50


# ① 启始位置随机


orderInfo = pd.read_pickle('../data/order.pkl')  # 30天订单信息

driverPreInit = pd.read_pickle('../data/driver_preference.pkl')  # 骑手初始化偏好信息
driverLocInit = pd.read_pickle('../data/driver_location.pkl') # 骑手初始化位置信息

driverNum = 1000

env = Env(driverPreInit,driverLocInit,orderInfo,locRange,driverNum,M,N,maxDay,maxTime)

replayBuffer = ReplayBuffer(memorySize, batchSize)

env.set_driver_info(driverPreInit,driverLocInit)
env.set_region_info()

while dayIndex < maxDay:
    env.set_day_info(dayIndex)
    env.reset_clean()

    T = 0
    dDict = {}
    while T < maxTime:
        for order in env.dayOrder[T]:
            driverList = env.driver_collect(order)
            if driverList == 0:
                continue
            candidateList = env.generate_candidate_set(order,driverList)
            driverStateArray = env.driver_state_calculate(candidateList)  # 44
            actionStateArray = env.action_state_calculate(candidateList,order) # 8
            contextualArray = env.con_state_calcualte() # 200
            stateArray = np.hstack((driverStateArray,contextualArray.reshape(1,-1).repeat(driverStateArray.shape[0], 0)))

            matchingStateArray = np.hstack((stateArray,actionStateArray)) # 252

           # action = agent.take_action(matchingStateArray) # action是一个值
            action = 18

            rightDriver = candidateList[action]
            rightRegion = env.regionList[order.oriRegion]
            state = stateArray[action] # 244
            matchingState = matchingStateArray # 252
            wt = actionStateArray[action,7]
            dt = actionStateArray[action,6]
            trs = int(math.ceil(wt + dt))
            #nextState = env.driver_next_state_calculate(candidateList[action],order, trs)
            meanwt = env.regionList[order.oriRegion].meanwt
            reward = env.cal_reward(wt,meanwt,trs)
            cost = env.cal_cost(order,candidateList[action])

            d = DispatchSolution()
            d.add_state(state)
            d.add_matchingState(matchingState)
            d.add_trs(trs)
            d.add_action(action)
            d.add_reward(reward)
            d.add_cost(cost)
            dDict[rightDriver] = d
            #d.add_nextState(nextState)

            rightDriver.accept_order(trs,order.destLoc,cost)
            rightRegion.accept_order(wt)

        env.step(dDict,replayBuffer)
        T = T + 1


