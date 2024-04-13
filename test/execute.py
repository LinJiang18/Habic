import pandas as pd
import numpy as np
import math
import pickle
import statistics
from simulator.envs import *
# from algorithm.PPO_LAG import *
from algorithm.AC import *

dayIndex = 90  # the current day
maxDay = 100 # Max day
maxTime = 180 # Max dispatch round

minlon = 113.90
maxlon = 114.05
minlat = 22.530
maxlat = 22.670

locRange = [minlon,maxlon,minlat,maxlat]

M = 10
N = 10

stateDim = 21
actionDim = 8
actorLr = 0.00005
criticLr = 0.00005
lagLr = 0.001
limit = 0.10
lagrange = 1
epochs = 5
eps = 0.2
gamma = 0.95
memorySize = 100000
batchSize = 100
DN = 150


orderInfo = pd.read_pickle('../data/order.pkl')

driverPreInit = pd.read_pickle('../data/driver_preference.pkl')
driverLocInit = pd.read_pickle('../data/driver_location.pkl')
regionWT = pd.read_pickle('../data/regionMeanWT.pkl')



env = Env(driverPreInit, driverLocInit, orderInfo, locRange, DN, M, N, maxDay, maxTime)

agent = torch.load('../testresult/agent.pth')
replayBuffer = ReplayBuffer(memorySize, batchSize)

env.set_driver_info(driverPreInit,driverLocInit)
env.set_region_info(regionWT)

updateRound = 0

regionFairnessMatrix = []
regionMeanMatrix = []


meanRewardList = []
meanCostList = []
dayCostList = []
dayMeanwtList = []
dayIntraRegionfairnesswtList = []
daymaxwtList = []

while dayIndex < maxDay:


    wtList = []
    orderList = []
    rewardList = []
    costList = []
    fairnessList = []
    regionMeanList = []


    env.set_day_info(dayIndex)
    env.reset_clean()

    T = 0
    dDict = {}
    while T < maxTime:
        maxwt = 0
        for order in env.dayOrder[T]:
            driverList = env.driver_collect(order)
            if driverList == 0:
                continue
            candidateList = env.generate_candidate_set(order,driverList)
            driverStateArray = env.driver_state_calculate(candidateList)  # 72
            actionStateArray = env.action_state_calculate(candidateList,order) # 8
            #contextualArray = env.con_state_calcualte() # 200
            #stateArray = np.hstack((driverStateArray,contextualArray.reshape(1,-1).repeat(driverStateArray.shape[0], 0)))

            stateArray = driverStateArray
            matchingStateArray = np.hstack((stateArray,actionStateArray)) # 80

            # action = np.argmin(matchingStateArray[:,-1])
            # action = random.choice(range(matchingStateArray.shape[0]))
            action = agent.take_action(matchingStateArray,dayIndex)

            rightDriver = candidateList[action]
            rightRegion = env.regionList[order.oriRegion]
            state = stateArray[action]  # 272
            matchingState = matchingStateArray # 280
            wt = actionStateArray[action,7]
            dt = actionStateArray[action,6]
            trs = int(math.ceil(wt + dt))
            meanwt = env.regionList[order.oriRegion].meanwt
            cost = env.cal_cost(order, candidateList[action])
            if maxwt < wt:
                maxwt = wt
            reward = env.cal_reward(wt,meanwt,trs,cost)


            d = DispatchSolution()
            d.add_driver_ID(rightDriver.driverID)
            d.add_state(state)
            d.add_matchingState(matchingState)
            d.add_trs(trs)
            d.add_action(action)
            d.add_reward(reward)
            d.add_cost(cost)
            dDict[rightDriver] = d


            rightDriver.accept_order(trs,order.destLoc,reward,cost)
            rightRegion.accept_order(wt)

        env.step(dDict,replayBuffer)
        daymaxwtList.append(maxwt)
        T = T + 1


        #if len(replayBuffer.buffer) >= 1000:
        # if T == maxTime - 1:
        #     updateRound += 1
        #     for _ in range(20):
        #         matchingState,state, action, reward,cost,nextState = replayBuffer.sample()
        #         agent.update_theta(matchingState,state,action,reward,cost,nextState,updateRound)
        # if (T % 40 == 0) and (T / 40) % 2 == 1:
        #     updateRound += 1
        #     matchingState, state, action, reward, cost, nextState = replayBuffer.sample()
        #     agent.update_lagrange(matchingState, state, action, reward, cost, nextState, updateRound)

    for driver in env.driverList:
        rewardList.append(driver.reward)
        costList.append(driver.cost)
    for region in env.regionList:
        wtList.append(region.dayAccwt)
        if region.dayaccOrder == 0:
            orderList.append(1)
        else:
            orderList.append(region.dayaccOrder)
        if len(region.wtList) < 2:
            fairness = 0
        else:
            fairness = statistics.variance(region.wtList)
        fairnessList.append(fairness)
    regionMeanList = [x / y for x, y in zip(wtList, orderList)]

    dayReward = sum(rewardList)
    meanReward = sum(rewardList) / sum(orderList)
    dayCost = sum(costList)
    meanCost = sum(costList) / sum(orderList)
    daywt = sum(wtList)
    dayMeanwt = daywt / sum(orderList)
    dayInnerRegionfairnesswt = fairnessList
    dayRegionMeanwt = regionMeanList
    regionFairnessMatrix.append(dayInnerRegionfairnesswt)
    regionMeanMatrix.append(dayRegionMeanwt)

    filterRegionMeanList = [x for x in regionMeanList if x != 0]
    dayIntraRegionfairnesswt = statistics.variance(filterRegionMeanList)



    print(f'Day {dayIndex}: mean reward: {meanReward}.')
    print(f'Day {dayIndex}: mean cost: {meanCost}.')
    print(f'Day {dayIndex}: day cost: {dayCost}.')
    # print(f'Day {dayIndex}: day waiting time: {daywt}.')
    print(f'Day {dayIndex}: day mean waiting time: {dayMeanwt}.')
    print(f'Day {dayIndex}: day fairness between regions: {dayIntraRegionfairnesswt}.')
    print(f'Day {dayIndex}:max waiting time: {max(wtList)}')
    meanRewardList.append(meanReward)
    meanCostList.append(meanCost)
    dayCostList.append(dayCost)
    dayMeanwtList.append(dayMeanwt)
    dayIntraRegionfairnesswtList.append(dayIntraRegionfairnesswt)

    dayIndex += 1

print(f'mean reward: {statistics.mean(meanRewardList)}.')
print(f'mean cost: {statistics.mean(meanCostList)}.')
print(f'day cost: {statistics.mean(dayCostList)}.')
# print(f'Day {dayIndex}: day waiting time: {daywt}.')
print(f'day mean waiting time: {statistics.mean(dayMeanwtList)}.')
print(f'day fairness between regions: {statistics.mean(dayIntraRegionfairnesswtList)}.')
print(f'max waiting time: {max(daymaxwtList)}')
print(f'day max waiting time: {statistics.mean(daymaxwtList)}.')


regionFairnessMatrix = np.mean(np.array(regionFairnessMatrix,dtype=float),axis = 0)
regionMeanMatrix = np.mean(np.array(regionMeanMatrix,dtype=float),axis = 0)

with open('../rebuttal/review 2 question 1/LagTRPO.pkl', 'wb') as file:
     pickle.dump(np.array(regionMeanMatrix), file)

