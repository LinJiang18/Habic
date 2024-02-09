import pandas as pd
import numpy as np
import math
import statistics
from simulator.envs import *
from algorithm.Habic import *
from algorithm.AC import *
import pickle

dayIndex = 0  # the current day
maxTime = 180 # Max dispatch round
maxDay = 50

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
lagLr = 1e-3
limit = 1
lagrange = 1
epochs = 5
eps = 0.2
gamma = 0.95
memorySize = 100000
batchSize = 100

# optimization

# ① 启始位置随机


orderInfo = pd.read_pickle('../data/order.pkl')

driverPreInit = pd.read_pickle('../data/driver_preference.pkl')
driverLocInit = pd.read_pickle('../data/driver_location.pkl')
regionWT = pd.read_pickle('../data/regionMeanWT.pkl')

driverNum = 150

env = Env(driverPreInit,driverLocInit,orderInfo,locRange,driverNum,M,N,maxDay,maxTime)

agent = Habic(stateDim, actionDim, actorLr, criticLr, lagLr, limit, lagrange, epochs, eps, gamma, batchSize)
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
filterRegionMeanMatrix = []


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
        for order in env.dayOrder[T]:
            driverList = env.driver_collect(order)
            if driverList == 0:
                continue
            candidateList = env.generate_candidate_set(order,driverList)
            driverStateArray = env.driver_state_calculate(candidateList)  #
            actionStateArray = env.action_state_calculate(candidateList,order) #
            #contextualArray = env.con_state_calcualte() # 200
            #stateArray = np.hstack((driverStateArray,contextualArray.reshape(1,-1).repeat(driverStateArray.shape[0], 0)))

            stateArray = driverStateArray
            matchingStateArray = np.hstack((stateArray,actionStateArray)) #

            #action = np.argmin(matchingStateArray[:,-1])
            action = agent.take_action(matchingStateArray,dayIndex) #
            #action = random.choice(range(matchingStateArray.shape[0]))

            rightDriver = candidateList[action]
            rightRegion = env.regionList[order.oriRegion]
            state = stateArray[action]  #
            matchingState = matchingStateArray #
            wt = actionStateArray[action,7]
            dt = actionStateArray[action,6]
            trs = int(math.ceil(wt + dt))
            meanwt = env.regionList[order.oriRegion].meanwt
            cost = env.cal_cost(order, candidateList[action])
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

            rightDriver.accept_order(trs,order.destLoc,-wt,cost)
            rightRegion.accept_order(wt)



        env.step(dDict,replayBuffer)
        T += 1

        if T == maxTime - 1:
            updateRound += 1
            for _ in range(10):
                matchingState,state, action, reward,cost,nextState = replayBuffer.sample()
                agent.update_theta(matchingState,state,action,reward,cost,nextState,updateRound)
        # if T == maxTime - 1 and dayIndex % 10 == 0 and dayIndex > 0:
        #     updateRound += 1
        #     for _ in range(10):
        #         matchingState, state, action, reward, cost, nextState = replayBuffer.sample()
        #         agent.update_lagrange(matchingState, state, action, reward, cost, nextState, updateRound)

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
    filterRegionMeanMatrix.append(filterRegionMeanList)
    dayIntraRegionfairnesswt = statistics.variance(filterRegionMeanList)

    print(f'Day {dayIndex}: mean reward: {meanReward}.')
    print(f'Day {dayIndex}: mean cost: {meanCost}.')
    print(f'Day {dayIndex}: day cost: {dayCost}.')
    # print(f'Day {dayIndex}: day waiting time: {daywt}.')
    print(f'Day {dayIndex}: day mean waiting time: {dayMeanwt}.')
    print(f'Day {dayIndex}: day fairness between regions: {dayIntraRegionfairnesswt}.')
    meanRewardList.append(meanReward)
    meanCostList.append(meanCost)
    dayCostList.append(dayCost)
    dayMeanwtList.append(dayMeanwt)
    dayIntraRegionfairnesswtList.append(dayIntraRegionfairnesswt)

    dayIndex += 1

# print(f'mean reward: {statistics.mean(meanRewardList)}.')
# print(f'mean cost: {statistics.mean(meanCostList)}.')
# print(f'day cost: {statistics.mean(dayCostList)}.')
# # print(f'Day {dayIndex}: day waiting time: {daywt}.')
# print(f'day mean waiting time: {statistics.mean(dayMeanwtList)}.')
# print(f'day fairness between regions: {statistics.mean(dayIntraRegionfairnesswtList)}.')

regionFairnessMatrix = np.mean(np.array(regionFairnessMatrix, dtype=float), axis=0)
regionMeanMatrix = np.mean(np.array(regionMeanMatrix, dtype=float), axis=0)

torch.save(agent,'../result/parameter information/agent1.pth')

# with open('../result/training information/agent4/meanRewardList2.pkl', 'wb') as file:
#     pickle.dump(meanRewardList, file)
# with open('../result/training information/agent4/filterRegionMeanMatrix2.pkl', 'wb') as file:
#     pickle.dump(filterRegionMeanMatrix, file)


#
# with open('../result/test information/agent1/regionFairnessMatrix1.pkl', 'wb') as file:
#     pickle.dump(np.array(regionFairnessMatrix), file)

# print('regionMean')
# print(np.array(regionMeanMatrix))
#
# with open('../result/test information/agent1/regionMeanMatrix1.pkl', 'wb') as file:
#     pickle.dump(np.array(regionMeanMatrix), file)



# for driver in env.driverList:
#     rewardList.append(driver.reward)
#     costList.append(driver.cost)
# for region in env.regionList:
#     wtList.append(region.dayAccwt)
#     if region.dayaccOrder == 0:
#         orderList.append(1)
#     else:
#         orderList.append(region.dayaccOrder)
#     if len(region.wtList) < 2:
#         fairness = 0
#     else:
#         fairness = statistics.variance(region.wtList)
#     fairnessList.append(fairness)
# regionMeanList = [x / y for x, y in zip(wtList, orderList)]
#
# dayReward = sum(rewardList)
# meanReward = sum(rewardList) / sum(orderList)
# dayCost = sum(costList)
# meanCost = sum(costList) / sum(orderList)
# daywt = sum(wtList)
#
#
# dayInnerRegionfairnesswt = fairnessList
# dayRegionMeanwt = regionMeanList
# regionFairnessMatrix.append(dayInnerRegionfairnesswt)
# regionMeanMatrix.append(dayRegionMeanwt)
#
# dayMeanwt = daywt / sum(orderList)
# filterRegionMeanList = [x for x in regionMeanList if x != 0]
# dayIntraRegionfairnesswt = statistics.variance(filterRegionMeanList)
#
# # print(f'Day {dayIndex}: day reward: {dayReward}.')
# print(f'Day {dayIndex}: mean reward: {meanReward}.')
# print(f'Day {dayIndex}: mean cost: {round(meanCost,5)}.')
# print(f'Day {dayIndex}: day cost: {dayCost}.')
# # print(f'Day {dayIndex}: day waiting time: {daywt}.')
# print(f'Day {dayIndex}: day mean waiting time: {dayMeanwt}.')
# print(f'Day {dayIndex}: day fairness between regions: {dayIntraRegionfairnesswt}.')
#
# meanRewardList.append(meanReward)
# meanCostList.append(meanCost)
# dayCostList.append(dayCost)
# dayMeanwtList.append(dayMeanwt)
# dayFairnessList.append(dayIntraRegionfairnesswt)
#
#
# regionFairnessMatrix = np.array(regionFairnessMatrix)
# regionMeanMatrix = np.array(regionMeanMatrix)

