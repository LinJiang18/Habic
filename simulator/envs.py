import pandas as pd
import numpy as np
import statistics
from simulator.regions import *
from simulator.orders import *
from simulator.unitity import *
from simulator.drivers import *

#rewardList = []
#gamma = 0.98

class Env(object):
    def __init__(self,driverPreInit,driverLocInit,orderInfo,locRange,driverNum,M,N,maxDay,maxTime):
        self.driverPreInit = driverPreInit
        self.driverLocInit = driverLocInit
        self.orderInfo = orderInfo
        self.locRange = locRange # locRange = [minlon,maxlon,minlat,maxlat]
        self.length = M
        self.width = N
        self.maxTime = maxTime
        self.maxDay = maxDay

        # 全体时间信息
        self.cityTime = 0 #
        self.cityDay = 0
        self.maxCityTime = maxTime # 从早上6点到晚上10点，共计统计16个小时



        # 全体骑手信息
        self.driverList = []  # 骑手列表
        self.driverDict = {}  # 骑手ID对应的字典
        self.driverNum = driverNum  # 对应的总骑手数目


        # 全体网格信息
        self.M = M  # 结点横轴数
        self.N = N  # 结点纵轴数
        self.regionNum = self.M * self.N # 总的结点数
        self.regionList = [Region(i,self.regionNum) for i in range(self.regionNum)]  # 区域的结点列表
        self.regionDict = {}


        # contextual Info
        self.candidateDriverSize = 20
        self.maxDriverPreNum = 16

        # speed
        self.speed = 30

        #
        self.alpha = 0
        self.gamma = 0.98

    def set_region_info(self,WT):
        for i in range(self.regionNum):
            region = self.regionList[i]
            self.regionDict[i] = region
            region.set_neighbors()
            region.set_region_meanwt(WT[i])

    def set_driver_info(self, driverPreInit, driverLocInit):
        for i in range(self.driverNum):
            driverID = i
            driverPre = driverPreInit[i]
            driverRegion = driverLocInit[i]
            driverLoc = generate_loc(driverRegion)
            driver = Driver(driverID, driverPre, driverLoc)
            self.driverList.append(driver)
            self.driverDict[driverID] = driver

    def set_day_info(self, dayIndex):
        self.cityDay = dayIndex
        for driver in self.driverList:
            driver.set_day_info(dayIndex)
        for region in self.regionList:
            region.set_day_info(dayIndex)

    def reset_clean(self):
        self.cityTime = 0
        self.dayOrder = []
        for driver in self.driverList:
            driver.reset_driver_info()
        for region in self.regionList:
            region.reset_region_info()
        self.boost_one_day_order()
        self.boost_step_order_info(self.cityTime)
        self.boost_step_region_info()

    def boost_step_order_info(self, T):
        stepOrderList = self.dayOrder[T]
        for order in stepOrderList:
            region = order.oriRegion
            self.regionList[region].add_order(order)


    def boost_step_region_info(self):
        for driver in self.driverList:
            if driver.state == 1:
                region = driver.region
                self.regionList[region].add_driver(driver)

    def boost_one_day_order(self):
        dayOrderList = [[] for _ in np.arange(self.maxCityTime)]
        for dayOrder in self.orderInfo[self.cityDay]:
            for order in dayOrder:
                startTime = order[2]
                # orderID,orderDay,orderMin,orderRegion,oriLon,oriLat,destLon,destLat
                # 订单ID(重写)，订单下单日期，订单下单时间（分钟）,订单所在区域，订单起点经度，订单起点纬度，订单终点经度，订单终点纬度
                orderRegion = self.regionList[order[3]]
                dayOrderList[startTime].append(Order(order[0], order[1], order[2], orderRegion, order[4], order[5],
                                                     order[7], order[8]))
        self.dayOrder = dayOrderList



    def driver_collect(self,order):
        orderRegion = order.orderRegion
        neighborLevelIndex = 3
        driverList = []
        neighborList = orderRegion.neighborLevel[neighborLevelIndex]
        for neighbor in neighborList:
            driverList.append(self.regionList[neighbor].driverList)
        driverList = [x for y in driverList for x in y]

        while len(driverList) == 0:
            neighborLevelIndex += 1
            if neighborLevelIndex == 4:
                print('All drivers have been full!')
                return 0
            driverList = []
            neighborList = orderRegion.neighborLevel[neighborLevelIndex]
            for neighbor in neighborList:
                driverList.append(self.regionList[neighbor].driverList)
            driverList = [x for y in driverList for x in y]

        return driverList

    def generate_candidate_set(self,order,driverList):
        disList = []
        for i in range(len(driverList)):
            driver = driverList[i]
            dis = cal_dis(order.oriLoc,driver.loc)
            disList.append((dis,i))
        disList = sorted(disList,key = lambda x:x[0],reverse = False)
        disList = disList[:self.candidateDriverSize]
        disList = [x[1] for x in disList]
        candidateList = [driverList[x] for x in disList]

        return candidateList

    def driver_state_calculate(self,driverList):
        driverArray = np.zeros((len(driverList),self.maxDriverPreNum + 5))
        index = 0
        for driver in driverList:
            region = np.array([driver.region + 1]) # 1
            t = np.array([driver.cityTime])
            lon = round((driver.loc.lon - minlon) / (maxlon - minlon),5)
            lon = np.array([lon])
            lat = round((driver.loc.lat - minlat) / (maxlat - minlat), 5)
            lat = np.array([lat])
            nearwt = 100
            for order in self.regionList[driver.region].orderList:
                orderDis = cal_dis(driver.loc,order.oriLoc)
                if orderDis <= nearwt:
                    nearwt = orderDis
            nearwt = np.array([nearwt])
            preRegion = np.pad(driver.preRegion, (0, self.maxDriverPreNum - len(driver.preRegion)), 'constant') # 62
            driverState = np.concatenate((region,lon,lat,t,nearwt,preRegion))
            driverArray[index,:] = driverState
            index += 1
        return driverArray

    # def driver_next_state_calculate(self,driver,order,trs):
    #     region = np.array([order.destRegion + 1])
    #     t = np.array([driver.cityTime + trs])
    #     lon = round((order.destLoc.lon - minlon) / (maxlon - minlon), 5)
    #     lon = np.array([lon])
    #     lat = round((order.destLoc.lon - minlat) / (maxlat - minlat), 5)
    #     lat = np.array([lat])
    #     preRegion = np.pad(driver.preRegion, (0, self.maxDriverPreNum - len(driver.preRegion)), 'constant')
    #     nextDriverState = np.concatenate((region,t,lon,lat,preRegion))
    #     return nextDriverState

    def action_state_calculate(self,driverList,order):
        actionArray = np.zeros((len(driverList),8))
        oriRegion = np.array([order.oriRegion + 1])
        destRegion = np.array([order.destRegion + 1])
        oriLon = round((order.oriLoc.lon - minlon) / (maxlon - minlon), 5)
        oriLon = np.array([oriLon])
        oriLat = round((order.oriLoc.lat - minlat) / (maxlat - minlat), 5)
        oriLat = np.array([oriLat])
        destLon = round((order.destLoc.lon - minlon) / (maxlon - minlon), 5)
        destLon = np.array([destLon])
        destLat = round((order.destLoc.lat - minlat) / (maxlat - minlat), 5)
        destLat = np.array([destLat])
        ds = np.array([(cal_dis(order.oriLoc,order.destLoc) / self.speed) * 60])
        index = 0
        for driver in driverList:
            wt = np.array([(cal_dis(driver.loc,order.oriLoc) / self.speed) * 60])
            orderState = np.concatenate((oriRegion,destRegion,oriLon,oriLat,destLon,destLat,ds,wt))
            actionArray[index,:] = orderState
            index += 1
        return actionArray

    def con_state_calcualte(self):
        supplyDemandList = []
        for region in self.regionList:
            supply = len(region.driverList)
            demand = len(region.orderList)
            supplyDemandList.append(supply)
            supplyDemandList.append(demand)
        return np.array(supplyDemandList)

    def cal_reward(self,wt,meanwt,trs,cost):
        reward = ((-wt) - self.alpha * abs(wt - meanwt))
        return reward
        # rewardList.append(reward)
        # if len(rewardList) == 1:
        #     reward = 0
        # else:
        #     reward = reward/statistics.stdev(rewardList)
         #   rt = reward / trs
         #   gammaReward = 0
         #   for t in range(trs):
         #       gammaReward += rt * pow(self.gamma,t)



    def cal_cost(self,order,driver):
        dest = order.destRegion
        if dest in driver.preRegion:
            cost = 1
        else:
            cost = 0
        return cost


    def step(self,dDict,replayBuffer):
        updateDriverList = []
        for driver in self.driverList:
            symbol = driver.step_update_driver_info()
            if symbol == 1:
                updateDriverList.append(driver)
        for region in self.regionList:
            region.step_update_region_info()
        if self.cityTime < self.maxCityTime - 1:
            self.boost_step_order_info(self.cityTime + 1)
            self.boost_step_region_info()  # update supply and demand
        for driver in updateDriverList:
            region = np.array([driver.region + 1])  # 1
            lon = np.array([driver.loc.lon])  # 1
            lat = np.array([driver.loc.lat])  # 1
            t = np.array([driver.cityTime])
            nearwt = 100
            for order in self.regionList[driver.region].orderList:
                orderWT = (cal_dis(driver.loc, order.oriLoc) / self.speed) * 60
                if orderWT <= nearwt:
                    nearwt = orderWT
            nearwt = np.array([nearwt])
            preRegion = np.pad(driver.preRegion, (0, self.maxDriverPreNum - len(driver.preRegion)),
                               'constant')  # 40
            driverState = np.concatenate((region, lon, lat,t,nearwt,preRegion))
            # contextualState = self.con_state_calcualte()
            #next_driver_state = np.concatenate((driverState,contextualState))
            next_driver_state = driverState
            dDict[driver].add_nextState(next_driver_state)
            replayBuffer.add(dDict[driver].matchingState, dDict[driver].state, dDict[driver].action,
                             dDict[driver].reward, dDict[driver].cost,dDict[driver].nextState)
            dDict.pop(driver, None)
        self.cityTime += 1



    # def cal_slot_info(self):
    #
    #     slotReward = 0
    #     slotCost = 0
    #     slotNum = 0
    #     slotwt = 0
    #     slotRFList = []
    #     slotIFList = []
    #
    #     for driver in self.driverList:
    #         slotReward += driver.slotReward
    #         slotCost += driver.slotCost
    #     for region in self.regionList:
    #         slotNum += region.slotnum
    #         slotwt += region.slotwt
    #         if len(region.slotwtList) > 1:
    #             slotRFList.append(statistics.mean(region.slotwtList))
    #         elif len(region.slotwtList) == 1:
    #             slotRFList.append(region.slotwtList[0])
    #         else:
    #             slotRFList.append(0)
    #         if len(region.slotwtList) > 1:
    #             slotIFList.append(statistics.variance(region.slotwtList))
    #         else:
    #             slotIFList.append(0)
    #
    #     return round(slotReward/slotNum,5),round(slotCost/slotNum,5),\
    #             round(slotwt/slotNum,5),round(statistics.variance(slotRFList),5),\
    #             round(statistics.mean(slotIFList),5)
    #
    # def update_slot_info(self):
    #     for driver in self.driverList:
    #         driver.slotReward = 0
    #         driver.slotCost = 0
    #     for region in self.regionList:
    #         region.slotwt = 0
    #         region.slotnum = 0
























