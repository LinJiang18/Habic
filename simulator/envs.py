import pandas as pd
import numpy as np
from simulator.regions import *
from simulator.orders import *
from simulator.unitity import *
from simulator.drivers import *



class Env(object):
    def __init__(self,driverPreInit,driverLocInit,orderInfo,locRange,driverNum,M,N,maxDay,maxTime):
        self.driverPreInit = driverPreInit
        self.driverLocInit = driverLocInit
        self.orderInfo = orderInfo
        self.locRange = locRange
        self.length = M
        self.width = N
        self.maxDay = maxDay
        self.maxTime = maxTime

        # 全体时间信息
        self.cityDay = 0  #
        self.cityTime = 0 #
        self.maxCityTime = maxTime # 从早上6点到晚上10点，共计统计16个小时
        self.maxCityDay = maxDay # 最大的城市天数

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


        # 时间步信息
        self.slotOrder = []

        # contextual Info
        self.candidateDriverSize = 20
        self.maxDriverPreNum = 40

        # speed
        self.speed = 30

        #
        self.alpha = 0.1
        self.gamma = 0.98




    def set_region_info(self):
        for i in range(self.regionNum):
            region = self.regionList[i]
            self.regionDict[i] = region
            region.set_neighbors()

    def set_driver_info(self,driverPreInit,driverLocInit):
        for i in range(self.driverNum):
            driverID = i
            driverPre = driverPreInit[i]
            driverRegion = driverLocInit[i]
            driverLoc = generate_loc(driverRegion)
            driver = Driver(driverID,driverPre,driverLoc)
            self.driverList.append(driver)
            self.driverDict[driverID] = driver

    def set_day_info(self,dayIndex):
        self.cityDay = dayIndex
        for driver in self.driverList:
            driver.set_day_info(dayIndex)
        for region in self.regionList:
            region.set_day_info(region)

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



    def boost_step_region_info(self):
        for driver in self.driverList:
            if driver.state == 1:
                region = driver.region
                self.regionList[region].add_driver(driver)

    def boost_step_order_info(self,T):
        stepOrderList = self.dayOrder[T]
        for order in stepOrderList:
            region = order.oriRegion
            self.regionList[region].add_order(order)



    def boost_one_day_order(self):
        dayOrderList = [[] for _ in np.arange(self.maxCityTime)]
        for dayOrder in self.orderInfo[self.cityDay]:
            for order in dayOrder:
                startTime = order[2]
                # orderID,orderDay,orderMin,orderRegion,oriLon,oriLat,destLon,destLat
                # 订单ID(重写)，订单下单日期，订单下单时间（分钟）,订单所在区域，订单起点经度，订单起点纬度，订单终点经度，订单终点纬度
                orderRegion = self.regionList[order[3]]
                dayOrderList[startTime].append(Order(order[0],order[1],order[2],orderRegion,order[4],order[5],
                                                     order[6],order[7]))
        self.dayOrder = dayOrderList

    def driver_collect(self,order):
        orderRegion = order.orderRegion
        neighborLevelIndex = 0
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
        driverArray = np.zeros((len(driverList),44))
        index = 0
        for driver in driverList:
            region = np.array([driver.region + 1]) # 1
            t = np.array([driver.cityTime])
            lon = np.array([driver.loc.lon]) # 1
            lat = np.array([driver.loc.lat]) # 1
            preRegion = np.pad(driver.preRegion, (0, self.maxDriverPreNum - len(driver.preRegion)), 'constant') # 40
            driverState = np.concatenate((region,t,lon,lat,preRegion))
            driverArray[index,:] = driverState
            index += 1
        return driverArray

    def driver_next_state_calculate(self,driver,order,trs):
        region = np.array([order.destRegion + 1])
        t = np.array([driver.cityTime + trs])
        lon = np.array([order.destLoc.lon])
        lat = np.array([order.destLoc.lat])
        preRegion = np.pad(driver.preRegion, (0, self.maxDriverPreNum - len(driver.preRegion)), 'constant')
        nextDriverState = np.concatenate((region,t,lon,lat,preRegion))
        return nextDriverState

    def action_state_calculate(self,driverList,order):
        actionArray = np.zeros((len(driverList),8))
        oriRegion = np.array([order.oriRegion + 1])
        destRegion = np.array([order.destRegion + 1])
        oriLon = np.array([order.oriLoc.lon])
        oriLat = np.array([order.oriLoc.lat])
        destLon = np.array([order.destLoc.lon])
        destLat = np.array([order.destLoc.lat])
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

    def cal_reward(self,wt,meanwt,trs):
        reward = (-wt) - self.alpha * abs(wt - meanwt)
        rt = reward / trs
        gammaReward = 0
        for t in range(trs):
            gammaReward += rt * pow(self.gamma,t)
        return gammaReward


    def cal_cost(self,order,driver):
        dest = order.destRegion
        if dest in driver.preRegion:
            cost = 0
        else:
            cost = 1
        return cost


    def step(self,dDict,replayBuffer):
        updateDriverList = []
        for driver in self.driverList:
            symbol = driver.step_update_driver_info()
            if symbol == 1:
                updateDriverList.append(driver)
        for region in self.regionList:
            region.step_update_region_info()
        self.boost_step_order_info(self.cityTime + 1)
        self.boost_step_region_info()  # update supply and demand
        for driver in updateDriverList:
            region = np.array([driver.region + 1])  # 1
            t = np.array([driver.cityTime])
            lon = np.array([driver.loc.lon])  # 1
            lat = np.array([driver.loc.lat])  # 1
            preRegion = np.pad(driver.preRegion, (0, self.maxDriverPreNum - len(driver.preRegion)),
                               'constant')  # 40
            driverState = np.concatenate((region, t, lon, lat, preRegion))
            contextualState = self.con_state_calcualte()
            next_driver_state = np.concatenate((driverState,contextualState))
            dDict[driver].add_nextState(next_driver_state)
            replayBuffer.add(dDict[driver].matchingState, dDict[driver].state, dDict[driver].action,
                             dDict[driver].reward, dDict[driver].cost,dDict[driver].nextState)
            dDict.pop(driver, None)

        self.cityTime += 1






















