from simulator.unitity import *


class Driver(object):
    def __init__(self,driverID,preRegion,oriLoc):
        self.driverID = driverID
        self.preRegion = preRegion
        self.oriLoc = oriLoc

        self.region = 0  # the current region
        self.loc = None # the current lon-lat

        self.cityDay = 0  #
        self.cityTime = 0

        self.state = 1  # 1 active  0 empty
        self.servingTime = 0
        self.destinationLoc = None

        self.serveOrderNum = 0
        self.cost = 0

    def set_day_info(self,day):
        self.cityDay = day

    def reset_driver_info(self):
        self.cityTime = 0
        self.loc = self.oriLoc
        self.region = cal_region(self.loc)

    def accept_order(self,trs,loc,cost):
        self.state = 0 # service order
        self.servingTime = trs
        self.destinationLoc = loc
        self.serveOrderNum += 1
        self.cost += 1

    def step_update_driver_info(self):
        self.cityTime += 1
        if self.servingTime > 0:
            self.servingTime -= 1
            if self.servingTime == 0:  # arrive
                self.state = 1 # active
                self.loc = self.destinationLoc
                self.region = cal_region(self.loc)
                self.destinationLoc = None
                return 1
            else:
                return 0  # not arrive
        else:
            return 0 # active
