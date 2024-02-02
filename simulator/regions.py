


class Region(object):
    def __init__(self,regionID,regionNum):
        self.regionID = regionID

        self.firstNeighbors = []
        self.secondNeighbors = []
        self.thirdNeighbors = []

        self.cityDay = 0
        self.cityTime = 0

        self.meanwt = 0

        self.regionNum = regionNum
        self.driverList = []
        self.orderList = []

        self.accWaitingTime = 0
        self.accOrder = 0

    def set_neighbors(self):
        x = self.regionID % 10
        y = int(self.regionID / 10)
        for i in range(self.regionNum):
            x1 = (i % 10)
            y1 = int(i/10)
            if pow(x - x1,2) + pow(y - y1,2) <= 2:
                self.firstNeighbors.append(i)
            elif pow(x - x1,2) + pow(y - y1,2) <= 8:
                self.secondNeighbors.append(i)
            elif pow(x - x1,2) + pow(y - y1,2) <= 18:
                self.thirdNeighbors.append(i)
            else:
                pass
        self.thirdNeighbors = self.thirdNeighbors + self.secondNeighbors + self.firstNeighbors
        self.secondNeighbors = self.secondNeighbors + self.firstNeighbors

        self.neighborLevel = [self.firstNeighbors, self.secondNeighbors, self.thirdNeighbors, list(range(100))]

    def set_day_info(self, day):
        self.cityDay = day

    def reset_region_info(self):
        self.cityTime = 0
        self.driverList = []
        self.orderList = []

    def add_driver(self,driver):
        self.driverList.append(driver)

    def add_order(self,order):
        self.orderList.append(order)

    def accept_order(self,wt):
        self.accWaitingTime += wt
        self.accOrder += 1

    def step_update_region_info(self):
        self.cityTime += 1
        self.driverList = []
        self.orderList = []
        if self.accOrder > 0:
            self.meanwt = self.accWaitingTime / self.accOrder




