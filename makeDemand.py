import numpy as np
import pandas as pd

import argparse

import matplotlib.pyplot as plt

JULY_DAYS = 31
JUNE_DAYS = 30

class Resampling:  # resamples data to proper time resolution
    def __init__(self, data, t, tnew):
        self.data = data
        self.t = t
        self.tnew = tnew

    def downsampling(self, data, t, tnew):
        t_ratio = tnew // t
        downsampled = np.zeros((np.size(data, 0), int(np.size(data, 1) / t_ratio)))
        #downsampled = np.zeros((np.size(data, 0), 3673))
        for i in range(np.size(downsampled, 0)):
            for j in range(np.size(downsampled, 1)):
                downsampled[i][j] = np.average(data[i][j * t_ratio:(j + 1) * t_ratio])

        return downsampled

    def upsampling(self, data, t, tnew, new_num_col):
        steps_per_day = int(1440 / t)
        # num_days = int((np.size(data) * t) / 1440)
        one_day = np.zeros((np.size(data, 0), steps_per_day))
        one_day_std = np.zeros((np.size(data, 0), steps_per_day))
        for a in range(np.size(one_day, 0)):
            for b in range(np.size(one_day, 1)):
                sub_array = data[a, b::steps_per_day]
                one_day[a, b] = np.mean(sub_array)
                one_day_std[a, b] = np.std(sub_array)

        t_ratio = t // tnew
        upsampled = np.zeros((np.size(data, 0), new_num_col))
        for i in range(np.size(upsampled, 0)):
            for j in range(np.size(upsampled, 1)):
                upsampled[i][j] = np.random.normal(one_day[i, int((j/t_ratio) % 24)],
                                                   one_day_std[i, int((j/t_ratio) % 24)])
                # upsampled[i][j] = np.random.normal(data[i, int(j / t_ratio)], one_day_std[i, int(j / t_ratio) % 24])
        return upsampled

# End of Resampling definition



def setStorageSolar(pDemandFull, sNormFull, storagePen, solarPen, nodesPen, rootIdx):
    """
    Inputs: pDemandFull - full matrix of real power demanded (nodes X time)
        sNormFull - full matrix of normalized solar data to be scaled and added to power demanded
        storagePen, solarPen, nodesPen - storage, solar, nodes penetration percentages
        rootIdx - index of the root node in the network
    Outputs: netDemandFull - full matrix of real net load
        sGenFull - full matrix of solar generation for storage nodes
        nodesLoad - list of nodes that have non-zero load
        nodesStorage - list of storage nodes
        qmin, qmax, umin, umax
    """

    # Pick storage nodes
    nodesLoad = np.nonzero(pDemandFull[:, 0])[0]
    if pDemandFull[rootIdx, 0] > 0:
        nodesLoad = np.delete(nodesLoad, np.argwhere(nodesLoad == rootIdx)) # remove substation node
    nodesStorage = np.random.choice(nodesLoad, int(np.rint(len(nodesLoad)*nodesPen)), replace=False)
    nodesStorage = np.sort(nodesStorage)

    # Assign solar
    loadSNodes = np.mean(pDemandFull[nodesStorage, :], 1)
    rawSolar = solarPen*sum(np.mean(pDemandFull, 1))
    rawStorage = storagePen*24*sum(np.mean(pDemandFull, 1))
    alphas = loadSNodes/sum(loadSNodes)
    alphas = alphas.reshape(alphas.shape[0], 1)
    netDemandFull = pDemandFull

    # portionSolarPerNode represents the portion of solar in each node.
    #print(alphas.shape)

    portionSolarPerNode = rawSolar * alphas

    #np.reshape(portionSolarPerNode, (34, 0))
    #np.reshape(sNormFull, (0, 3673))

    # sGenFull is the full solar generation for all the nodes that were assigned storage.
    # sNormFull represents the shape of the solar generation over time.
    sGenFull = np.dot(portionSolarPerNode, sNormFull)
    # We then select only the first two months of solar data.
    sGenFull = sGenFull[:, :(JUNE_DAYS + JULY_DAYS) * 24]
    netDemandFull[nodesStorage, :] = netDemandFull[nodesStorage, :] - sGenFull

    # Assign storage
    qmax = np.multiply(rawStorage, alphas)
    qmin = np.zeros_like(qmax)
    umax = qmax/3 # it takes 3 hours to fully charge batteries
    umin = -umax

    return netDemandFull, sGenFull, nodesStorage, qmin, qmax, umin, umax



def aggregateHomes(spot_loads, homeData):
    
    scale = 1 # scale home power to fit with the case data

    dayIdx = np.arange(1, homeData.shape[1], 24*60)
    maxsum = np.zeros((1, homeData.shape[0]))
 
   # for i in range(len(dayIdx)):
    #    maxsum = maxsum + np.max(homeData[:, dayIdx[i]:(dayIdx[i+1]-1)])
    # end of for loop

   # meanPeakLoad = maxsum/len(dayIdx)


    N = len(spot_loads) # total number of buses
    Lbuses = np.where(spot_loads != 0)[0] # indexes of load buses
    print('load bus indices', Lbuses)
    print('number of load buses', len(Lbuses))

    NHomes = np.zeros((N, 1)) # number of homes for each load bus
    pDemand = np.zeros((N, homeData.shape[1])) # power demanded by each load bus

    for i in range(len(Lbuses)):
        currpeak = 0
        currload = np.zeros((1, homeData.shape[1]))
        nextload = np.zeros((1, homeData.shape[1]))
        
        while currpeak < spot_loads[Lbuses[i]]:
            NHomes[i] = NHomes[i] + 1
            currload = nextload
            homeIdx = np.random.randint(homeData.shape[0])#, size=(1,1))

            if np.isnan(np.sum(homeData[homeIdx, :])):
                # check if data is valid.
                # if it is Nan, then resample until it is valid

                while np.isnan(np.sum(homeData[homeIdx, :])):
                    homeIdx = np.random.randint(homeData.shape[0])
                # end while loop

            # end if

            nextload = currload + homeData[homeIdx, :]
            maxsum = 0

            for j in range(len(dayIdx)-1):
                maxsum = maxsum + np.max(nextload[0, dayIdx[j]:(dayIdx[j+1])])
            # end for loop
            
            currpeak = maxsum/len(dayIdx)

        # end while loop
        print('average daily peak for load bus', i, currpeak)
        pDemand[Lbuses[i], :] = currload

    # end for loop

    # pDemand = [zeros(1,size(pDemand,2)); pDemand]; % add root node back

    return pDemand, Lbuses


def makeRawDemandData():
    # load ppc data and initial residential data
    ppc = np.load('data/case123_ppc_reg_pv.npy', allow_pickle=True).item()
    spot_loads = ppc['bus'][:, 2] * 1000  # convert to kW from MW
    print('number of load_buses', np.sum(spot_loads>0))

    # use_data_test.csv represents the entire year 2016
    # The columns of use_data_test.csv are: month day hour minute weekday timestamp 3938 ... ...
    # 3938 ... ... are individual home IDs
    # Each line contains the data for 1 minute.

    # We load the file 'data/use_data_test.csv'
    home_demand = pd.read_csv('data/use_data_test.csv')

    # Select only the lines of data from June (6th month) and July (7th month)
    home_demand_untilJuly = home_demand[home_demand['month'] < 8]
    home_demand = home_demand_untilJuly[home_demand_untilJuly['month'] > 5]

    # print(home_demand)

    # Convert home_demand to a numpy array and remove the columns that specify time (month, day, minute, weekday,
    # timestamp)
    homeData = home_demand.to_numpy()
    print('type of homeData', type(homeData))
    homeData = homeData[:, 7:]
    homeData = homeData.T
    print('shape of homeData', homeData.shape)

    # In the first run, save homeData in 'JuneJuly1min.npy'. Comment out this line in following simulations.
    # np.save('JuneJuly1min.npy', homeData)

    # Keep this line commented out during the first run. Use it to load homeData in all following simulations.
    # homeData = np.load('JuneJuly1min.npy', allow_pickle=True)
    # print('homeData', homeData)

    pDemand, load_buses = aggregateHomes(spot_loads, homeData)

    # In the first run, save data in 'Demand_123bus_JunJul_1min'. Comment out this line in following simulations.
    # np.savez('Demand_123bus_JunJul_1min', pDemand=pDemand, load_buses=load_buses)
    """

    # Keep these 3 lines commented out during the first run. Use them to load pDemand and load_buses in all
    # following simulations.
    data = np.load('Demand_123bus_JunJul_1min.npz')
    pDemand = data['pDemand']
    load_buses = data['load_buses']
    """
    print('number of load buses', np.sum(np.mean(pDemand, axis=1)>0))
    plt.figure(1)
    plt.plot(pDemand[load_buses, 0:60 * 48].T)
    plt.xlabel("Time [min]")
    plt.ylabel("Power Demand [kW]")
    plt.title("Power Demand across 123 nodes over 48h")
    plt.show()

    # Change the resolution to hours instead of minutes:
    resolution_current = 1  # units is minutes
    resolution_new = 60
    t_res = resolution_new / 1
    # units_scale = 1e3 # from watts to kW
    units_scale = 1

    sample = Resampling(pDemand, resolution_current, resolution_new)
    if resolution_current > resolution_new:
        new_num_col = int(pDemand.shape[1] * resolution_current / resolution_new)
        ret = sample.upsampling(pDemand, resolution_current, resolution_new, new_num_col)
    else:
        ret = sample.downsampling(pDemand, resolution_current, resolution_new)
    power = ret / units_scale  # data is now hour resolution and in kW units

    plt.figure(2)
    plt.plot(power[load_buses[0], 0:48].T)
    plt.title('sample 2 days of demand')
    plt.show()

    # load solar data
    solar_dict = np.load('data/solar_ramp_data.npz')

    # aggregate residential data into network
    sr_NormFull = solar_dict['sNormFull']

    # save data when complete
    np.savez('data/Demand_123bus_JunJul_1hr', power=power, sr_NormFull=sr_NormFull)
    print('saved raw data without solar and storage added to network in file', 'Demand_123bus_JunJul_1hr')

    return True


if __name__ == '__main__':
    # input parameters from user
    parser = argparse.ArgumentParser(description='Generate solar and storage demand data')
    parser.add_argument('--seed', default=0, help='random seed')
    parser.add_argument('--storagePen', default=4, help='storage penetration percentage times 10')
    parser.add_argument('--solarPen', default=6, help='solar penetration percentage times 10')
    parser.add_argument('--aggregate', default=0, help='boolean for aggregating homes to network')

    FLAGS, unparsed = parser.parse_known_args()
    print('running with arguments: ({})'.format(FLAGS))

    storagePen = float(FLAGS.storagePen) / 10
    solarPen = float(FLAGS.solarPen) / 10
    seed = int(FLAGS.seed)

    np.random.seed(seed)  # set random seed

    nodesPen = np.max((solarPen, storagePen)) + 0.1
    #nodesPen = 0.9
    rootIdx = 0

    # make raw demand data if not done already
    if FLAGS.aggregate:
        makeRawDemandData()

    # load raw demand data
    data = np.load('data/Demand_123bus_JunJul_1hr.npz')
    power = data['power']
    sr_NormFull = data['sr_NormFull']

    rDemandFull = power/2

    netDemandFull, sGenFull, nodesStorage, qmin, qmax, umin, umax \
        = setStorageSolar(power, sr_NormFull, storagePen, solarPen, nodesPen, rootIdx)

    print("Net demand full: ", netDemandFull)

    np.savez('data/demand_solStor' + str(solarPen) + str(storagePen),
             netDemandFull=netDemandFull, sGenFull=sGenFull, nodesStorage=nodesStorage, qmin=qmin, qmax=qmax,
             umin=umin, umax=umax, storagePen=storagePen, solarPen=solarPen, nodesPen=nodesPen,
             rDemandFull=rDemandFull
             )

    print('Saved to', 'demand_' + str(solarPen) + str(storagePen) + '.npz')
