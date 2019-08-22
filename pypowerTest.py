# Load results and evaluate performance

# from scipy.io import loadmat
import numpy as np
# import matplotlib.pyplot as plt
from pypower.api import runpf, ppoption

import argparse


def Violation_Process(allVoltage, Vmin, Vmax):
    vGreater = (allVoltage - Vmax).clip(min=0)
    vLess = (Vmin - allVoltage).clip(min=0)
    vio_plus_sum = np.sum(vGreater, axis=1)  # bus# X sum of all over voltage violations
    vio_min_sum = np.sum(vLess, axis=1)  # bus# X sum of all under voltage violations

    vio_plus_max = np.max(vGreater)
    vio_min_max = np.max(vLess)

    vio_timesbig = (vGreater + vLess) > 0
    vio_times = np.sum(vio_timesbig, axis=1)  # bus# X number of times there are violations

    print('Maximum over voltage violation: ', vio_plus_max)
    print('Maximium under voltage violation: ', vio_min_max)
    vioTotal = np.sum(vio_min_sum + vio_plus_sum)
    print('Sum of all voltage violations magnitude: ', vioTotal)
    viosNum = sum(vio_times)
    print('Number of voltage violations: ', viosNum)
    vioAve = vioTotal / viosNum
    print('Average voltage violation magnitude: ', vioAve)

    vio_when = np.sum(vio_timesbig, axis=0)

    return vio_times, vio_plus_sum, vio_min_sum, vio_when


def PF_Sim(ppc, pDemand, rDemand):
    """
    Uses PyPower to calculate PF to simulate node voltages after storage control
    Inputs: ppc - PyPower case dictionary
        GCtime - number of time steps between GC runs
        pDemand/rDemand - true values of real and reactive power demanded
        nodesStorage - list of storage nodes indexes
        U - storage control action
        rootV2 - voltage of the substation node
    Outputs: runVoltage - (buses X time) array of voltages
    """

    ppc['bus'][:, 2] = pDemand/1000 # convert back to Mw from Kw
    # print(pDemand/1000)
    rDemand = rDemand/1000

    #print(rDemand)
    #print(rDemand.shape)

    ppc['bus'][:, 3] = rDemand
    # ppc['bus'][rootIdx,7] = rootVoltage # Doesnt actually set PF root voltage

    # for surpressing runpf output
    ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
    ppc_out = runpf(ppc, ppopt)

    runVoltage = ppc_out[0]['bus'][:, 7]

    return runVoltage


def EvaluateResults(fName, startIdx, ppc_name='data/case123_ppc_reg_pv.npy'):
    print('evaluating file', fName)

    try:
        allData = np.load(fName)
    except:
        print('no file found')
        count = 0
        return np.nan, np.nan, np.nan, np.nan, count, np.nan

    netDemandFull = allData['netDemandFull']

    ppc = np.load(ppc_name, allow_pickle=True)[()]
    # print(ppc['bus'][[12, 32, 72], 1])
    # ppc['bus'][[12, 32, 72], 1] = 1
    # print(ppc['bus'][[12, 32, 72], 1])

    # Define t_idx
    try:
        t_idx = allData['t_idx']
    except:
        print('no t_idx in data')
        t_idx = netDemandFull.shape[1] - startIdx

    # Define Uall and nodesStorage
    try:
        Uall = allData['Uall']
        nodesStorage = allData['nodesStorage']
        rDemandFull = allData['rDemandFull']
    except:
        Uall = np.zeros(netDemandFull.shape) + 1e-9 # no storage simulation
        print('no storage info in data')
        nodesStorage = np.arange(netDemandFull.shape[0])
        # ppc['bus'][[12, 32, 72], 3] = [0.02, 0.02, -0.7]
        rDemandFull = ppc['bus'][:, 3]
        # rDemandFull = allData['netDemandFull'] / 2
        # print(rDemandFull)

    # Define pricesFull
    try:
        pricesFull = allData['pricesFull']
    except:
        print('no price in data')
        prices = np.hstack(
            (.202 * np.ones((1, 12)), .463 * np.ones((1, 6)), .202 * np.ones((1, 6)))) / 4 / 2
        pricesFull = np.reshape(np.tile(prices, (1, 61)), (61 * 24, 1)).T

    # Define forecast_error (only if it is present in the data file)
    try:
        forecast_error = allData['forecast_error']
        print(('forecast_error:', forecast_error))
    except:
        print('no forecast_error in data')

    print('t_idx', t_idx)


    # startIdx = 0 # for pecan street datasets

    # find transformer capacity violations
    # rating = findTransformerSizes(netDemandFull)
    # For solar penetration = 0
    # print('original base case ratings', rating)
    rating = [2500, 37.5, 37.5, 25, 45, 25, 25, 25, 37.5, 25, 37.5, 45, 45, 15, 45, 45, 45, 45, 37.5, 45, 45, 25, 25,
              37.5, 37.5, 45, 25, 25, 25, 25, 45, 25, 25, 37.5, 75, 37.5, 37.5, 25, 37.5, 50, 25, 25, 25, 25, 15, 45,
              45, 100, 37.5, 100, 25, 37.5, 15, 45, 37.5, 45, 45, 100, 45, 37.5, 45, 37.5, 15, 25, 45, 25, 45, 25,
              37.5, 37.5, 37.5, 37.5, 25, 37.5, 37.5, 45, 25, 45, 45, 37.5, 45, 45, 25, 25, 45, 15]
    # """
    rating_n, t_cap_vio, sub_t_cap_vio = afterStorageTransformerSizes(netDemandFull, Uall, nodesStorage, t_idx, startIdx, rating)

    nB, T = netDemandFull.shape
    nS, tt = Uall.shape

    # calculate before storage cost of electricity
    clip_demand = netDemandFull[nodesStorage, startIdx:(startIdx + t_idx)].clip(min=0)
    all_demand = np.sum(clip_demand, axis=0)

    cost_pre = np.dot(pricesFull[:, 0:t_idx], all_demand.T)

    # simulate voltages
    allVoltage = np.zeros((nB, t_idx))
    netDemandFull2 = netDemandFull.copy()
    pDemand = netDemandFull2[:, startIdx:startIdx + t_idx]

    # Is this supposed to be:
    # print('nodes storage', nodesStorage)
    pDemand[nodesStorage, 0:t_idx] += Uall[:, 0:t_idx]
    if rDemandFull.size < 2*nB:
        rDemandFull = 1000 * np.tile(rDemandFull.reshape((rDemandFull.size, 1)), (1,t_idx))

    for i in range(t_idx):
        allVoltage[:, i] = PF_Sim(ppc, pDemand[:, i], rDemandFull[:, i])

    print('maximum voltages', np.max(allVoltage, axis=1))
    print('minimum voltages', np.min(allVoltage, axis=1))
    print('max', np.max(allVoltage))
    print('min', np.min(allVoltage))
    vio_times, vio_plus_sum, vio_min_sum, vio_when = Violation_Process(allVoltage, 0.95, 1.05)

    vio_total_square = np.sum(np.square(vio_min_sum + vio_plus_sum))

    print('-- Desired Traits --')

    print('vio total square', vio_total_square)

    # calculate after storage cost
    all_net = netDemandFull[nodesStorage, startIdx:startIdx + t_idx] + Uall[:, 0:t_idx]

    clip_demand = all_net.clip(min=0)
    all_demand = np.sum(clip_demand, axis=0)

    cost_post = np.dot(pricesFull[:, startIdx:(startIdx + t_idx)], all_demand.T)

    print('before storage cost', cost_pre)
    print('after storage cost', cost_post)
    print('difference', cost_pre - cost_post)
    print('leaf transformer vio', t_cap_vio)
    print('substation transformer vio', sub_t_cap_vio)

    cost_savings = cost_pre - cost_post

    count = 1

    save_name = 'results/results_solStor' + str(allData['solarPen']) + str(allData['storagePen']) + 'ppc_reg_pq-0.79.npz'

    np.savez(save_name,
             vio_total_square=vio_total_square, allVoltage=allVoltage,
             cost_savings=cost_savings,
             t_cap_vio=t_cap_vio, sub_t_cap_vio=sub_t_cap_vio,
             count=count,
             t_idx=t_idx)

    print('saved to', save_name)

    return vio_total_square, cost_savings, t_cap_vio, sub_t_cap_vio, count, t_idx


def findTransformerSizes(netDemandFull):
    # finding transformer sizes (is not totally correct)
    print('finding transformer sizes')

    # typical transformer sizes from palo alto utility (units in kVA):
    typical_transformers = [5, 7.5, 10, 15, 25, 30, 37.5, 45, 50, 75, 100, 112.5, 150, 167, 225, 300, 500, 750, 1000, 1500,
                    2000, 2500]
    transformers_copy = [5, 7.5, 10, 15, 25, 30, 37.5, 45, 50, 75, 100, 112.5, 150, 167, 225, 300, 500, 750, 1000, 1500,
                         2000, 2500]

    max_leafs = np.max(np.abs(netDemandFull), axis=1)
    max_total = np.max(np.abs(np.sum(netDemandFull, axis=0)))
    print('max total power', max_total)
    rating = []
    if max_total < 1500:
        rating.append(1500)
    elif max_total < 2000:
        rating.append(2000)
    elif max_total < 2500:
        rating.append(2500)
    else:
        print('max total power greater than 2500 largest transformer')
        rating.append(2500)
    for c in max_leafs:
        if c < 1e-3:
            continue
        transformers = [5, 7.5, 10, 15, 25, 30, 37.5, 45, 50, 75, 100, 112.5, 150, 167, 225, 300, 500, 750, 1000, 1500,
                    2000, 2500]
        transformers.append(c)
        transformers.sort()
        idx = np.where(transformers == c)[0]
        rating.append(transformers[(int(idx) + 1) if (idx.size > 0) else 1])
        # print('Current transformers vector: ', transformers)
    # print('transformer ratings', rating)

    return rating


def afterStorageTransformerSizes(netDemandFull, Uall, nodesStorage, t_idx, startIdx, rating_o):

    # Is this line supposed to be
    netDemandFull[nodesStorage, startIdx:startIdx + t_idx] = netDemandFull[nodesStorage,
                                                             startIdx:(startIdx + t_idx)] + Uall[:, 0:t_idx]
    netDemandFull = netDemandFull[:, 0:t_idx]

    rating_n = findTransformerSizes(netDemandFull)

    max_consumption = np.max(np.abs(netDemandFull), axis=1)
    max_total = np.max(np.abs(np.sum(netDemandFull, axis=0)))
    max_consumption = np.concatenate((np.reshape(np.array(max_total), 1), max_consumption[max_consumption>1e-3]))
    # print('maximum consumption after storage', max_consumption)

    load_nodes = np.where(np.max(netDemandFull,axis=1) > 1e-3)[0]
    print('percent of leaf transformers with new capacities',
          np.sum(np.array(rating_n) > np.array(rating_o)) / len(load_nodes))

    substation_profile = np.sum(netDemandFull, axis=0)
    t_demands = np.abs(np.vstack((substation_profile, netDemandFull[load_nodes,:])))

    cap_vios = (t_demands - np.tile(np.array(rating_o).reshape((len(rating_o), 1)), (1,t_demands.shape[1]))).clip(min=0)
    sub_t_cap_vio = np.sum(np.square(cap_vios[0, :]))
    # print(sub_t_cap_vio)
    t_cap_vio = np.sum(np.square(cap_vios)) - sub_t_cap_vio
    # print(t_cap_vio)

    max_cap_vio = np.sum(np.square((max_consumption - np.array(rating_o)).clip(min=0)))
    print('maximum capacity violation', max_cap_vio)

    return rating_n, t_cap_vio, sub_t_cap_vio


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate solar and storage demand data')
    parser.add_argument('--seed', default=0, help='random seed')
    parser.add_argument('--storagePen', default=4, help='storage penetration percentage times 10')
    parser.add_argument('--solarPen', default=6, help='solar penetration percentage times 10')
    parser.add_argument('--Qcap', default=-0.79, help='power injection in ppc file name')

    FLAGS, unparsed = parser.parse_known_args()
    print('running with arguments: ({})'.format(FLAGS))
    Qcap = FLAGS.Qcap
    storagePen = float(FLAGS.storagePen) / 10
    solarPen = float(FLAGS.solarPen) / 10

    if storagePen == 0:
        presampleIdx = 0 - 1
        startIdx = presampleIdx + 1
        print('starting at 0 for training models')
    else:
        presampleIdx = 168 - 1
        startIdx = presampleIdx + 1

    # names = ['results/PF_orig_sol0.0.npz', 'results/PF_EV_sol0.0.npz', 'results/PF_EVnoScale_sol0.0.npz',
    #         'results/PF_orig_sol0.6.npz', 'results/PF_EV_sol0.6.npz', 'results/PF_EVnoScale_sol0.6.npz']

    # Put the final file name in here (should not be 1 min resolution, but hourly resolution
    names = ['data/demand_solStor' + str(solarPen) + str(storagePen) + '.npz']

    # switch these comments to change ppc file
    ppc_name = 'data/case123_ppc_reg_pq'+str(Qcap) + '.npy'
    # ppc_name = 'data/case123_ppc_reg_pv.npy'
    # ppc_name = 'data/case123_ppc_none.npy'  # this is the data when voltage regulators are 0

    vio_total_square2 = []
    transformer_capacity_square = []
    arb_prof2 = []
    count2 = []
    t_idxs = []

    l = len(names)

    for fName in names:
        vio_total_square, arb_prof, t_cap_vio, sub_t_cap_vio, count, t_idx = EvaluateResults(fName, startIdx, ppc_name)
        t_idxs.append(t_idx)
        vio_total_square2.append(vio_total_square)
        arb_prof2.append(arb_prof)
        transformer_capacity_square.append(t_cap_vio)
        count2.append(count)

    """
    print('t_idxs', t_idxs)
    print('total vio square', vio_total_square2)
    print('cost savings from battery', arb_prof2)
    print('squared maximum transformer capacity violations', transformer_capacity_square)
    print(count2)
    """
