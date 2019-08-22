# Import packages.
import cvxpy as cp
import numpy as np

import argparse

import matplotlib.pyplot as plt


def Bounds_optimizer(s_imag, d_fair, p_forecast, direction, A, b, load_bool, T = 48, N = 123, Vmax = 105, Vmin = 95,
                     lambda_v = 100, lambda_Pinj=0.01, lambda_direction=1):
    # calculates upper or lower bound depending on sign of objective
    # s_imag is forecasted reactive power demand with shape (N,T)
    # d_fair is fair real power injections that we wish to target
    # d_fair for non control nodes is the same as the forecasted power
    # A is model coefficients shape (, b is intercept
    # N is number of nodes in network
    # direction is +1 for max and -1 for min bound

    # not_load_bool will contain the values between 0 and N that are not in the loads array.
    not_load_bool = np.logical_not(load_bool)

    b = np.tile(b, (1, T))

    # s is split into the real part and imaginary part: s_real & s_imag.
    # The two parts are stacked up real part on top to form the full 's' with shape (2N, T).
    s_real = cp.Variable(shape=(N, T))
    s = cp.vstack([s_real, s_imag])

    v = cp.Variable(shape=(N, T))

    # Constraints to the minimization problem:
    # constraints.append(s_real[load_bool, :] == d_fair[load_bool, :])
    constraints = [s_real[not_load_bool, :] == d_fair[not_load_bool, :],
                   v == A @ s + b
                   ]

    objective = cp.Minimize(lambda_v*cp.sum((cp.pos(v - Vmax) + cp.pos(Vmin - v))**2)
                            + cp.sum((lambda_Pinj*s_real - lambda_Pinj*d_fair)**2)
                            + lambda_direction*cp.sum(cp.pos(direction*(p_forecast - s_real))))
                            # + cp.norm(s_real - d_fair, 'fro'))  # - cp.norm(u_neg-u_pos, "fro"))

    # need to make very punishing when max is less than mid (pforecast/netdemand) and when min is greater than mid

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)

    return s_real.value, v.value, problem.status


def get_EV_charging_window(t_arrival, t_depart, T, charge_power):
    # this is another function you should implement for a separate purpose
    # this function is unrelated to the forecasting

    # t_arrival is a 1D array of arrival times of many EVs
    # same for t_depart, but it is the departure times

    # initialize array to have size T
    total_charge = np.zeros(T)

    # this function needs to add up one array for each element in the input arrays
    for i in range(len(t_arrival)):

        # create array with number of elements = t_depart - t_arrival

        charge = np.zeros(t_depart[i] - t_arrival[i])

        # We are considering the worst case scenario, where all the cars are being charged at max power for the full
        # duration. So all values in the array are set to charge_power
        charge[:] = charge_power
        # Add new array to total_charge starting at index t_arrival and ending at index_t_depart
        #print(total_charge)
        total_charge[t_arrival[i] : t_depart[i]] += charge

    # end for loop

    return total_charge


def get_fair_bounds(p_forecast, std_scaling, error_std, T, t_arrival, t_depart, charge_power):
    # Input error_std and multiply by scaling constant (std_scaling)
    scaled_error_std = error_std * std_scaling

    # Then add the last 2 days (48 points) of the scaled std to the forecast
    new_p_forecast = p_forecast + scaled_error_std[-T:]

    # then calculate EV total charge from function get_EV_charging_window and add that to the new forecast
    # this new value is op_max
    total_charge = get_EV_charging_window(t_arrival, t_depart, T, charge_power)
    op_max = new_p_forecast + total_charge

    # Now subtract the scaled std from the original forecast to get op_min
    op_min = p_forecast - scaled_error_std[-T:]

    return op_max, op_min


if __name__ == '__main__':
    # input parameters from user
    parser = argparse.ArgumentParser(description='Generate solar and storage demand data')
    parser.add_argument('--storagePen', default=4, help='storage penetration percentage times 10')
    parser.add_argument('--solarPen', default=6, help='solar penetration percentage times 10')
    parser.add_argument('--evPen', default=4, help='EV penetration percentage times 10')
    parser.add_argument('--Qcap', default=-0.79, help='power injection in ppc file name')

    FLAGS, unparsed = parser.parse_known_args()
    print('running with arguments: ({})'.format(FLAGS))

    storagePen = float(FLAGS.storagePen) / 10
    solarPen = float(FLAGS.solarPen) / 10
    evPen = float(FLAGS.evPen) / 10
    Qcap = FLAGS.Qcap

    # load load and storage data
    data = np.load('data/demand_solStor' + str(solarPen) + str(storagePen) + '.npz')
    netDemandFull=data['netDemandFull']
    nodesStorage=data['nodesStorage']
    qmin=data['qmin']
    qmax=data['qmax']
    umin=data['umin']
    umax=data['umax']
    N, steps_all = netDemandFull.shape  # number of buses in network and total time periods
    # rDemandFull=l_data['rDemandFull']

    # load network data
    ppc_name = 'data/case123_ppc_reg_pq' + str(Qcap) + '.npy'
    ppc = np.load(ppc_name, allow_pickle=True)[()]
    rDemandFull = ppc['bus'][:, 3]
    if rDemandFull.size < 2*N:
        rDemandFull = 1000 * np.tile(rDemandFull.reshape((rDemandFull.size, 1)), (1, steps_all))

    # load EV data
    spot_loads = ppc['bus'][:, 2] * 1000  # convert to kW from MW
    load_nodes = np.where(spot_loads > 0)[0]
    load_bool = np.zeros(N)
    load_bool[load_nodes] = 1
    load_bool = load_bool == 1
    not_load_bool = np.logical_not(load_bool)
    data = np.load('data/EV_charging_data' + str(evPen) + '.npz', allow_pickle=True)
    start_dict=data['start_dict'][()]
    end_dict=data['end_dict'][()]
    charge_dict=data['charge_dict'][()]
    charging_power=data['charging_power'][()]

    # load model coefficients
    model_name = 'data/SVR_coeffs_solStor' + str(solarPen) + str(storagePen) + 'ppc_reg_pq-0.79.npz'
    data = np.load(model_name)
    coefs_mat = data['coefs_mat']
    intercept_mat = data['intercept_mat']
    presampleIdx = data['presampleIdx']
    startIdx = presampleIdx + 1

    # load forecaster data
    data = np.load('data/forecaster_stats_data.npz')
    error_mean = data['error_mean']
    error_std = data['error_std']

    # constants of problem
    GC_time = 24  # number of steps between each run
    T = 48  # optimization horizon
    #iters = int(np.floor((steps_all - startIdx)/GC_time))

    iters = 2 # testing only 2 iterations for now

    std_scaling = 3
    prices = np.hstack((.202 * np.ones((1, 12)), .463 * np.ones((1, 6)), .202 * np.ones((1, 6))))
    pricesFull = np.reshape(np.tile(prices, (1, 61)), (61 * 24, 1)).T
    # lambda_v = 100 # using default value in function def

    for i in range(iters):
        # Run optimization
        print('running day', i)
        # gather data for current run
        netDemand = netDemandFull[:, (startIdx + GC_time * i):(startIdx + GC_time * i + T)]
        rDemand = rDemandFull[:, (startIdx + GC_time * i):(startIdx + GC_time * i + T)]

        # generate fair target bounds
        d_fair_max = np.zeros(netDemand.shape)
        d_fair_min = np.zeros(netDemand.shape)
        d_fair_max[not_load_bool, :] = netDemand[not_load_bool, :]
        d_fair_min[not_load_bool, :] = netDemand[not_load_bool, :]
        load_node_idx = 0
        for node in load_nodes:
            t_arrival = start_dict[load_node_idx][i]
            t_depart = end_dict[load_node_idx][i]
            load_node_idx += 1
            # print(t_arrival)
            # print(t_depart)
            d_fair_max[node, :], d_fair_min[node, :] = get_fair_bounds(netDemand[node, :], std_scaling, error_std, T,
                                                                       t_arrival, t_depart, charging_power)

        # print(d_fair_max)
        # print(d_fair_min)
        # print(np.sum(d_fair_max - d_fair_min < 0, axis=1))
        print(netDemand.shape)

        Pinj_max, v_at_max, status = Bounds_optimizer(rDemand, d_fair_max, netDemand, 1, coefs_mat, intercept_mat, load_bool,
                                    T, N, Vmax=105, Vmin=95)
        if status != 'optimal':
            print('max bounds optimization status is', status)

        Pinj_min, v_at_min, status = Bounds_optimizer(rDemand, d_fair_min, netDemand, -1, coefs_mat, intercept_mat, load_bool,
                                    T, N, Vmax=105, Vmin=95)
        if status != 'optimal':
            print('min bounds optimization status is', status)

        Pinj_max[np.abs(Pinj_max) < 1e-7] = 0
        Pinj_min[np.abs(Pinj_min) < 1e-7] = 0

        print('bounds shape', Pinj_max.shape)
        print('Maximum allowed power injection', Pinj_max)
        print('voltages at max power injection', v_at_max)
        print(' Minimum allowed power injection', Pinj_min)
        print('voltages at min power injection', v_at_min)

        print(np.sum(Pinj_max - Pinj_min < 0, axis=1))
        # print(np.sum(Pinj_max - Pinj_min > 0, axis=1))

        # start optimization with a single node
        load_node_idx = 3 # test one node first. Will need to make a for loop over all the nodes later
        start_time = start_dict[load_node_idx][i]
        end_time = end_dict[load_node_idx][i]
        charge = charge_dict[load_node_idx][i]

        node_idx = load_nodes[load_node_idx]
        Pinj_max_single = Pinj_max[node_idx, :]
        Pinj_min_single = Pinj_min[node_idx, :]

        print('start time', start_time)
        print('end time', end_time)
        print('charge (desired final soc)', charge)
        print('charging power', charging_power)

        ##### Check to make sure SVR model is good (rerun SVR code that I sent you)

        # make storage optimization function in c_storage_opt.py
        # place function here
        # all constants are inputs to function (
        # the values Pinj_max here -> u_max in the function, Pinj_min -> u_min
        # return values of c, d, Q, ev_c_all








        # After running the c_storage_opt combine c and d and ev_c_all in some way that I will determine later













