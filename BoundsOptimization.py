# Import packages.
import cvxpy as cp
import numpy as np

import argparse

import matplotlib.pyplot as plt

def local_optimization_solve(P, lambda_b, lambda_cap, T, Qmin, Qmax, Qo,
                       cmax, dmax, y_l, y_d, y_c,
                       lambda_e, umin, umax,
                       start_time, end_time, charge, charging_power):

    # introduce many new variables to represent ev charging
    # it is possible to use a for loop and a dictionary to make many new variables
    # ex. for i in range: ev_c_dict[i] = cp.Variable
    # Each variable should consist of a charge profile for each EV that only exists between start_time to end_time
    # (so has length end-start)
    # start_time and end_time are arrays that contain the time for each EV (one value in the arrays for each EV)
    # therefore the number of EV variables = len(start_time)
    ev_c_dict = {}

    ev_q0 = 0
    ev_q_dict = {}

    for i in range(len(start_time)):
        ev_c_dict[i] = cp.Variable(T)
        ev_q_dict[i] = cp.Variable(T+1)
    # end for loop

    # make a value called ev_c_all with shape (1, T) = combined ev_c sum where the start/end indexes are respected
    # something like for i in range: ev_c_all[start_time[i]:end_time[i]] += ev_c_dict[i]
    # ev_c_all = cp.Variable(T)
    # Q is battery charge level. First entry must be placed to a constant input
    Q = cp.Variable(T + 1)
    # c is charging variable
    c = cp.Variable(T)
    # d is discharging variable
    d = cp.Variable(T)

    soc_constraints = [
        c >= 0,
        c <= np.tile(cmax, T),
        d >= 0,
        d <= np.tile(dmax, T),
        Q >= Qmin,
        Q <= Qmax,
        # modify constraints P + c - d <= u_max and u_min to be P + c - d + ev_c_all
        # P + c - d + ev_c_all >= umin,
        # P + c - d + ev_c_all <= umax,
        # np.tile(u_min,T) <= P+c-d, P+c-d <= np.tile(u_max,T),  # moved to soft constraint
        Q[0] == Qo,
        Q[1:(T+1)] == y_l * Q[0:T] + y_c * c[0:T] - y_d * d[0:T]
    ]

    # introduce many new variables ev_q_dict to represent EV SOC for each car
    # you can add constraints in a for loop like: for i in range: constraints.append( constraint )
    # each variable in ev_q_dict should be equal to the previous SOC + ev_c very similarly to the current storage model Q
    # add constraints ev_q_dict[i] = ev_q0 = 0
    # you can assume the efficiency values are 1 like with the storage
    # add constraint ev_q_dict[i][end time] = charge for each car in ev charge

    # ev_c_all == 0 when not charging

    # print('len of dict',len(ev_q_dict))

    for i in range(len(start_time)):
        ev_times_not = np.ones(T, dtype=int)
        ev_times_not[start_time[i]:end_time[i]] = 0
        ev_times_not = ev_times_not == 1
        soc_constraints.append(ev_c_dict[i][ev_times_not] == 0)
        soc_constraints.append(ev_c_dict[i] >= 0)
        soc_constraints.append(ev_q_dict[i][start_time[i]] == ev_q0)
        #print(charge[i])
        #print(ev_q_dict[i][-1])
        soc_constraints.append(ev_q_dict[i][end_time[i]+1] == charge[i])
        # add constraints where each variable in ev_c_dict is between 0 and ev_cmax = charging_power
        soc_constraints.append(ev_c_dict[i] >= 0)
        soc_constraints.append(ev_c_dict[i] <= charging_power)
        soc_constraints.append(ev_q_dict[i][1:] == y_l * ev_q_dict[i][0:-1] + y_c * ev_c_dict[i])

    # end for loop

    if len(start_time) > 0:
        ev_c_all = ev_c_dict[0]
        if len(start_time) > 1:
            for i in range(1,len(start_time)):
                ev_c_all += ev_c_dict[i]
    else:
        ev_c_all = cp.Variable(1)
        soc_constraints.append(ev_c_all == 0)


    objective = cp.Minimize(
            lambda_e.reshape((1, lambda_e.size)) @ cp.reshape(cp.pos(P + c - d + ev_c_all), (T, 1))
            + lambda_cap * cp.sum((cp.pos(P + c - d + ev_c_all - umax) + cp.pos(umin - (P + c - d + ev_c_all))) ** 2)
            + lambda_b * cp.sum(c + d + ev_c_all)
            )

    prob = cp.Problem(objective, soc_constraints)
    prob.solve(solver=cp.ECOS)

    # print('P', P)
    # print(np.sum(P))
    # print(P + c.value - d.value + ev_c_all.value)

    try:
        bounds_cost = lambda_cap * np.sum((np.maximum(P + c.value - d.value + ev_c_all.value - umax, 0)
                            + np.maximum(umin - (P + c.value - d.value + ev_c_all.value), 0)) ** 2)
    except:
        print('optimization cost evaluation failed')
        bounds_cost = np.nan

    # print("cost e", lambda_e.reshape((1, lambda_e.size)) @ np.maximum(P + c.value - d.value + ev_c_all.value, 0).reshape((T, 1)) )
    # print('cost of bounds', bounds_cost )

    return c.value, d.value, Q.value, ev_c_all.value, ev_c_dict, ev_q_dict, bounds_cost, prob.status



def Bounds_optimizer(s_imag, d_fair, p_forecast, direction, A, b, load_bool, T = 48, N = 123, Vmax = 105., Vmin = 95.,
                     lambda_v = 100, lambda_Pinj=0.01, lambda_direction=1):
    # calculates upper or lower bound depending on sign of objective
    # s_imag is forecasted reactive power demand with shape (N,T)
    # d_fair is fair real power injections that we wish to target
    # d_fair for non control nodes is the same as the forecasted power
    # A is model coefficients shape (, b is intercept
    # N is number of nodes in network
    # direction is +1 for max and -1 for min bound
    # b = intercept_mat from the load of model coefficients (SVR_coeffs_solStor...)

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

    objective = cp.Minimize(lambda_v * cp.sum((cp.pos(v - Vmax) + cp.pos(Vmin - v))**2)
                            + cp.sum((lambda_Pinj * s_real - lambda_Pinj * d_fair)**2)
                            + lambda_direction * cp.sum(cp.pos(direction * (p_forecast - s_real))))
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

        if t_depart[i] < t_arrival[i]:
            print(t_depart[i], t_arrival[i])
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

    return op_min, op_max

def limit_fair_bounds(op_min, op_max, load_nodes=None, rating=None):
    # limit the size of the fair bounds by static transformer ratings

    n, t = op_min.shape
    nl = len(load_nodes)
    if rating is None:
        # For solar penetration = 0
        # substation rating has been removed
        # rating = np.array([37.5, 37.5, 25, 45, 25, 25, 25, 37.5, 25, 37.5, 45, 45, 15, 45, 45, 45, 45, 37.5, 45, 45, 25, 25,
        #          37.5, 37.5, 45, 25, 25, 25, 25, 45, 25, 25, 37.5, 75, 37.5, 37.5, 25, 37.5, 50, 25, 25, 25, 25, 15, 45,
        #          45, 100, 37.5, 100, 25, 37.5, 15, 45, 37.5, 45, 45, 100, 45, 37.5, 45, 37.5, 15, 25, 45, 25, 45, 25,
        #          37.5, 37.5, 37.5, 37.5, 25, 37.5, 37.5, 45, 25, 45, 45, 37.5, 45, 45, 25, 25, 45, 15])
        rating = np.array([45, 50, 30, 50, 30, 30, 30, 45, 30, 50, 50, 50, 25, 45, 50, 50, 50, 45, 45, 50, 30, 30, 45, 50,
                  50, 30, 30, 25, 30, 50, 30, 30, 45, 100, 45, 45, 30, 45, 50, 30, 30, 30, 30, 25, 45, 75, 112.5, 45, 112.5,
                  30, 50, 15, 50, 45, 45, 45, 100, 50, 45, 50, 50, 25, 30, 50, 30, 50, 30, 50, 45, 45, 50, 30, 45, 45,
                  50, 25, 50, 50, 45, 50, 50, 30, 30, 50, 25])

    if rating.size < op_min.size:
        rating_nl = np.tile(rating.reshape((nl, 1)), (1, t))
        rating = np.zeros((n, t))
        rating[load_nodes, :] = rating_nl

    # print(rating.shape)
    op_min = np.maximum(op_min, -rating)
    op_max = np.minimum(op_max, rating)
    # print(op_min.shape)

    return op_min, op_max


if __name__ == '__main__':
    # input parameters from user
    parser = argparse.ArgumentParser(description='Generate solar and storage demand data')
    parser.add_argument('--storagePen', default=4, help='storage penetration percentage times 10')
    parser.add_argument('--solarPen', default=6, help='solar penetration percentage times 10')
    parser.add_argument('--evPen', default=4, help='EV penetration percentage times 10')
    parser.add_argument('--Qcap', default=-0.79, help='power injection in ppc file name')
    parser.add_argument('--bounds', default=1, help='1 for dynamic bounds, 0 for infinite bounds, 2 for static bounds')

    FLAGS, unparsed = parser.parse_known_args()
    print('running with arguments: ({})'.format(FLAGS))

    storagePen = float(FLAGS.storagePen) / 10
    solarPen = float(FLAGS.solarPen) / 10
    evPen = float(FLAGS.evPen) / 10
    Qcap = FLAGS.Qcap
    bounds_flag = int(FLAGS.bounds)

    # load load and storage data
    data = np.load('data/demand_solStorEV' + str(solarPen) + str(storagePen) + str(evPen) + '.npz')
    netDemandFull=data['netDemandFull']
    nodesStorage=data['nodesStorage']
    qminS=data['qmin']
    qmaxS=data['qmax']
    dmaxS=-data['umin'] # charge and discharge variables separated
    cmaxS=data['umax']
    # print('battery specs: qmax, dmax, cmax', qmax, dmax, cmax)
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
    """
    if evPen == 0:
        evPen_flag = 1
        evPen = 0.1
    """
    data = np.load('data/EV_charging_data' + str(evPen) + '.npz', allow_pickle=True)
    start_dict = data['start_dict'][()]
    end_dict = data['end_dict'][()]
    charge_dict = data['charge_dict'][()]
    charging_power = data['charging_power'][()]
    print('charging power', charging_power)

    # Fill non storage nodes with 0
    qmin = np.zeros(N)
    qmax = np.zeros(N)
    cmax = np.zeros(N)
    dmax = np.zeros(N)
    qmin[nodesStorage] = qminS.flatten()
    qmax[nodesStorage] = qmaxS.flatten()
    cmax[nodesStorage] = cmaxS.flatten()
    dmax[nodesStorage] = dmaxS.flatten()
    qmin = qmin[load_nodes]
    qmax = qmax[load_nodes]
    cmax = cmax[load_nodes]
    dmax = dmax[load_nodes]

    # load model coefficients
    #model_name = 'data/SVR_coeffs_solStor' + str(solarPen) + str(storagePen) + 'ppc_reg_pq-0.79.npz'
    model_name = 'data/SVR_coeffs_solStor' + str(solarPen) + str(0.0) + 'ppc_reg_pq-0.79.npz' # try model with no DERs
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
    iters = int(np.floor((steps_all - startIdx)/GC_time) - 1)
    # iters = 2 # testing only 2 iterations for now

    std_scaling = 1
    price_shape = np.hstack((.202 * np.ones((1, 16)), .463 * np.ones((1, 5)), .202 * np.ones((1, 3))))
    pricesFull = np.reshape(np.tile(price_shape, (1, 61)), (61 * 24, 1)).T
    # lambda_v = 100 # using default value in function def

    # Constants for optimizations
    lambda_v = 100
    lambda_Pinj = 0.1
    lambda_direction = 1
    lambda_b = 0.01
    #lambda_b = 0
    lambda_cap = 100
    q0 = qmax / 2

    # y are efficiency values
    y_l = 0.9999
    y_d = 1/.99
    y_c = .99

    # print useful info
    print('starting at index', startIdx)
    print('load nodes', load_nodes)
    print('storage nodes', nodesStorage)

    # initialize values to save
    # local values
    c_final = np.zeros((len(load_nodes), iters*GC_time))
    d_final = np.zeros((len(load_nodes), iters*GC_time))
    ev_c_final = np.zeros((len(load_nodes), iters*GC_time))
    u_net_final = np.zeros((len(load_nodes), iters * GC_time))
    # global values
    d_fair_max_final = np.zeros((N, iters * GC_time))
    d_fair_min_final = np.zeros((N, iters * GC_time))
    Pinj_max_final = np.zeros((N, iters * GC_time))
    Pinj_min_final = np.zeros((N, iters * GC_time))
    v_at_max_final = np.zeros((N, iters * GC_time))
    v_at_min_final = np.zeros((N, iters * GC_time))
    bounds_opt_max_flags = np.zeros(iters)
    bounds_opt_min_flags = np.zeros(iters)
    LC_opt_flags = np.zeros((len(load_nodes), iters))

    for i in range(iters):
        # Run optimization
        print('running day', i)
        # gather data for current run
        netDemand = netDemandFull[:, (startIdx + GC_time * i):(startIdx + GC_time * i + T)]
        rDemand = rDemandFull[:, (startIdx + GC_time * i):(startIdx + GC_time * i + T)]
        # print(pricesFull.shape)
        prices = pricesFull[:, (startIdx + GC_time * i):(startIdx + GC_time * i + T)]

        # print('prices', prices)
        # print(prices.shape)

        # generate fair target bounds
        d_fair_max = np.zeros(netDemand.shape)
        d_fair_min = np.zeros(netDemand.shape)
        d_fair_max[not_load_bool, :] = netDemand[not_load_bool, :]
        d_fair_min[not_load_bool, :] = netDemand[not_load_bool, :]
        load_node_idx = 0

        for node in load_nodes:
            # print('running node', node)
            t_arrival = start_dict[load_node_idx][i]
            t_depart = end_dict[load_node_idx][i]
            load_node_idx += 1
            # print(t_arrival)
            # print(t_depart)
            d_fair_min[node, :], d_fair_max[node, :] = get_fair_bounds(netDemand[node, :], std_scaling, error_std, T,
                                                                       t_arrival, t_depart, charging_power)

        # print(d_fair_max)
        # print(d_fair_min)
        # print(np.sum(d_fair_max - d_fair_min < 0, axis=1))

        d_comp_min = d_fair_min.copy()
        d_comp_max = d_fair_max.copy()
        # print(d_fair_min[:,10])
        # print(d_fair_max[:,10])
        d_fair_min, d_fair_max = limit_fair_bounds(d_fair_min, d_fair_max, load_nodes, rating=None)
        # print(d_fair_min[:, 10])
        # print(d_fair_max[:, 10])
        # print('min limits', np.sum(np.abs(d_fair_min - d_comp_min) > 1e-6))
        # print('max limits', np.sum(np.abs(d_fair_max - d_comp_max) > 1e-6))

        if bounds_flag == 1:

            Pinj_max, v_at_max, status = Bounds_optimizer(rDemand, d_fair_max, netDemand, 1, coefs_mat, intercept_mat,
                                                          load_bool, T, N, Vmax=104.5, Vmin=95.5, lambda_v=lambda_v,
                                                          lambda_Pinj=lambda_Pinj, lambda_direction=lambda_direction)
            if status != 'optimal':
                print('max bounds optimization status is', status)
                bounds_opt_max_flags[i] = 1

            Pinj_min, v_at_min, status = Bounds_optimizer(rDemand, d_fair_min, netDemand, -1, coefs_mat, intercept_mat,
                                                          load_bool, T, N, Vmax=104.5, Vmin=95.5, lambda_v=lambda_v,
                                                          lambda_Pinj=lambda_Pinj, lambda_direction=lambda_direction)
            if status != 'optimal':
                print('min bounds optimization status is', status)
                bounds_opt_min_flags[i] = 1

            d_comp_min = Pinj_min.copy()
            d_comp_max = Pinj_max.copy()
            # print(Pinj_min[:, 10])
            # print(Pinj_max[:, 10])
            Pinj_min, Pinj_max = limit_fair_bounds(Pinj_min, Pinj_max, load_nodes, rating=None)
            # print(Pinj_min[:, 10])
            # print(Pinj_max[:, 10])
            # print('min limits', np.sum(np.abs(Pinj_min - d_comp_min) > 1e-6))
            # print('max limits', np.sum(np.abs(Pinj_max - d_comp_max) > 1e-6))

        elif bounds_flag == 2:
            # implement transformer static bounds
            # Pinj_max = rating tiled into the correct shape
            # Pinj_min = -Pinj_max
            # shape N, T

            rating = np.array([45, 45, 30, 50, 30, 30, 30, 45, 30, 45, 50, 50, 25, 50, 50, 50, 50, 45, 50, 50, 30, 30, 45, 45, 50,
              30, 30, 30, 30, 50, 30, 30, 45, 100, 50, 45, 30, 45, 75, 30, 30, 30, 30, 25, 50, 50, 112.5, 45, 112.5, 30,
              45, 25, 50, 45, 50, 50, 112.5, 50, 45, 50, 45, 25, 30, 50, 30, 50, 30, 45, 45, 45, 50, 30, 45, 45, 50, 30,
              50, 50, 45, 50, 50, 30, 30, 50, 25])

            Pinj_max = np.zeros(d_fair_max.shape)
            Pinj_min = np.zeros(d_fair_max.shape)
            Pinj_max[load_nodes, :] = np.tile(rating.reshape((rating.size, 1)), (1, T))
            Pinj_min[load_nodes, :] = np.tile(-rating.reshape((rating.size, 1)), (1, T))
            v_at_max = np.zeros(d_fair_max.shape)
            v_at_min = np.zeros(d_fair_max.shape)

        else:
            Pinj_max = 1000 * np.ones(d_fair_max.shape)
            v_at_max = np.zeros(d_fair_max.shape)
            Pinj_min = -1000 * np.ones(d_fair_max.shape)
            v_at_min = np.zeros(d_fair_max.shape)
            lambda_cap = 0.1

        # set very small values to 0
        Pinj_max[np.abs(Pinj_max) < 1e-7] = 0
        Pinj_min[np.abs(Pinj_min) < 1e-7] = 0

        # print('bounds shape', Pinj_max.shape)
        # print('Maximum allowed power injection', Pinj_max)
        # print('voltages at max power injection', v_at_max)
        # print(' Minimum allowed power injection', Pinj_min)
        # print('voltages at min power injection', v_at_min)

        bounds_infeasible = np.sum(Pinj_max - Pinj_min < -1e-7, axis=1)
        if np.sum(bounds_infeasible) > 0:
            idx_inf = np.where(bounds_infeasible > 0)
            print('bounds infeasible at', idx_inf, bounds_infeasible[idx_inf])
        # print(np.sum(Pinj_max - Pinj_min > 0, axis=1))

        d_fair_max_final[:, (GC_time * i):(GC_time * (i+1))] = d_fair_max[:, 0:GC_time]
        d_fair_min_final[:, (GC_time * i):(GC_time * (i + 1))] = d_fair_min[:, 0:GC_time]
        Pinj_max_final[:, (GC_time * i):(GC_time * (i + 1))] = Pinj_max[:, 0:GC_time]
        Pinj_min_final[:, (GC_time * i):(GC_time * (i + 1))] = Pinj_min[:, 0:GC_time]
        v_at_max_final[:, (GC_time * i):(GC_time * (i + 1))] = v_at_max[:, 0:GC_time]
        v_at_min_final[:, (GC_time * i):(GC_time * (i + 1))] = v_at_min[:, 0:GC_time]

        """
        plt.figure()
        plt.plot(netDemand[load_nodes[1], 0:24])
        plt.show()
        """

        load_node_idx = 0
        for node_idx in load_nodes:
            # load_node_idx = 3 # test one node first. Will need to make a for loop over all the nodes later
            start_time = start_dict[load_node_idx][i]
            end_time = end_dict[load_node_idx][i]
            charge = charge_dict[load_node_idx][i]

            Pinj_max_single = Pinj_max[node_idx, :]
            Pinj_min_single = Pinj_min[node_idx, :]

            # print('start time', start_time)
            # print('end time', end_time)
            # print('charge (desired final soc)', charge)
            # print('charging power', charging_power)

            c, d, Q, ev_c_all, ev_c_dict, ev_q_dict, bounds_cost, status = local_optimization_solve(netDemand[node_idx, :], lambda_b, lambda_cap,
                                               T, qmin[load_node_idx], qmax[load_node_idx], q0[load_node_idx],
                                               cmax[load_node_idx], dmax[load_node_idx], y_l, y_d, y_c,
                                               prices, Pinj_min_single, Pinj_max_single,
                                               start_time, end_time, charge, charging_power)

            if status != 'optimal':
                print('Local optimization status is: ', status)
                LC_opt_flags[load_node_idx, i] = 1

            if bounds_cost > 10:
                print('bounds cost was high at node:', node_idx, bounds_cost)

            # print('battery', c - d)
            # print('Q', Q)
            # print('ev charging', ev_c_all)
            u_net = c - d + ev_c_all

            """
            if load_node_idx == 1:
                plt.figure()
                plt.plot(10*prices[:,0:GC_time].T)
                plt.plot(u_net[0:GC_time] - ev_c_all[0:GC_time])
                plt.plot(netDemand[node_idx, 0:GC_time].T + ev_c_all[0:GC_time])
                plt.plot(netDemand[node_idx, 0:GC_time].T + u_net[0:GC_time])
                plt.legend(('prices', 'u net', 'net demand', 'net all'))
                plt.show()
            """

            if len(Q) < GC_time:
                q0[load_node_idx] = 0
            else:
                q0[load_node_idx] = Q[GC_time]

            c_final[load_node_idx, (GC_time * i):(GC_time * (i+1))] = c[0:GC_time]
            d_final[load_node_idx, (GC_time * i):(GC_time * (i+1))] = d[0:GC_time]
            ev_c_final[load_node_idx, (GC_time * i):(GC_time * (i + 1))] = ev_c_all[0:GC_time]
            u_net_final[load_node_idx, (GC_time * i):(GC_time * (i + 1))] = u_net[0:GC_time]

            load_node_idx += 1

    # Save values
    save_name = 'results/boundsLocalAns_solStorEV' + str(solarPen) + str(storagePen) +str(evPen) + \
                '_bounds' + str(bounds_flag) + 'ppc_reg_pq-0.79.npz'
    np.savez(save_name, solarPen=solarPen, storagePen=storagePen, evPen=evPen, bounds_flag=bounds_flag,
             netDemandFull=netDemandFull, rDemandFull=rDemandFull, pricesFull=pricesFull,
             nodesStorage=nodesStorage, load_nodes=load_nodes, qmax=qmax, iters=iters, presampleIdx=presampleIdx,
             startIdx=startIdx, GC_time=GC_time, T=T, std_scaling=std_scaling,
             lambda_b=lambda_b, lambda_cap=lambda_cap, lambda_v=lambda_v, lambda_Pinj=lambda_Pinj, lambda_direction=lambda_direction,
             d_fair_max_final=d_fair_max_final, d_fair_min_final=d_fair_min_final, Pinj_max_final=Pinj_max_final,
             Pinj_min_final=Pinj_min_final, v_at_max_final=v_at_max_final, v_at_min_final=v_at_min_final,
             c_final=c_final, d_final=d_final, ev_c_final=ev_c_final, u_net_final=u_net_final,
             bounds_opt_min_flags=bounds_opt_min_flags, bounds_opt_max_flags=bounds_opt_max_flags, LC_opt_flags=LC_opt_flags)

    print('SAVED to file', save_name)
