# Import packages.
import cvxpy as cp
import numpy as np

# Change this into function format with inputs (

##### New EV additions here
# introduce many new variables to represent ev charging
# it is possible to use a for loop and a dictionary to make many new variables
# ex. for i in range: ev_c_dict[i] = cp.Variable
# Each variable should consist of a charge profile for each EV that only exists between start_time to end_time
# (so has length end-start)
# add constraints where each variable in ev_c_dict is between 0 and ev_cmax = charging_power
# start_time and end_time are arrays that contain the time for each EV (one value in the arrays for each EV)
# therefore the number of EV variables = len(start_time)

# introduce many new variables ev_q_dict to represent EV SOC for each car
# you can add constraints in a for loop like: for i in range: constraints.append( constraint )
# each variable in ev_q_dict should be equal to the previous SOC + ev_c very similarly to the current storage model Q
# add constraints ev_q_dict[i] = ev_q0 = 0
# you can assume the efficiency values are 1 like with the storage
# add constraint ev_q_dict[i][end time] = charge for each car in ev charge

# make a value called ev_c_all with shape (1, T) = combined ev_c sum where the start/end indexes are respected
# something like for i in range: ev_c_all[start_time[i]:end_time[i]] += ev_c_dict[i]

# modify constraints P + c - d <= u_max and u_min to be P + c - d + ev_c_all
# also modify objective to include P + c - d + ev_c_all

# add this term to objective:
# lambda_bounds*cp.sum((cp.pos(P + c - d + ev_c_all - u_max) + cp.pos(u_min - P + c - d + ev_c_all))**2)


""" move all of these constants outside of the function and input them into the function

lambda_b = 0.001
T = 48 # T is size of inputs

Qmin = 0
Qmax = 14
Qo = 1
cmax = 4
dmax = cmax

# y are efficiency
y_l = 1
y_d = 1
y_c = 1

# Lambda e is price electricity
lambda_e = np.hstack((.202*np.ones((1,12)), .463*np.ones((1,6)), .202*np.ones((1,6))))
lambda_e = np.tile(lambda_e,2)

"""

# Q is battery charge level. First entry must be placed to a constant input
Q = cp.Variable(T+1)
# c is charging variable
c = cp.Variable(T)
# d is discharging variable
d = cp.Variable(T)

soc_constraints = [
        0 <= c,
        c <= np.tile(cmax,T),
        0 <= d,
        d <= np.tile(dmax,T),
        Qmin <= Q,
        Q <= Qmax,
        # np.tile(u_min,T) <= P+c-d, P+c-d <= np.tile(u_max,T),  # moved to soft constraint
        Q[0] == Qo,
        Q[1:T+1] == y_l*Q[0:T]+y_c*c[0:T]-y_d*d[0:T]
]


prob = cp.Problem(cp.Minimize(
        lambda_e.reshape((1, lambda_e.size)) @ cp.reshape(cp.pos(P + c - d), (T, 1)) +
        # lambda_d*cp.max(cp.hstack([P+c-d-Pomax, np.zeros(1)])) +  # no demand charge for this application
        lambda_b*cp.sum(c+d)),
        soc_constraints)

prob.solve(solver=cp.ECOS)

if prob.status != 'optimal':
        print('Optimization status is: ', prob.status)

print(c.value)
print(d.value)
print(Q.value)