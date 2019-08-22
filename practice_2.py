# Import packages.
import cvxpy as cp
import numpy as np

# Equation: https://drive.google.com/file/d/17kgVzMmcQnL0PaKWqD2YUeGN-yPwvPHc/view?usp=sharing

# N is the size of the network
# T is the number of hours in the optimization horizon
# n_b is the number of batteries

def practice_function(batt_idx, N = 123, Vpen_pos = 105, Vpen_neg = 95, lambda_v = 10, T = 48, n_b = 30):

    # not_batt_idx will contain the values between 0 and N that are not in the batt_idx array.
    not_batt_idx = []

    for i in range(N):

        if not i in batt_idx:
            not_batt_idx.append(i)

    # end of for loop

    Vtol_pos = Vpen_pos
    Vtol_neg = Vpen_neg

    dhat = np.ones((N, T))

    # s is split into the real part and imaginary part: s_real & s_imag.
    # The two parts are stacked up real part on top to form the full 's' with shape (2N, T).
    s_imag = np.ones((N, T))
    s_real = cp.Variable(shape=(N, T))
    s = cp.vstack([s_real, s_imag])

    A = np.zeros((2*N, N))
    b = 100 * np.ones((N, T))

    v = cp.Variable(shape=(N, T))

    u_pos = cp.Variable(shape=(n_b, T))
    # u_neg and u_pos are the positive and negative bounds.
    u_neg = u_pos

    cp.sum(cp.sum((cp.maximum(v - Vpen_pos, 0) + cp.maximum(Vpen_neg - v, 0))**2))

    # Constraints to the minimization problem:
    constraints = [v >= 0, v <= N, u_neg >= -100, u_pos <= -100]
    constraints.append(s_real[batt_idx, :] == dhat[batt_idx, :] + u_pos)
    constraints.append(s_real[not_batt_idx, :] == dhat[not_batt_idx, :])
    constraints.append(Vtol_neg <= A.T @ s + b)
    constraints.append(Vtol_pos >= A.T @ s + b)
    constraints.append(u_neg <= u_pos)

    objective = cp.Minimize(lambda_v*cp.sum((cp.maximum(v - Vpen_pos, 0) + cp.maximum(Vpen_neg - v, 0))**2))# - cp.norm(u_neg-u_pos, "fro"))
    problem = cp.Problem(objective, constraints)
    problem.solve()

    print(problem.status)
