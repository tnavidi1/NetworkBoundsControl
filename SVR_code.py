# from first_svm import svm_train
import matplotlib.pyplot as plt

import numpy as np

# from sampling import Resampling
# from forecaster_primitive import predict
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

import argparse


def main(X, voltage_array, presampleIdx):
    # v2 = 1 # when training on voltage squared
    v2 = 0

    """
    training_data = np.load('/Users/chloeleblanc/Documents/Stanford/PowerNet/ML_power_solar.npz')
    new_data_stacked = training_data['new_data_stacked']
    voltage_classified = training_data['voltage_classified']
    voltage_array = training_data['voltage_array']
    """

    # print new_data_stacked.shape
    # print voltage_array.shape
    # print voltage_array

    # print np.sum(voltage_array > 1.05)
    # print np.sum(voltage_array < 0.95)

    # new_data_stacked_normalized = sklearn.preprocessing.normalize(new_data_stacked)

    # X = new_data_stacked.T

    # print X

    print('shape of input PQ X', X.shape)
    print('shape of voltages y', voltage_array.shape)

    # fit regressions
    # svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    # svr_lin = SVR(kernel='linear', C=100, gamma='auto')
    # svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
    # svr_lin = SVR(kernel='linear', C=1, gamma='auto')
    # y_rbf = svr_rbf.fit(X_train, y_train).predict(X_train)
    # y_lin = svr_lin.fit(X_train, y_train).predict(X_train)
    # y_poly = svr_poly.fit(X_train, y_train).predict(X_train)

    error_train = []
    error_test = []
    error_vio = []
    error_max = []
    num_vios = []

    coefs_mat = np.zeros((voltage_array.shape[1], X.shape[1]))
    intercept_mat = np.zeros((voltage_array.shape[1], 1))

    worst_error = []
    worst_ave = []
    ave_ave = []
    n_points = []

    # n_points_h = [524, 466, 408, 351, 293, 236, 178, 120, 63, 5, 57, 51, 46, 40, 34, 28, 23, 17, 11, 5, 4, 3, 2]
    n_points_h = [presampleIdx+1]

    for points in n_points_h:
        print('training size', points)
        for i in range(voltage_array.shape[1]):
            print('training model for node', i)

            if v2 == 1:
                y = voltage_array[:, i] ** 2 * 100
            # print 'training on voltage squared'
            else:
                y = voltage_array[:, i] * 100
            # print 'training on linear voltage'

            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            # print 'Training dataset size:', X_train.shape

            X_train = X[0:points, :]
            y_train = y[0:points]
            X_test = X[points:, :]
            y_test = y[points:]

            mask_key = y_train <= 95
            num_vio = np.sum(mask_key)
            # print 'Number of voltage violations', num_vio
            num_vios.append(num_vio)

            svr_lin = SVR(kernel='linear', C=1, gamma='auto')
            y_lin = svr_lin.fit(X_train, y_train).predict(X_train)

            error_train.append(mean_absolute_error(y_train, y_lin))

            max_err = np.max(np.abs(y_train - y_lin))

            y_hat = svr_lin.predict(X_test)
            error_test.append(mean_absolute_error(y_test, y_hat))

            max_err = np.max(np.hstack((np.abs(y_test - y_hat), max_err)))
            error_max.append(max_err)

            if num_vio > 0:
                y_hat = svr_lin.predict(X_train[mask_key, :])
                error_vio.append(mean_absolute_error(y_train[mask_key], y_hat))
            else:
                error_vio.append(np.nan)

            coefs = svr_lin.coef_
            intercept = svr_lin.intercept_

            coefs_mat[i, :] = coefs
            intercept_mat[i, :] = intercept

        n_points.append(X_train.shape[0])

        worst_ave.append(np.max(error_test))
        worst_error.append(np.max(error_max))
        ave_ave.append(np.mean(error_test))

    """
    plt.figure()
    plt.title('Support Vector Regression')
    plt.xlabel('Number of points')
    plt.ylabel('Error')
    plt.plot(n_points, worst_ave, 'x')
    plt.plot(n_points, ave_ave, 'x')
    plt.plot(n_points, worst_error, 'x')
    plt.legend(['worst node average', 'average node average', 'worst case'])
    plt.show()
    """

    return coefs_mat, intercept_mat

def test(X, voltage_array, coefs_mat, intercept_mat):
    v2 = 0

    print('testing model')

    if v2 == 1:
        v_real = voltage_array ** 2 * 100
        # print 'voltage squared'
    else:
        v_real = voltage_array * 100
        # print 'linear voltage'

    y_hat = X @ coefs_mat.T + intercept_mat.T

    test_errors = y_hat - v_real

    return test_errors


if __name__ == '__main__':
    # input parameters from user
    parser = argparse.ArgumentParser(description='Generate solar and storage demand data')
    parser.add_argument('--train', default=1, help='random seed')
    parser.add_argument('--storagePen', default=4, help='storage penetration percentage times 10')
    parser.add_argument('--solarPen', default=6, help='solar penetration percentage times 10')
    parser.add_argument('--Qcap', default=-0.79, help='power injection in ppc file name')

    FLAGS, unparsed = parser.parse_known_args()
    print('running with arguments: ({})'.format(FLAGS))

    train = int(FLAGS.train)
    storagePen = float(FLAGS.storagePen) / 10
    solarPen = float(FLAGS.solarPen) / 10
    Qcap = FLAGS.Qcap

    # specify load data file information
    fName = 'data/demand_solStor'+ str(solarPen) + str(storagePen) +'.npz'
    ppc_name = 'data/case123_ppc_reg_pq'+str(Qcap) + '.npy'
    # ppc_name = 'data/case123_ppc_none.npy'  # this is the data when voltage regulators are 0
    presampleIdx = 168 - 1 # first week as training data for forecasters/PF models/other
    startIdx = presampleIdx + 1  # starting index for the load dataset

    # load load data
    l_data = np.load(fName)
    netDemandFull = l_data['netDemandFull']
    # Define t_idx
    try:
        t_idx = l_data['t_idx']
    except:
        print('no t_idx in data')
        t_idx = netDemandFull.shape[1]
    # load rDemandFull reactive power data
    ppc = np.load(ppc_name, allow_pickle=True)[()]
    rDemandFull = ppc['bus'][:, 3]
    if rDemandFull.size < 2*netDemandFull.shape[0]:
        rDemandFull = 1000 * np.tile(rDemandFull.reshape((rDemandFull.size, 1)), (1,t_idx))
    pq_stacked = np.vstack((netDemandFull[:, 0:t_idx], rDemandFull))

    # load voltage data
    results_name = 'results/results_solStor' + str(l_data['solarPen']) + str(
        l_data['storagePen']) + 'ppc_reg_pq-0.79.npz'
    #results_name = 'results/results_solStor' + str(l_data['solarPen']) + str(
    #    l_data['storagePen']) + 'ppc_reg_pv.npz'
    v_data = np.load(results_name)
    allVoltage = v_data['allVoltage']

    if t_idx > allVoltage.shape[1]:
        print('all voltage is smaller than net demand full')

    t_idx = np.min((allVoltage.shape[1], t_idx))

    # define X and voltage array here
    voltage_array = allVoltage[:, 0:t_idx].T
    # print(voltage_array.shape)
    X = pq_stacked[:, 0:t_idx].T
    # print(X.shape)

    if train == 1:
        print('if')
        coefs_mat, intercept_mat = main(X, voltage_array, presampleIdx)
        save_name = 'data/SVR_coeffs_solStor'+ str(l_data['solarPen']) + str(l_data['storagePen']) + 'ppc_reg_pq-0.79.npz'
        np.savez(save_name,
                 coefs_mat=coefs_mat,
                 intercept_mat=intercept_mat,
                 presampleIdx=presampleIdx)

        print('saved to', save_name)

    else:
        print('else')
        load_name = 'data/SVR_coeffs_solStor'+ str(0.0) + str(0.0) + 'ppc_reg_pq-0.79.npz'
        model_data = np.load(load_name)
        coefs_mat = model_data['coefs_mat']
        intercept_mat = model_data['intercept_mat']
        print('shape of coefs', coefs_mat.shape)
        print(intercept_mat.shape)
        test_errors = test(X, voltage_array, coefs_mat, intercept_mat)
        print('average absolute error', np.mean(np.abs(test_errors)))
        print('max error', np.max(test_errors))
        print('min error', np.min(test_errors))

        # make histogram of test errors (try matplotlib.pyplot.hist)
        n, bins, patches = plt.hist(x=test_errors, bins='auto') #, color='#0504aa'),
                                  #  alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.title('Test Errors')
        #plt.ylim(ymax=np.ceil(90))
        # plt.show()

        # Save plot as image
        # Have filename include solar penetration value and ppc_reg_pq-0.79
        plt.savefig('results/test_errors_' + str(solarPen) + 'ppc_reg_pq' + str(Qcap) + '.png')
