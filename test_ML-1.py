# Learning P,Q -> V relationship

import sklearn
from sklearn import preprocessing
#from aggregate import aggregate
import numpy as np
from numpy import array
#from first_svm import svm_train

#from sampling import Resampling
#from forecaster_primitive import predict
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error


def main():

	#v2 = 1 # when training on voltage squared
	v2 = 0

	training_data = np.load('ML_power_solar.npz')
	new_data_stacked = training_data['new_data_stacked']
	voltage_classified = training_data['voltage_classified']
	voltage_array = training_data['voltage_array']

	#print new_data_stacked.shape
	#print voltage_array.shape
	#print voltage_array

	#print np.sum(voltage_array > 1.05)
	#print np.sum(voltage_array < 0.95)

	#new_data_stacked_normalized = sklearn.preprocessing.normalize(new_data_stacked)

	X = new_data_stacked.T

	#print X

	print X.shape
	print voltage_array.shape

	# fit regressions
	#svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
	#svr_lin = SVR(kernel='linear', C=100, gamma='auto')
	#svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
	#svr_lin = SVR(kernel='linear', C=1, gamma='auto')
	#y_rbf = svr_rbf.fit(X_train, y_train).predict(X_train)
	#y_lin = svr_lin.fit(X_train, y_train).predict(X_train)
	#y_poly = svr_poly.fit(X_train, y_train).predict(X_train)

	error_train = []
	error_test = []
	error_vio = []
	error_max = []
	num_vios = []

	coefs_mat = np.zeros((voltage_array.shape[0],X.shape[1]))
	intercept_mat = np.zeros((voltage_array.shape[0],1))

	test_sizes = np.arange(10)/10. + 0.09
	refine = np.arange(10)/100. + 0.9
	test_sizes = np.concatenate((test_sizes,refine))
	worst_error = []
	worst_ave = []
	ave_ave = []
	n_points = []
	print test_sizes

	n_points_h = [524, 466, 408, 351, 293, 236, 178, 120, 63, 5, 57, 51, 46, 40, 34, 28, 23, 17, 11, 5, 4, 3, 2, 1]
	#n_points_h = [408]

	#for test_size in test_sizes:
	for points in n_points_h:
		for i in range(voltage_array.shape[0]):

			if v2 == 1:
				y = voltage_array[i,:].T**2 * 100
				#print 'training on voltage squared'
			else:
				y = voltage_array[i,:].T * 100
				#print 'training on linear voltage'

			#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
			#print 'Training dataset size:', X_train.shape

			X_train = X[0:points,:]
			y_train = y[0:points]
			X_test = X[points:,:]
			y_test = y[points:]

			mask_key = y_train <= 95
			num_vio = np.sum(mask_key)
			#print 'Number of voltage violations', num_vio
			num_vios.append( num_vio )

			svr_lin = SVR(kernel='linear', C=1, gamma='auto')
			y_lin = svr_lin.fit(X_train, y_train).predict(X_train)

			error_train.append( mean_absolute_error(y_train, y_lin) )

			max_err = np.max(np.abs(y_train - y_lin))

			y_hat = svr_lin.predict(X_test)
			error_test.append( mean_absolute_error(y_test, y_hat) )

			max_err = np.max(np.hstack((np.abs(y_test - y_hat),max_err)))
			error_max.append( max_err )

			if num_vio > 0:
				y_hat = svr_lin.predict(X_train[mask_key,:])
				error_vio.append( mean_absolute_error(y_train[mask_key], y_hat) )
			else:
				error_vio.append( np.nan )

			coefs = svr_lin.coef_
			intercept = svr_lin.intercept_

			coefs_mat[i,:] = coefs
			intercept_mat[i,:] = intercept

			"""
			if i == 122:
				y_hat_122 = y_hat
				y_train_122 = y_train[mask_key]
				print y_hat_122
				print y_train[mask_key]
			"""

		"""
		print num_vios
		print '################################################################################'
		print error_train
		print '################################################################################'
		print error_test
		print '################################################################################'
		print error_vio
		print '################################################################################'
		print error_max
		"""

		print 'Worst node average error', np.max(error_test)
		print 'Worst error', np.max(error_max)

		n_points.append( X_train.shape[0] )

		worst_ave.append( np.max(error_test) )
		worst_error.append( np.max(error_max) )
		ave_ave.append( np.mean(error_test) )

	#print coefs_mat
	#print intercept_mat

	#print n_points

	#plt.figure()
	#plt.plot(y_train,y_train)
	#plt.plot(y_train,np.abs(y_lin-y_train),'x')
	#plt.show()

	print 'average node average error', np.sort(ave_ave)
	print 'number of training points', -np.sort(-np.array(n_points_h))

	print 'average voltage value', np.mean(y)

	plt.figure()
	plt.plot(n_points,worst_ave,'x')
	plt.plot(n_points,ave_ave,'x')
	plt.plot(n_points,worst_error,'x')
	plt.legend(['worst node average','average node average','worst case'])
	plt.show()

	"""
	if v2 == 1:
		np.savez('SVR_PQ-V2_'+str(n_points), new_data_stacked=new_data_stacked, voltage_array=voltage_array, coefs_mat=coefs_mat, intercept_mat=intercept_mat)
	else:
		np.savez('SVR_PQ-V_'+str(n_points), new_data_stacked=new_data_stacked, voltage_array=voltage_array, coefs_mat=coefs_mat, intercept_mat=intercept_mat)
	#print 'Results saved'
	"""
	"""
	#print error_train

	print 'training accuracy', mean_absolute_error(y_train, y_lin)

	print 'Trained'

	# Look at the results

	svrs = [svr_rbf, svr_lin, svr_poly]
	error = []

	for model in svrs:
		y_hat = model.predict(X_test)
		error.append( mean_squared_error(y_test, y_hat) )
	
	#print error

	y_hat = svr_lin.predict(X_test)
	test_acc = mean_absolute_error(y_test, y_hat)
	print 'test accuracy % error', test_acc

	coefs = svr_lin.coef_
	intercept = svr_lin.intercept_

	#print coefs
	#print intercept

	#print coefs.shape
	#print intercept.shape

	#print y_lin
	
	"""

	


if __name__ == '__main__':
	main()