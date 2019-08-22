import numpy as np
import time
from pypower.api import runpf, ppoption
import argparse


def Violation_Process(allVoltage, Vmin, Vmax):
	vGreater = (allVoltage-Vmax).clip(min=0)
	vLess = (Vmin-allVoltage).clip(min=0)
	vio_plus_sum = np.sum(vGreater, axis=1) # bus# X sum of all over voltage violations
	vio_min_sum = np.sum(vLess, axis=1) # bus# X sum of all under voltage violations

	vio_plus_max = np.max(vGreater)
	vio_min_max = np.max(vLess)

	vio_timesbig = (vGreater + vLess) > 0
	vio_times = np.sum(vio_timesbig, axis=1) # bus# X number of times there are violations

	print( 'Maximum over voltage violation: ', vio_plus_max )
	print( 'Maximium under voltage violation: ', vio_min_max )
	vioTotal = np.sum(vio_min_sum+vio_plus_sum)
	print( 'Sum of all voltage violations magnitude: ', vioTotal )
	viosNum = sum(vio_times)
	print( 'Number of voltage violations: ', viosNum )
	vioAve = vioTotal/viosNum
	print( 'Average voltage violation magnitude: ', vioAve )

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

	ppc['bus'][:,2] = pDemand
	ppc['bus'][:,3] = rDemand
	#ppc['bus'][rootIdx,7] = rootVoltage # Doesnt actually set PF root voltage
	
	# for surpressing runpf output
	ppopt = ppoption(VERBOSE = 0, OUT_ALL = 0)
	ppc_out = runpf(ppc, ppopt)

	runVoltage = ppc_out[0]['bus'][:,7]

	return runVoltage


def opt_avoid_voltage(coefs_mat, intercept_mat, pq):

	V2upBound = 105
	V2lowBound = 95
	V_weight = 100

	n, T = pq.shape
	n = n/2

	intercept_mat = np.tile(intercept_mat, (1,T))
	c_ahead = np.matrix(coefs_mat)*np.matrix(pq) + np.matrix(intercept_mat)

	p = pq[0:n,:]
	q = pq[n:,:]

	pad = np.zeros(q.shape)

	print 'problem size', pq.shape

	U = cvx.Variable(2*n,T)
	Vn = cvx.Variable(n,T)


	obj = cvx.Minimize( cvx.norm(U, 'fro')
			 		+ V_weight*cvx.sum_entries(cvx.square(cvx.pos(Vn - V2upBound)) + cvx.square(cvx.pos(V2lowBound - Vn))) )

	constraints = [ Vn == coefs_mat*U + c_ahead ]

	prob = cvx.Problem(obj, constraints)

	#print kdkdkd

	prob.solve(solver = cvx.ECOS)

	return prob.status, prob.value, Vn.value, U.value


def main(vFlag=0):

	svr_data = np.load('SVR_PQ-V.npz')
	new_data_stacked=svr_data['new_data_stacked']
	voltage_array=svr_data['voltage_array']
	coefs_mat=svr_data['coefs_mat']
	intercept_mat=svr_data['intercept_mat']

	# load network data
	fName = 'PF_orig_sol0.6.npz'
	allData = np.load(fName)
	try:
		ppc=allData['ppc'][()]
		#ppc['gen'][0,5] = 1.022
		#ppc['bus'][0,7] = 1.022
	except:
		print( 'no PPC' )

	#print new_data_stacked.shape
	#print voltage_array.shape

	prices = np.reshape(np.hstack((.25*np.ones((1,16)) , .35*np.ones((1,5)), .25*np.ones((1,3)))), (1, 24))

	sellFactor = 0

	n, T = new_data_stacked.shape
	nB = n/2

	# need to add all storage node info
	# umin umax qmin qmax q0

	allVoltage = np.zeros((nB,T))
	netDemandFull2 = new_data_stacked[0:nB,:]
	rDemandFull = new_data_stacked[nB:,:]

	pDemand = netDemandFull2

	if vFlag == 0:

		print 'removing voltage violations using SVR learned linear model'
		print 'total problem size', new_data_stacked.shape

		Vn = np.zeros((nB,T))
		U = np.zeros((n,T))

		interval = 72

		for i in range(int(T/interval)):
			print 'runnning day_ of _', i, int(T/interval)
			st = time.time()
			status, opt, Vn_p, U_p = opt_avoid_voltage(coefs_mat, intercept_mat, new_data_stacked[:,interval*i:interval*(i+1)])
			if status != 'optimal':
				print 'Opt status is:', status
			
			print 'comp time', time.time() - st

			Vn[:,interval*i:interval*(i+1)] = Vn_p
			U[:,interval*i:interval*(i+1)] = U_p

		print Vn
		print np.sum(Vn < 95)
		print np.sum(Vn > 105)

		np.savez('U_ML_opt-v0_b5', U=U, Vn=Vn)
		print 'Saved optimization results'

	else:

		print pDemand.shape
		print rDemandFull.shape

		for i in range(T):
			allVoltage[:,i] = PF_Sim(ppc, pDemand[:,i], rDemandFull[:,i])

		vio_times, vio_plus_sum, vio_min_sum, vio_when = Violation_Process(allVoltage, 0.95, 1.05)

		vio_total_square = np.sum(np.square(vio_min_sum + vio_plus_sum))

		print( 'before storage voltage violations' )
		print( 'vio total square', vio_total_square )

		controlData = np.load('U_ML_opt-v0_b5.npz')
		U = controlData['U']
		Vn = controlData['Vn']

		pDemand += U[0:nB,:]
		rDemandFull += U[nB:,:]

		for i in range(T):
			allVoltage[:,i] = PF_Sim(ppc, pDemand[:,i], rDemandFull[:,i])

		vio_times, vio_plus_sum, vio_min_sum, vio_when = Violation_Process(allVoltage, 0.95, 1.05)

		vio_total_square = np.sum(np.square(vio_min_sum + vio_plus_sum))

		print( '-- Desired Traits --' )

		print( 'vio total square', vio_total_square )





if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Simulate Control')
	parser.add_argument('--v', default=0, help='do not simulate voltages')
	FLAGS, unparsed = parser.parse_known_args()
	print 'running with arguments: ({})'.format(FLAGS)
	v = float(FLAGS.v)

	if v == 0:
		print 'importing cvxpy'
		import cvxpy as cvx
	
	main(vFlag=v)

