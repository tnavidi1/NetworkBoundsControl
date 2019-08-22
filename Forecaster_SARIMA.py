import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX


class Forecaster:
    def __init__(self, my_order=(3, 0, 3), my_seasonal_order=(3, 0, 3, 24), pos=False, training_mean=None):
        self.pos = pos  # boolean indicating whether or not the forecast should always be positive
        self.my_order = my_order
        self.my_seasonal_order = my_seasonal_order
        self.model_params_p = np.nan
        self.model_params_r = np.nan
        self.model_params_s = np.nan
        self.training_mean = training_mean

    def input_training_mean(self, training_mean):
        self.training_mean = training_mean
        return True

    def scenarioGen(self, pForecast, scens, battnodes):
        """
        Inputs: battnodes - nodes with storage
            pForecast - real power forecast for only storage nodes
            pMeans/Covs - dictionaries of real power mean vector and covariance matrices
                            keys are ''b'+node#' values are arrays
            scens - number of scenarios to generate
        Outputs: sScenarios - dictionary with keys scens and vals (nS X time)
        """

        nS, T = pForecast.shape
        sScenarios = {}
        for j in range(scens):
            counter = 0
            tmpArray = np.zeros((nS, T))
            if nS == 1:
                sScenarios[j] = pForecast  # no noise
            else:
                for i in battnodes:
                    # resi = np.random.multivariate_normal(self.pMeans['b'+str(i+1)],self.pCovs['b'+str(i+1)])
                    # tmpArray[counter,:] = pForecast[counter,:] + resi[0:T]
                    tmpArray[counter, :] = pForecast[counter, :]  # no noise
                    counter += 1
                sScenarios[j] = tmpArray

        return sScenarios

    def netPredict(self, prev_p, time):
        # just use random noise function predict
        pForecast = self.predict(prev_p, time, model_name='p')
        return pForecast

    def rPredict(self, prev_data_r, time):
        # just use random noise function predict
        rForecast = self.predict(prev_data_r, time, model_name='r')
        return rForecast

    def train(self, data, model_name='p'):
        model = SARIMAX(data, order=self.my_order, seasonal_order=self.my_seasonal_order, enforce_stationarity=False,
                        enforce_invertibility=False)
        # What do the model_names correspond to?
        if model_name == 'r':
            model_fitted_r = model.fit(disp=False)  # maxiter=50 as argument
            self.model_params_r = model_fitted_r.params
        elif model_name == 's':
            model_fitted_s = model.fit(disp=False)  # maxiter=50 as argument
            self.model_params_s = model_fitted_s.params
        else:
            model_fitted_p = model.fit(disp=False)  # maxiter=50 as argument
            self.model_params_p = model_fitted_p.params

    def saveModels(self, fname):
        np.savez(fname, model_fitted_p=self.model_fitted_p, model_fitted_r=self.model_fitted_r,
                 model_fitted_s=self.model_fitted_s)

    def loadModels(self, model_params_p=None, model_params_r=None, model_params_s=None):
        """
        self.model = SARIMAX(data, order=self.my_order, seasonal_order=self.my_seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        self.model_fit = self.model.filter(model_fitted.params)
        """

        if model_params_p is not None:
            self.model_params_p = model_params_p
        if model_params_r is not None:
            self.model_params_r = model_params_r
        if model_params_s is not None:
            self.model_params_s = model_params_s

    def predict(self, prev_data, period, model_name='p'):

        # stime = time.time()

        model = SARIMAX(prev_data, order=self.my_order, seasonal_order=self.my_seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        if model_name == 'r':
            model_fit = model.filter(self.model_params_r)
        elif model_name == 's':
            model_fit = model.filter(self.model_params_s)
        else:
            model_fit = model.filter(self.model_params_p)

        yhat = model_fit.forecast(period)

        if self.pos:
            yhat = yhat.clip(min=0)  # do not allow it to predict negative values for demand or solar

        # print 'pred time', time.time()-stime

        return yhat


def trainForecaster(power, n_samples, fname):
    # train models

    power = power.reshape((power.size, 1))

    # for training
    training_data = power[0:n_samples]

    #print('training data', training_data)

    training_mean = np.mean(training_data)
    print('mean', training_mean)

    forecaster = Forecaster(my_order=(3, 0, 3), my_seasonal_order=(3, 0, 3, 24), pos=False)
    forecaster.train(training_data, model_name='p')
    forecaster_params = forecaster.model_params_p

    np.savez('forecast_models/' + fname + str(n_samples) + '.npz', forecaster_params=forecaster_params,
             training_mean=training_mean)
    print('SAVED forecaster params at', 'forecast_models/' + fname + str(n_samples) + '.npz')

    return forecaster, training_mean


def testForecaster(power, n_samples):
    # test
    # for loading the forecaster from a saved file
    model_data = np.load('SARIMA_model_params' + str(n_samples) + '.npz')
    forecaster_params = model_data['forecaster_params']
    training_mean = model_data['training_mean']

    forecaster = Forecaster(my_order=(3, 0, 3), my_seasonal_order=(3, 0, 3, 24), pos=False)
    model_fitted_p = forecaster_params
    forecaster.loadModels(model_params_p=model_fitted_p)

    # begin test
    yhats = []
    offset = 4 * 3 * 24
    iters = 4 * 24 * 3
    period = 4 * 24
    step = 1  # test error of only 1 step ahead prediction
    for i in range(iters):
        prev_p = power[n_samples - offset + step * i:n_samples + step * i]
        prev_p = power.reshape((prev_p.size, 1))
        yhat = forecaster.predict(prev_p, period, model_name='p')

        # yhat += -np.min(yhat)
        yhat += training_mean

        yhats.append(yhat[0:step])  # only save next step

    yhat = np.concatenate(yhats)
    y_real = power[n_samples:n_lsamples + step * (i + 1)]

    print('MAE', np.mean(np.abs(y_real - yhat)))
    print('mean', np.mean(y_real))

    """
    plt.figure()
    plt.plot(yhat)
    plt.plot(y_real)
    plt.show()
    """

    return forecaster

def load_or_train_forecaster(n_samples):
    # !!! This function is incomplete !!!

    try:
        # load SARIMA forecaster
        print('Loading forecaster parameters from', 'forecast_models/SARIMA_model_params' + str(n_samples) + '.npz')
        model_data = np.load('forecast_models/SARIMA_model_params' + str(n_samples) + '.npz')
        forecaster_params = model_data['forecaster_params']
        training_mean = model_data['training_mean']
        forecaster = Forecaster(my_order=(3, 0, 3), my_seasonal_order=(3, 0, 3, 24), pos=False)
        model_fitted_p = forecaster_params
        forecaster.loadModels(model_params_p=model_fitted_p)
        forecaster.input_training_mean(training_mean)
    except:
        # train SARIMA forecaster
        print('no forecaster data found, so training new forecaster')
        forecaster, training_mean = trainForecaster(power, n_samples, 'SARIMA_model_params')
        forecaster.input_training_mean(training_mean)

    return forecaster, training_mean


def usage_example(forecaster, power, period, start_idx, offset):
    # !!! This function is just an example and is incomplete !!!
    # !!! This function is no longer used !!!

    i = 0

    # get the previous real data from array called power
    # amount of real data determined by offset, start_idx is constant and t_horizon is meaningless in this case
    prev_p = power[:, start_idx - offset: start_idx]
    prev_p = prev_p.reshape((prev_p.size, 1))

    # enter the previous data and the number of steps into the future to predict and get the forecast
    # p_curr is the forecast
    # period is the number of steps into the future to predict
    p_curr = forecaster.predict(prev_p, period, model_name='p')
    p_curr = p_curr.reshape((1, p_curr.size)) + forecaster.training_mean

    return p_curr


def get_forecast_error_stats(power, forecaster, period, start_idx, offset):
    # Do not forget to fill the function input with the required inputs
    # This is the function you will implement

    # take input data power and make predictions in chunks of 72
    # split data into windows that have size 72 and each window is separated by 24 points
    # this means that there will be 48 points overlap between two windows
    # number of windows = ((total size of data - start_idx) / 24) - 2
    num_windows = int(((power.size - start_idx) / 24) - 2)


    # initialize error array, which has shape (number of windows, 72)
    error = np.zeros((num_windows, period))

    # put for loop for repeating prediction every 24 time steps
    for i in range(num_windows):
        window_power = power[:, start_idx + i*24:start_idx + i*24 + period]

        prev_p = power[:, start_idx + i*24 - offset: start_idx + i*24]
        prev_p = prev_p.reshape((prev_p.size, 1))

        p_forecast = forecaster.predict(prev_p, period, model_name='p')
        p_forecast = p_forecast.reshape((1, p_forecast.size)) + forecaster.training_mean

        # compare forecast to true data
        error[i, :] = window_power - p_forecast

    # end for loop
    error_mean = np.mean(error, axis=0)
    error_std = np.std(error, axis=0)

    print('std shape', error_std.shape)

    return error_mean, error_std


def get_EV_charging_window(t_arrival, t_depart, capacities, charge_power):
    # this is another function you should implement for a separate purpose
    # this function is unrelated to the forecasting

    # t_arrival is a 1D array of arrival times of many EVs
    # same for t_depart, but it is the departure times

    # initialize array to have size 48
    total_charge = np.zeros(48)

    # this function needs to add up one array for each element in the input arrays
    for i in range(len(t_arrival)):

        # create array with number of elements = t_depart - t_arrival
        charge = np.zeros(t_arrival[i] - t_depart[i])
        # We are considering the worst case scenario, where all the cars are being charged at max power for the full
        # duration. So all values in the array are set to charge_power
        charge[:] = charge_power
        # Add new array to total_charge starting at index t_arrival and ending at index_t_depart
        total_charge[t_depart[i] : t_arrival[i]] += charge

    # end for loop

    return total_charge


def get_op_bounds(prev_p, period, std_scaling, error_std, t_arrival, t_depart, charge_power, capacities):
    # to be implemented after the previous two functions
    # Function should input the previous power (prev_p) same as in get_forecast_error_stats
    # use that to calculate the forecast using :
    """
    prev_p = prev_p.reshape((prev_p.size, 1))

    # make sure period is now input as 48 instead of the previous 72
    p_forecast = forecaster.predict(prev_p, period, model_name='p')
    p_forecast = p_forecast.reshape((1, p_forecast.size)) + p_forecast.training_mean
    """

    prev_p = prev_p.reshape((prev_p.size, 1))
    p_forecast = forecaster.predict(prev_p, period, model_name='p')
    p_forecast = p_forecast.reshape((1, p_forecast.size)) + forecaster.training_mean

    # Input error_std and multiply by scaling constant (std_scaling)
    scaled_error_std = error_std * std_scaling

    # Then add the last 2 days (48 points) of the scaled std to the forecast
    new_p_forecast = p_forecast + scaled_error_std[-48:]

    # then calculate EV total charge from function get_EV_charging_window and add that to the new forecast
    # this new value is op_max
    total_charge = get_EV_charging_window(t_arrival, t_depart, capacities, charge_power)
    op_max = new_p_forecast + total_charge

    # Now subtract the scaled std from the original forecast to get op_min
    op_min = p_forecast - scaled_error_std[-48:]

    return op_max, op_min


if __name__ == '__main__':
    # run example training

    power_data = np.loadtxt('synthFarmData_15minJuly.csv')
    power_data = power_data.reshape((1, len(power_data)))

    n_samples = 480  # how many samples to use for training

    start_idx = 4 * 24 * 4  # when to start the prediction
    offset = 4 * 24 * 3  # how many past time steps needed for prediction

    t_horizon = 1

    period = 72  # how many time steps into the future to predict

    forecaster, training_mean = load_or_train_forecaster(n_samples)

    error_mean, error_std = get_forecast_error_stats(power_data, forecaster, period, start_idx, offset)
    # the axis=1 arguments means to do the mean and std for each column
    print('Mean error: ', error_mean)
    print('Error standard deviation: ', error_std)

    np.savez('forecaster_stats_data', error_mean=error_mean, error_std=error_std)

    # saved the data from before and now loading it here to avoid recalculating every time
    print('loading forecaster stats from', 'forecaster_stats_data.npz')
    f_data = np.load('forecaster_stats_data.npz')
    error_std = f_data['error_std']

    # implement get_op_bounds here
    # first take the previous power (prev_p) same as in get_forecast_error_stats as input to the function
    # period = 48 : we would only like to predict 2 days and use the last two days in the 3 day forecast
    std_scaling = 3

    # Arrival and departure times
    t_arrival = [18, 16, 20, 19]
    t_depart = [8, 6, 9, 14]
    charge_power = 12
    capacities = []
    op_max, op_min = get_op_bounds(power_data, 48, std_scaling, error_std, t_arrival, t_depart, charge_power, capacities)



