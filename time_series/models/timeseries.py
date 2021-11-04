import pandas as pd 
import numpy as np
import sys,os
from pandas import datetime
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot
import pandas as pd 
import numpy as np 


class TimeSeries():
    '''
    Pass a filepath <or> column <or> series to this class to access many predictive or analytical methods that can be performed on your input data
    The functionality for these methods is defined in the inner directories of the timeseries folder
    '''


    def __init__(self, fp:str=None, col:str = None, series:pd.Series = None, df:pd.DataFrame = None) -> None:
        self.fp = fp
        self.series = series
        self.df = df

        self.data = self.read_data()
    
    def __str__(self):
        print('TimeSeries Object')


    def read_data( self, skiprows=0) -> pd.DataFrame:
        '''
        detect raw data from filepath or series as supplied in constructor
        '''
        if type(self.fp) == str:
            if '.csv' in self.fp:
                data = pd.read_csv(self.fp, skiprows=skiprows)
            elif '.xls' in self.fp: # catch xlsx, xlsm, xls
                data = pd.read_excel(self.fp, skiprows=skiprows)
        elif isinstance(self.series, pd.Series):
            data = self.series # returns pd.series
        elif isinstance(self.df, pd.DataFrame):
            data = self.df # returns pd.series
        else:
            sys.exit()
            print('No File Input Data Supplied')
        return data


    ''' time series utility functions '''

    def interpolate_na(self, col_name=None, ix_name = None, method='time'):
        try:
            self.data[col_name] = self.data[col_name].interpolate(method, axis = 0) #linear or time
        except ValueError:
            print('except')
            self.data[ix_name] = pd.to_datetime(self.data[ix_name])
            self.data.set_index(ix_name, inplace = True)
            self.data[col_name] = self.data[col_name].interpolate(method, axis = 0) #linear or time
        print(self.data)
        return self

    def smoothing(self, data, col_name=None,):
        # data[col_name] = data.groupby([data.index.date]).apply(lambda x: pd.ewm(x, alpha=0.84))
        #kalman

        # seasonal
        from statsmodels.tsa.seasonal import seasonal_decompose
        # data = data.sort_index(inplace=True)
        result = seasonal_decompose(data[col_name], model='additive')
        print(result.trend)
        print(result.seasonal)
        print(result.resid)
        print(result.observed)



    def check_stationarity(data=None, col_name=None,):
        #https://machinelearningmastery.com/time-series-data-stationary-python/
        X = data[col_name].values
        split = round(len(X) / 2)
        X1, X2 = X[0:split], X[split:]
        mean1, mean2 = X1.mean(), X2.mean()
        var1, var2 = X1.var(), X2.var()
        print('mean1=%f, mean2=%f' % (mean1, mean2))
        print('variance1=%f, variance2=%f' % (var1, var2))
        data[col_name].hist()
        plt.show()

        X = np.log(X) #log transform
        plt.hist(X)
        plt.show()


    def self_correlation(data=None, col_name=None,):
        from statsmodels.graphics import tsaplots

        # Display the autocorrelation plot of your time series
        fig = tsaplots.plot_acf(data[col_name], lags=24)
        plt.show()

        # spurious & lagging correlation??
        # cointegration
