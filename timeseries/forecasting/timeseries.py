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


    def __init__(self, fp:str=None, col:str = None, series:str = None) -> None:
        self.fp = fp
        self.series = series
        self.data = self.read_data()


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
        else:
            sys.exit()
            print('No File Input Data Supplied')
        return data


