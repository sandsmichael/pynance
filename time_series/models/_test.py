
import pandas as pd
import numpy as np 

from arima import Arima
from timeseries import TimeSeries 

fp_stock = r"C:\Users\sands\Documents\_data\JPM.csv"
fp_factors = r"C:\Users\sands\Documents\_data\F-F_Research_Data_5_Factors_2x3.csv"
series = pd.read_csv(fp_stock)['Close'].dropna().iloc[:500]
df = pd.read_csv(fp_stock).dropna().iloc[:500]


'''  '''

# Ex. Arima Model
#Arima(fp= fp_stock).Arima().evaluate()
#Arima(series = series).evaluate_models(p_values=[0, 1, 2, 4, 6, 8, 10], d_values=range(0, 3), q_values=range(0, 3))

# Ex. Data setup
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace = True)

# Ex. interpolate missing values 
df.iloc[3] = np.nan
print(df.head())
ts = TimeSeries(df=df)
ts.interpolate_na(col_name = 'Close') # TODO not calling method poperly if at all


# Ex. Smoothing
print(df.head())
ts.smoothing(data = df, col_name ='Close')
# ts.check_stationarity()