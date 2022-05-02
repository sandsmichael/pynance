# evaluate an ARIMA model using a walk-forward validation
# https://www.statsmodels.org/dev/examples/notebooks/generated/tsa_arma_0.html
''' https://www.itl.nist.gov/div898/handbook/pmc/section6/pmc624.htm '''

from timeseries import TimeSeries 

import warnings
from math import sqrt
import pandas as pd
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA as _ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from math import sqrt
warnings.filterwarnings("ignore")
from matplotlib import pyplot

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from math import sqrt
warnings.filterwarnings("ignore")
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf, acf, plot_pacf, pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
import statsmodels.api as sm
from scipy.stats import probplot, moment
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

class Arima:
    ''' auto regressive integrated moving average
    
    autoregression --> self correlation; correlation of a time series to a lagged version of itself
    integrated     --> ha
    moving average --> rolling mean; series of average values based on lagged subsets of data
    
    ARIMA(p, d, q)
    :param: p (AR) --> use the autocorrelation and partial autocorrelation from the correlelogram to determine a good p value
    :param: d differencing/integration: (I)
    :param: q (MA)
    --> contained with in self.order attribute

    p_values = [0, 1, 2, 4, 6, 8, 10] # Autoregressive order ---> use the maximum statisticaly significant lag from partial auto correlation plot
    d_values = range(0, 3) # integration order
    q_values = range(0, 3) # moving average order --> use the maximum statistically significant lag from auto correlation plot
    
    adjust p,d,q and train test size to prevent LinAlgError: Schur decomposition solver error.
    
    '''
    def __init__(self, df:pd.DataFrame=None, col:str = None):
        self.data = df
        self.col = col
        self.order = (4,1,10) 
        self.train_test_size = 0.50 # % of train / test set to slice

    def model(self):
        self.data.index = pd.to_datetime(self.data.index)
        self.series = self.data[self.col]
        # self.series.index = self.series.index.to_period('M')
        
        # fit model
        model = _ARIMA(self.series, order=self.order)
        model_fit = model.fit()
        
        # summary of fit model
        print(model_fit.summary())
        
        # line plot of residuals
        residuals = pd.DataFrame(model_fit.resid)
        residuals.plot()
        pyplot.show()
        
        # density plot of residuals
        residuals.plot(kind='kde')
        pyplot.show()
        # summary stats of residuals
        print(residuals.describe())
        return self
    
    
    def evaluate(self):
        X = self.series.values
        size = int(len(X) * self.train_test_size)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            model = _ARIMA(history, order=self.order)
            model_fit = model.fit()
            self.model_fit = model_fit
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            print('predicted=%f, expected=%f' % (yhat, obs))
        rmse = sqrt(mean_squared_error(test, predictions))
        print('Test RMSE: %.3f' % rmse)
        # plot forecasts against actual outcomes
        pyplot.plot(test, linewidth=1, linestyle=':')
        pyplot.plot(predictions, color='red', linewidth=1, linestyle='-')
        # pyplot.ylim(30, 150)
        pyplot.show()


    def evaluate_for_optimization(self, X, order):
        '''use the evaluate method for direct calls to evaluate a model; use this evaluate_for_optimization()
        when passing specific variations of different orders to find the optimal pdq paramater values
        ''' 
        train_size = int(len(X) * self.train_test_size)
        train, test = X[0:train_size].reset_index(drop=True), X[train_size:].reset_index(drop=True)
        history = [x for x in train]
        # make predictions
        predictions = list()
        for t in range(len(test)):
            model = _ARIMA(history, order=order)
            model_fit = model.fit()
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(test[t])
        # calculate out of sample error
        rmse = sqrt(mean_squared_error(test, predictions))
        return rmse
    
    
    def evaluate_models(self, p_values, d_values, q_values):
        '''find optimal p, d, q values via brute force
        '''
        dataset = self.series.astype('float32')
        best_score, best_cfg = float("inf"), None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p,d,q)
                    try:
                        rmse = self.evaluate_for_optimization(dataset, order)
                        if rmse < best_score:
                            best_score, best_cfg = rmse, order
                        print('ARIMA%s RMSE=%.3f' % (order,rmse))
                    except:
                        continue
                        print('error')

        print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))


    def moving_average(a:pd.array, n:int=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n


    def plot_correlogram(self, x, lags=None, title=None):    
        lags = min(10, int(len(x)/5)) if lags is None else lags
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
        axes[0][0].plot(x) # Residuals
        axes[0][0].plot(self.moving_average(x, n=21), c='k', lw=1) # moving average of risiduals
        q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
        stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f}'
        axes[0][0].text(x=.02, y=.85, s=stats, transform=axes[0][0].transAxes)
        probplot(x, plot=axes[0][1])
        mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
        s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}'
        axes[0][1].text(x=.02, y=.75, s=s, transform=axes[0][1].transAxes)
        plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
        plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
        axes[1][0].set_xlabel('Lag')
        axes[1][1].set_xlabel('Lag')
        fig.suptitle(title, fontsize=14)
        sns.despine()
        fig.tight_layout()
        fig.subplots_adjust(top=.9)
        fig1 = plt.gcf()
        plt.show()


    # FIXME: arima_order = (p, d, q)
    def plot_model_summary(model_summary, title = None):
        plt.rc('figure', figsize=(12, 7))
        plt.text(0.01, 0.05, str(model_summary), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{str(iop)}{title}.png')


    def arima_diagnostics(resids, n_lags=40):
        '''
        Function for diagnosing the fit of an ARIMA model by investigating the residuals.
        
        Parameters
        ----------
        resids : np.array
            An array containing the residuals of a fitted model
        n_lags : int
            Number of lags for autocorrelation plot
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Created figure
        '''
        
        # create placeholder subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        r = resids
        resids = (r - np.nanmean(r)) / np.nanstd(r)
        resids_nonmissing = resids[~(np.isnan(resids))]
        
        # residuals over time
        sns.lineplot(x=np.arange(len(resids)), y=resids, ax=ax1)
        ax1.set_title('Standardized residuals')

        # distribution of residuals
        x_lim = (-1.96 * 2, 1.96 * 2)
        r_range = np.linspace(x_lim[0], x_lim[1])
        norm_pdf = scs.norm.pdf(r_range)
        
        sns.distplot(resids_nonmissing, hist=True, kde=True, 
                    norm_hist=True, ax=ax2)
        ax2.plot(r_range, norm_pdf, 'g', lw=2, label='N(0,1)')
        ax2.set_title('Distribution of standardized residuals')
        ax2.set_xlim(x_lim)
        ax2.legend()
            
        # Q-Q plot
        qq = sm.qqplot(resids_nonmissing, line='s', ax=ax3)
        ax3.set_title('Q-Q plot')

        # ACF plot
        plot_acf(resids, ax=ax4, lags=n_lags, alpha=0.05)
        ax4.set_title('ACF plot')

        # arima_diagnostics(arima.resid, 40)


    # FIXME
    def _forecast():
        auto_arima_pred = auto_arima.predict(n_periods=n_forecasts, 
                                    return_conf_int=True, 
                                    alpha=0.05)

        auto_arima_pred = [pd.DataFrame(auto_arima_pred[0], 
                                        columns=['prediction']),
                        pd.DataFrame(auto_arima_pred[1], 
                                        columns=['ci_lower', 'ci_upper'])]
        auto_arima_pred = pd.concat(auto_arima_pred, 
                                    axis=1).set_index(test.index)

        fig, ax = plt.subplots(1)

        ax = sns.lineplot(data=test, color=COLORS[0], label='Actual')

        ax.plot(arima_pred.prediction, c=COLORS[1], label='ARIMA(2,1,1)')
        ax.fill_between(arima_pred.index,
                        arima_pred.ci_lower,
                        arima_pred.ci_upper,
                        alpha=0.3, 
                        facecolor=COLORS[1])

        ax.plot(auto_arima_pred.prediction, c=COLORS[2], 
                label='ARIMA(3,1,2)')
        ax.fill_between(auto_arima_pred.index,
                        auto_arima_pred.ci_lower,
                        auto_arima_pred.ci_upper,
                        alpha=0.2, 
                        facecolor=COLORS[2])

        ax.set(title="Google's stock price  - actual vs. predicted", 
            xlabel='Date', 
            ylabel='Price ($)')
        ax.legend(loc='upper left')

        plt.tight_layout()
        #plt.savefig('images/ch3_im25.png')
        plt.show()




    # FIXME
    def univariate_time_series_optimal_model():
        
        ''' ___ @ set params ___ '''
        initial_p, initial_q = (1,4)
        pq_iterations = 5
        d = 12 # d = 0 if series is stationary; use dicky fuller test to determine --> see decomposition.py
        ''' ___________________  '''

        series_name, time_series, time_series_log, time_series_log_diff = get_data(d)

        ''' ARMA ''' # but we are using the x_log_diff time series data to fit the ARMA model --> primitive AR"I"MA
        model = tsa.ARMA(endog=time_series_log_diff, order=(initial_p, initial_q)).fit() # endogenous variable; order(p,q) ---> ARIMA order is really (p,d,q) ; p=autoregressive, q=movingaverage
        print(model.summary())
        plot_model_summary(model.summary(), title = f'ARMA_Model_Summary_{initial_p}_{initial_q}_{series_name}')
        plot_correlogram(model.resid, title=f'ARMA_Residuals_Correlogram_{series_name}')
        '''
        Find optimal ARMA lags "We iterate over various (p, q) lag combinations 
        & collect diagnostic statistics to compare the result" 
        '''
        train_size = 120
        test_results = {}
        y_true = time_series_log_diff.iloc[train_size:]
        for p in range(pq_iterations):
            for q in range(pq_iterations):
                aic, bic = [], []
                if p == 0 and q == 0:
                    continue
                print(p, q)
                convergence_error = stationarity_error = 0
                y_pred = []
                for T in range(train_size, len(time_series_log_diff)):
                    train_set = time_series_log_diff.iloc[T-train_size:T] # split data into test train to prevent overfitting when predicting
                    try:
                        model = tsa.ARMA(endog=train_set, order=(p, q)).fit() # fit model by iterating through p,q values
                    except LinAlgError:
                        convergence_error += 1
                    except ValueError:
                        stationarity_error += 1

                    forecast, _, _ = model.forecast(steps=1)
                    y_pred.append(forecast[0])
                    aic.append(model.aic)
                    bic.append(model.bic)

                result = (pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}) # collect results on this instance of the iteration
                        .replace(np.inf, np.nan)
                        .dropna())

                rmse = np.sqrt(mean_squared_error(
                    y_true=result.y_true, y_pred=result.y_pred)) # calculate prediction error

                test_results[(p, q)] = [rmse,
                                        np.mean(aic),
                                        np.mean(bic),
                                        convergence_error,
                                        stationarity_error] # aggregate results of each p,q iteration

        test_results = pd.DataFrame(test_results).T
        test_results.columns = ['RMSE', 'AIC', 'BIC', 'convergence', 'stationarity']
        test_results.index.names = ['p', 'q']
        test_results.info()
        test_results.dropna()

        print(test_results.nsmallest(5, columns=['RMSE']))
        print(test_results.nsmallest(5, columns=['BIC']))

        # '''Root mean squared error'''
        sns.heatmap(test_results.RMSE.unstack().mul(10), fmt='.2', annot=True, cmap='Blues_r')
        fig1 = plt.gcf()
        plt.show()
        fig1.savefig(f'{str(iop)}{series_name}_RMSE_heatmap.png')

        # '''Bayesian Information Criterion'''
        sns.heatmap(test_results.BIC.unstack(), fmt='.2f', annot=True, cmap='Blues_r')
        fig2 = plt.gcf()
        plt.show()
        fig2.savefig(f'{str(iop)}{series_name}_BIC_heatmap.png')

        #''' use optimized ARMA lags to refit model '''
        best_p, best_q = test_results.rank().loc[:, ['RMSE', 'BIC']].mean(1).idxmin()  # utilize best p,q values as determined by lowest RMSE,BIC
        best_arma_model = tsa.ARMA(endog=time_series_log_diff, order=(best_p, best_q)).fit()
        print(best_arma_model.summary())
        plot_model_summary(best_arma_model.summary(), title = f'ARMA_Opt_Model_Summary_Opt_{best_p}_{best_q}_{series_name}')
        plot_correlogram(best_arma_model.resid, lags=20, title=f'ARMA_Opt_Residuals_Correlogram_{best_p}_{best_q}_{series_name}')
    
