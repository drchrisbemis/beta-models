# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

import jump_funcs as jf

def time_series_plots(df_bta, tkr, start_date=None, end_date=None):
    # df_bta is the dataframe where information has been stored
    # but for time, the data here would be obtained from a db call
    # tkr is a string ticker
    # start_date and end_date are datetime objects or date strings
    
    if tkr in df_bta.index.get_level_values(0):
        df_ts = df_bta.xs(tkr, level=0)
        date_index = df_ts.index

        if start_date is not None:
            start_msk = date_index >= start_date
        else:
            start_msk = np.ones(date_index.size, dtype='bool')
            
        if end_date is not None:
            end_msk = date_index <= end_date
        else:
            end_msk = np.ones(date_index.size, dtype='bool')
            
        df_ts = df_ts[start_msk & end_msk]
        
        if df_ts.shape[0] > 2:
            
            # this could have been done elsewhere, but the project had a 
            # feature of allowing for flexibility...these figures are for a 
            # pinned down version of things
            df_ts['beta_hat_lower'] = [b[0] for b in df_ts['beta_hat_ci']]
            df_ts['beta_hat_upper'] = [b[1] for b in df_ts['beta_hat_ci']]
            
            # okay, time series plots of betas
            f, ax = plt.subplots(2, sharex=True, sharey=False, figsize=(12,8))
              
            # time series of betas and their confidence intervals
            date_index = df_ts.index
            ax[0].plot(df_ts['beta_hat'], label=r'$\hat{\beta}$', color='tab:blue', alpha=0.5)
            ax[0].fill_between(date_index, df_ts['beta_hat_lower'], df_ts['beta_hat_upper'], color='tab:blue', alpha=0.15)
            ax[0].grid(True)
            ax[0].set_ylabel(r'$\hat{\beta}$', color='tab:blue')
            ax[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            ax[0].set_title(r'{0} CAPM-$\beta$ 95% Confidence Interval Time Series'.format(tkr))
            
            
            # time series of volatility measures and fit statistic
            ax[1].plot(df_ts['R2'], label=r'Fit $R^2$', color='tab:red', alpha=0.5)
            ax[1].grid(True)
            ax[1].set_ylabel(r'Fit $R^2$', color='tab:red')
            ax[1].tick_params(axis='y', colors='tab:red')
            ax[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            ax[1].set_title('Volatility and Fit Metrics \n Excluding Any Outliers in Analysis')
            
            ax_alt = ax[1].twinx()
            ax_alt.plot(np.sqrt(252)*df_ts['rvol'], label='Realized Vol', color='k', alpha=0.5)
            ax_alt.plot(np.sqrt(252)*df_ts['idio_vol'], label='Idio Vol', color='tab:purple', alpha=0.5)
            ax_alt.set_ylabel('Annualized Volatility Measures', color='k')
            ax_alt.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
            ax_alt.legend(loc='upper left')
            
            plt.show()
            
        else:
            print('No figure available based on date range')
            
            
            
    else:
        print('Ticker ({0}) not available in dataset'.format(tkr))
        
        
def fit_diagnostic_plots(df_ref, tkr, calc_date, window=180):
    # ideally, this information would not be calculated again, but would have
    # been stored.  There is a friction here
    
    calc_date = pd.to_datetime(calc_date, errors='coerce')
    date_index = df_ref.index.get_level_values(1).unique()
    if calc_date not in date_index:
        # TODO better solution would be to have date_index value before calc_date
        # solution here for expediency
        calc_date = min( date_index, key = lambda x: abs( calc_date - x ) )
    
    if tkr in df_ref.index.get_level_values(0):
        df_ts = df_ref.xs(tkr, level=0).copy()
        df_ts.reset_index(inplace=True)
        df_ts['Ticker'] = tkr
        df_ts.set_index(['Ticker', 'Date'], inplace=True)
            
        reg_dict = jf.calc_betas_single_date(df_ts, calc_date, window, beta_func=jf.calc_beta_fuller)
        reg_dict = reg_dict[tkr]
        
        # stock returns
        s = reg_dict['s']
        
        # check if we should continue
            
        
        if ( np.isnan(s) ).mean() == 1:
                print('No analysis available for {0} on {1} available'.format(tkr, calc_date.strftime('%Y-%m-%d')))
        else:
            
            # design matrix (full)
            X = reg_dict['X']
            
            # design matrix with outliers removed
            X_res = reg_dict['X_res']
            
            # market returns
            mkt = np.array([r[1] for r in X])
            mkt_res = np.array([r[1] for r in X_res])
            
            # betas 
            betas = reg_dict['betas']
            
            # sigma hat (from fit)
            sigma_hat = reg_dict['sigma_hat']
            
            # degrees of freedom
            df = reg_dict['df']
            
            # expected return under model (all observations in mkt)
            s_hat = X.dot(betas)    
            
            # studentized residuals (raw, before refitting)
            st_res_raw = reg_dict['st_res_raw']
            outlier_msk = np.abs(st_res_raw)>3
            
            # calculate mean response confidence interval with this gathered data
            s_hat_ci = jf.mean_response_ci(X_res, betas, sigma_hat, df, alpha=0.05)
            ci_tmp = pd.DataFrame(np.array([mkt_res, s_hat_ci[0], s_hat_ci[1]]).T)
            ci_tmp.sort_values(0, axis=0, inplace=True) # plotting requires sorted x; i.e., tie these output to inputs
            
            # a more interesting plot is a prediction interval
            # # No heavier lift.  This work here is proof of concept, though,
            # # so not included
        
            
            f, ax = plt.subplots(2, sharex=True, sharey=False, figsize=(12,8))
            
            ax[0].scatter(x=mkt, y=s, alpha=0.7, s = 10)
            ax[0].plot(mkt, s_hat, alpha=0.7, color='red', label=r'$s = \hat{\alpha} + \hat{\beta}\cdot m$')
            ax[0].fill_between(ci_tmp[0], ci_tmp[1], ci_tmp[2], color='tab:red', alpha=0.15, facecolors='tab:red')
            ax[0].scatter(x=mkt[outlier_msk], y=s[outlier_msk], alpha=1, s=80, facecolors='none', edgecolors='orange')  # circle outliers
            ax[0].grid(True)
            ax[0].set_ylabel('Daily Stock Returns')
            ax[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=3))
            ax[0].set_xlabel('')
            ax[0].xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=3))    
            ax[0].set_title(r'CAPM-$\beta$ Fit for {0} on {1}'.format(tkr, calc_date.strftime('%Y-%m-%d')))
            
            ax[1].scatter(x=mkt, y=st_res_raw, color='tab:purple', alpha=0.7, s = 10)
            ax[1].scatter(x=mkt[outlier_msk], y=st_res_raw[outlier_msk], alpha=1, s=80, facecolors='none', edgecolors='orange')  # circle outliers
            ax[1].grid(True)
            ax[1].set_ylabel('Studentized Residuals (Raw)')
            # ax[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=3))
            ax[1].set_xlabel('Daily Market Returns')
            ax[1].xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=3))    
            ax[1].set_title('Studentized Residuals (Raw)')
        
            plt.show()            
    else:
        print('Ticker ({0}) not available in dataset'.format(tkr))
        
        
def huber_regressor_plot_poc(hr_mdl, X, s, tkr, calc_date):
    # quick example of output from a huber model
     fit_reg = hr_mdl.best_estimator_.named_steps['hr']
     s_hat = hr_mdl.predict(X)
     f, ax = plt.subplots(1, sharex=True, sharey=False, figsize=(12,8))
            
     ax.scatter(x=X, y=s, alpha=0.7, s = 10)   
     ax.plot(X, s_hat, alpha=0.7, color='red', label=r'Huber Regressor')
     ax.scatter(x=X[fit_reg.outliers_], y=s[fit_reg.outliers_], alpha=1, s=80, facecolors='none', edgecolors='orange')  # circle outliers
     ax.grid(True)
     ax.set_ylabel('Daily Stock Returns')
     ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=3))
     ax.set_xlabel('')
     ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=3))    
     ax.set_title(r'Huber Regressor Fit for {0} on {1}'.format(tkr, calc_date.strftime('%Y-%m-%d')))
    
     plt.show()
        
            

