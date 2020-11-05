# -*- coding: utf-8 -*-
import datetime as dt
import numpy as np
import pandas as pd
import sys

# # # # also, since the spec said 'numpy' 
# # # # I've stuck to numpy and built most math using this package
# # # # but I need a t stat in some spots, so I pull in scipy with minimal touch
from scipy.stats import t

def assign_market_segment(df_qd):
    # assign a market segment based on percentiles and the cross-section
    # of market caps
    # assumes some bespoke knowledge of df_qd ... namely, has market_cap column
    # Market cap assignment is conditional on the cross-section input
    
    # df_segs returns market cap segments in dataframe if possible
    if 'market_cap' not in df_qd.columns:
        print('No market cap information available')
        df_segs = pd.DataFrame()
    else:
        print('Assigning market cap segments')
        df_segs = df_qd.copy()
        df_segs['Market Segment'] = np.nan
        
        ptiles = [0.0, 0.25, 0.50, 0.75, 0.95, 1.0]
        ptiles = df_segs['market_cap'].quantile(q=ptiles).values
        ptiles[0] = 0 # make farthest left endpoint less than smallest value observed
        ptiles[-1] = np.inf # make farthest right endpoint greater than smallest value observed
        ptiles = list(zip(ptiles[:-1], ptiles[1:]))
        market_segment = ['Micro Cap', 'Small Cap', 'Mid Cap', 'Large Cap', 'Mega Cap']
        
        ptile_dict = dict(zip(market_segment,ptiles))
        
        for market_segment in ptile_dict:
            p = ptile_dict[market_segment]
            msk = ( df_segs['market_cap']>p[0] ) & ( df_segs['market_cap']<=p[1] )
            df_segs.loc[msk, 'Market Segment'] = market_segment
        
    return df_segs

def select_ticker_by_mkt_cap_seg(df_qd):
    # subselect some strata from Large Cap, Mid Cap, and Small Cap
    # takes in a cross section of market caps
    # assigns market cap segments based on percentiles
    # tries to get a hundred each from Largest, Mid, and Small Cap
    
    df_segs = assign_market_segment(df_qd)
    
    df_segs.sort_values('market_cap', ascending=False, inplace=True)
    segment_list = ['Mega Cap', 'Mid Cap', 'Small Cap']
    ticker_list = list()
    for segment in segment_list:
        ticker_list = ticker_list +\
                        list(df_segs[df_segs['Market Segment'] == segment].index[:100])
                        
    return ticker_list

def calc_returns(df_prc):
    # calculate returns from pricing dataframe, assuming
    # multiindex with level 0 being identifier (ticker), and
    # level 1 being date
    # no gaps in date assumed; i.e. NaNs if no trade, not skipped date
    
    if df_prc.index.get_level_values(0).unique().shape[0] > 1:
        df_prc = df_prc.assign(Returns = df_prc.groupby(level=0).\
                                  apply(lambda x: x['Adj Close'] / x['Adj Close'].shift(1) - 1 ).\
                                       droplevel(level=0, axis=0 ))
    else:
        df_prc = df_prc.assign(Returns = df_prc.groupby(level=0).\
                                  apply(lambda x: x['Adj Close'] / x['Adj Close'].shift(1) - 1 ).T)
            
    return df_prc

def calc_beta_basic(s, mkt, backfill_candidates=None):
    # simplest version of a function to calculate linear beta in 
    # s = alpha + beta*mkt + epsilon
    # using least squares 
    # s and mkt both arrays
    # # this is a very naive implementation
    
    assert len(s) == len(mkt)
    
    if len(s) > 60: #TODO this is arbitrary; could make an estimate based on c.i. prior
        # build a design matrix, and include an intercept
        X = np.vstack([np.ones(len(mkt)), mkt]).T
                     
        beta = np.linalg.lstsq(X, s, rcond=None)[0][1]
    else:
        beta = np.nan
    
    # one may verify that this is the OLS beta via
#    stock_mkt_cov = np.cov(s, mkt)
#    beta = stock_mkt_cov[0][1] / stock_mkt_cov[1][1]
    # # aside...I tried to make something faster than np.linalg.lstsq here 
    # # but this is in fact slightly slower by about 0.6 ms
    # # Similarly, normal equations solution 
    # # beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(s)[1]
    # # is about 2.4 ms slower
    
    beta_dict = {'beta_hat':beta}
    
    if backfill_candidates is not None:
        beta_dict = {k:beta_dict[k] for k in beta_dict.keys() if k in backfill_candidates}
        
    return beta_dict

def calc_beta_fuller(s, mkt, backfill_candidates=None):
    # slightly more involved version of a function to calculate linear beta in 
    # s = alpha + beta*mkt + epsilon
    # using least squares 
    # s and mkt both arrays
    # # this implementation keeps track of
    # # # estimate of capm beta
    # # # uncertainty windows about beta (don't allow user to define band, just set it at 95% under G-M)
    # # # full vector of betas
    # # # unbiased estimator of residual standard deviation
    # # # idiosyncratic volatility (biased estimator of residual s.d.)
    # # # degrees of freedom for final fit 
    # # # (internal) standardized residuals from raw regression for outlier detection 
    # # # # external would be preferred, but this is proof of concept, no?
    # # # restricted design matrix, with outliers removed 
    # # # # having access to this is necessary for confidence intervals if calculated later
    # # # restricted returns, with outliers removed 
    # # # # having access to this is necessary for model diagnostics if calculated later
    # # it also
    # # # ensures that there are sufficient observations (naively)
    # # # removes outliers from calculation based on studentized residuals (|st_res|>3)
    
    assert len(s) == len(mkt)

    if len(s) > 60: #TODO this is arbitrary; could make an estimate based on c.i. prior
        # First pass, before taking out any outliers
        # build a design matrix, and include an intercept
        X = np.vstack([np.ones(len(mkt)), mkt]).T
        
        # degrees of freedom delta and degrees of freedom
        ddf = X.shape[1]
        df = X.shape[0] - ddf
                    
        # betas
        betas = np.linalg.lstsq(X, s, rcond=None)[0]
        
        # residuals
        eps_hat = s - X.dot(betas)        
        
        # idio vol
        idio_vol = ( eps_hat ).std()
        
        # realized vol
        rvol = s.std()
        
        # estimate of standard deviation of epsilon (unbiased, with df)
        sigma_hat = eps_hat.std(ddof=ddf)
        
        # hat matrix ( in H*Y = X\hat{\beta} )
        H = X.dot(np.linalg.pinv(X))
        
        # studentized residuals for full regression
        student_res_raw = eps_hat / ( sigma_hat * np.sqrt( 1 - np.diag(H) ) ) 
        
        msk = np.abs(student_res_raw) > 3
        if msk.sum() >0: # there are outliers
            mkt_res = mkt[~msk]
            s_res = s[~msk]
            # rerun analysis omitting 'outliers'
            X_res = np.vstack([np.ones(len(mkt_res)), mkt_res]).T
            
            # degrees of freedom delta and degrees of freedom
            ddf = X_res.shape[1]
            df = X_res.shape[0] - ddf
                    
            # betas
            betas = np.linalg.lstsq(X_res, s_res, rcond=None)[0]
            
            # residuals (on res set)
            eps_hat_res = s_res - X_res.dot(betas)  
            
            # estimate of variance of epsilon  (based on res set)
            sigma_hat = eps_hat_res.std(ddof=ddf)
            
            # idiosyncratic vol (includes outliers, but with updated betas)
            idio_vol = ( s - X.dot(betas) ).std()
            
        else:
            X_res = X
            s_res = s
            eps_hat_res = eps_hat

              
        # the capm beta is*         
        beta_hat = betas[1]
        # *capm doesn't have an intercept, but it's the right thing to do    
        
        # 95% confidence interval for beta_hat (with outliers removed)          
        beta_ci = beta_hat_ci(X_res, betas, sigma_hat, df, alpha=0.05)
        bhat_ci = beta_ci[1]
        
        # R2 from fit (this will be greater than reality; i.e., an upper bound)
        R2 = 1 -  ( ( eps_hat_res**2 ).sum() )/( ( ( s_res - s_res.mean() )**2 ).sum() )  
        
    else: # not enough observations for analysis
        #TODO there should probably be a success/failure flag in here somewhere
        beta_hat = np.nan
        bhat_ci = np.nan*np.ones(2)
        betas = np.nan
        sigma_hat = np.nan
        idio_vol = np.nan
        rvol = np.nan
        df = np.nan
        R2 = np.nan
        student_res_raw = np.nan
        X_res = np.nan
        s_res = np.nan
        X = np.nan
        s = np.nan
        
    
    beta_dict = {'beta_hat':beta_hat, 
                 'beta_hat_ci':bhat_ci, 
                 'betas':betas, 
                 'sigma_hat':sigma_hat,
                 'idio_vol':idio_vol, 
                 'rvol':rvol,
                 'df':df,
                 'R2':R2,
                 'st_res_raw':student_res_raw,
                 'X_res':X_res,
                 's_res':s_res,
                 'X':X,
                 's':s}
    
    if backfill_candidates is not None:
        beta_dict = {k:beta_dict[k] for k in beta_dict.keys() if k in backfill_candidates}
        
    
    return beta_dict

def mean_response_ci(X, betas, sigma_hat, df, alpha=0.05):
        # for design matrix, X
        # estimate of betas, betas
        # estimate of s.d. of residual
        # and degrees of freedom, df
        # return a (1-alpha)% confidence interval as an array
        
        # set critical value
        # TODO check alpha is in appropriate range
        # TODO sanity check on df, etc
        t_val = t.ppf(1-alpha/2, df=df)
        
        # keep track of (X'X)^{-1} for prediction interval / standard error
        C = np.linalg.inv(X.T.dot(X)) 
        
        # confidence interval for mean response (assuming G-M)
        mean_res_delta = t_val*sigma_hat*np.sqrt( np.diag( ( X.dot(C) ).dot(X.T) ) )
        mean_res_ci = np.vstack([X.dot(betas) - mean_res_delta, X.dot(betas) + mean_res_delta])
        
        return mean_res_ci
    
def beta_hat_ci(X, betas, sigma_hat, df, alpha=0.05):
        # for design matrix, X
        # estimate of betas, betas
        # estimate of s.d. of residual
        # and degrees of freedom, df
        # return a (1-alpha)% confidence interval as an array
        
        # set critical value
        # TODO check alpha is in appropriate range
        # TODO sanity check on df, etc
        t_val = t.ppf(1-alpha/2, df=df)
        
        # Confidence intervals (assuming G-M)
        t_val = t.ppf(0.975, df=df)
        
        # keep track of (X'X)^{-1} for prediction interval / standard error
        C = np.linalg.inv(X.T.dot(X))        
        
        # 95% confidence interval for beta_hat (with outliers removed)  
        c = np.sqrt(np.diag(C))
        beta_delta = t_val*c*sigma_hat
        beta_ci = np.vstack([betas - beta_delta, betas + beta_delta]).T
        
        return beta_ci   
    
    
def make_reference_dataframe(df_prc, df_mkt):
    # for ease, merge in market returns as an extra column for easy alignment
    df_ref = df_prc.reset_index().\
                            set_index('Date').\
                            merge(df_mkt.reset_index().set_index('Date')[['Returns']], left_index=True, right_index=True)
    df_ref.rename({'Returns_x':'Returns', 'Returns_y':'Market'}, axis = 1, inplace=True)
    df_ref = df_ref.reset_index().set_index(['Ticker','Date']).sort_index()
    
    # this exercise is not going to identify a procedure for filling nans
    df_ref.dropna(how='any', axis=0, inplace=True)  
    
    # don't need price and volume in what we are doing here
    df_ref.drop(['Adj Close', 'Volume'], axis=1, inplace=True)
    
    return df_ref

def calc_betas_single_date(df_ref, calc_date, window=180, beta_func=calc_beta_basic, date_index=None, backfill_candidates=None):
    # calculate betas on a single day
    # df_ref is a dataframe with both stock and market returns 
    # it has a multiindex of ticker x date
    # calc date is a date in df_ref 
    # beta calculations will be made with beta_func
    # output is a dictionary 
    
    print('\n Calculating betas for {0}'.format(calc_date))
    
    calc_date = pd.to_datetime(calc_date, errors='coerce')
    
    if date_index is None:
        date_index = df_ref.index.get_level_values(1)
        # prevents having to call it all the time

    msk = ( date_index>=calc_date-dt.timedelta(days = window) ) & ( date_index<=calc_date)
    

    
    if sum(msk) > 0:
        df_res = df_ref[msk].copy() # restriced dataframe by date range
        
        # I kind of hate this here because it's in a loop in another function
        # but not all tickers will exist in all time windows
        unique_tickers = df_res.index.get_level_values(0).unique() 
        
        # do the work
        tickers_done = 0
        total_tickers = len(unique_tickers)

        # do the work    
        output_dict = dict()
        
        for ticker in unique_tickers:    
            if total_tickers>10:
                done = int(50 * (tickers_done / total_tickers))
                percent_progress = '%.0f' % (tickers_done / total_tickers * 100)
                sys.stdout.write('\r[%s%s] %s%%' % ('#' * done, ' ' * (50 - done), percent_progress))
                sys.stdout.flush()
            
            df_ts = df_res.xs(ticker, level=0) # time series dataframe for given ticker
            if df_ts.index.max() == calc_date:  # the calc date is in the time series
                beta_dict = beta_func(df_ts['Returns'].values, df_ts['Market'].values, backfill_candidates=backfill_candidates)
                beta_dict['date'] = calc_date
                
                output_dict[ticker] = beta_dict
                
            tickers_done = tickers_done + 1
    else:
        print('Invalid date range')
            
    return output_dict

def calc_betas_rolling(df_ref, window=180, beta_func=calc_beta_basic):
    # df_ref is a dataframe with both stock and market returns
    # it has a multiindex of ticker x date
    # df_bta will be the output.  The column candidates will be 
    backfill_candidates = ['beta_hat', 'beta_hat_ci', 'R2', 'idio_vol', 'rvol']
    # but the beta_func we use is only required to calculate beta_hat
    
    # we need to get some handle on all dates and unique dates
    # available to us
    date_index = df_ref.index.get_level_values(1)
    unique_dates = date_index.unique()
    
    # figure out the first viable date based on the window you are looking at
    first_date = min(unique_dates) + dt.timedelta(days = window)
    
    # loop dates; restrict dataframe to that date range; loop tickers
    print('Looping by dates')
    df_bta = pd.DataFrame()
    
    for calc_date in unique_dates[unique_dates>=first_date]:       
        calc_date_dict = calc_betas_single_date(df_ref, 
                                                calc_date, 
                                                window=window, 
                                                beta_func=beta_func, 
                                                date_index=date_index,
                                                backfill_candidates=backfill_candidates)
        calc_date_df = pd.DataFrame(calc_date_dict).T
        calc_date_df.index.name = 'Ticker'
        
        
        col_msk = [ calc_cols for calc_cols in backfill_candidates if calc_cols in calc_date_df.columns ]
        calc_date_df = calc_date_df[col_msk]
        calc_date_df['Date'] = calc_date
        
        if df_bta.shape[0] == 0:
            df_bta = calc_date_df
        else:
            df_bta = pd.concat([df_bta, calc_date_df])
            
    df_bta = df_bta.reset_index().set_index(['Ticker', 'Date'])
            
    return df_bta
    
                 