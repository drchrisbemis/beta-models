# -*- coding: utf-8 -*-
# Perhaps nonstandard dependencies
# # quandl
# # yfinance
# # requests

import pandas as pd
import time

# modules for this project
import config_files as config 
import mod_downloads as download
import jump_figures as jfig
import jump_funcs as jf


''' Input Parameters '''
# set the start date and end date for price retrieval 
sdate = '2018-01-01'
edate = None

# and note, beta calculations have a lookback window
trailing_calendar_days = config.TRAILING_WINDOW

''' Download ticker cross section and market cap '''
# first thing, get a bulk download from quandl 
# ... gets cross section of tickers as zip
download.bulk_files()

# process the ticker cross-section we will be using (from that zip file)
df_tkr = download.ticker_file()

# note the full ticker list
full_ticker_list = list(df_tkr.index.get_level_values(0))

# get market cap data for the full ticker list from quandl
df_qd = download.quandl_recent_market_caps(full_ticker_list)

''' Subselect 300 names per spec '''
# select a list of 300 tickers based on market cap segment strata
# # nothing too special in selection.  didn't want to add randomness.  
# # takes top (up to) 100 from Largest, Mid, and Small cap
ticker_list = jf.select_ticker_by_mkt_cap_seg(df_qd)

''' Download pricing information '''
# pull yahoo price information for the ticker list defined above and 
# date range prescribed
df_prc = download.yahoo_prices(ticker_list, start_date=sdate, end_date=None)
df_prc = jf.calc_returns(df_prc)
df_prc.sort_index(inplace=True) # doesn't matter in beta calcs, but good practice for serial dependence

''' Massage data '''
# since the project is about beta, need to get a market reference; using SPY
df_mkt = download.yahoo_prices(config.MKT_TICKER, start_date=sdate, end_date=None)
df_mkt = jf.calc_returns(df_mkt)

# make a reference dataframe with both stock and market returns for ease
# # this is proxying for some database call in some sense
# # it is also duplicating market returns across dates (since there is a multiindex in df_prc)
df_ref = jf.make_reference_dataframe(df_prc, df_mkt)

# here is a smaller dataframe with 10 names to play around with
df_sml = df_ref.copy()
df_sml = df_sml.reset_index().set_index('Ticker')
df_sml = df_sml.loc[ [t for t in ticker_list[:10] if t in df_sml.index ] ]
df_sml = df_sml.reset_index().set_index(['Ticker', 'Date']).sort_index()

''' Example usage for backfills '''
# I allowed for multiple function inputs as a beta calculator
# # the spec had this feeling of expecting an iterative growth; 
# # I tried to reflect that

''' ~Simple '''
# first, barebones example (on some small subset), using just np least squares
t0 = time.time()
df_bta0 = jf.calc_betas_rolling(df_sml, 
                               window=trailing_calendar_days, 
                               beta_func=jf.calc_beta_basic)
time_simple = time.time() - t0

''' ~Fuller Suite '''
# next, more involved example (on some small subset), 
# with some outlier detection, and stat calcs
t0 = time.time()
df_bta1 = jf.calc_betas_rolling(df_sml, 
                               window=trailing_calendar_days, 
                               beta_func=jf.calc_beta_fuller)
time_fuller = time.time() - t0

''' ~Some timing notes '''
# say some things about timing (could use timeit here, but kept it simple)
print('Using just a least squares fit (no bells and whistles)...')
names = len(df_bta0.index.get_level_values(0).unique())
dates = len(df_bta0.index.get_level_values(1).unique())
print('Backfilling {0} names over {1} dates took {2:.2f} seconds'.\
      format(names,
             dates,
             time_simple))
print(df_bta0.head())

print('Using just a slightly more feature complete approach...')
names = len(df_bta1.index.get_level_values(0).unique())
dates = len(df_bta1.index.get_level_values(1).unique())
print('Backfilling {0} names over {1} dates took {2:.2f} seconds'.\
      format(names,
             dates,
             time_fuller))
print(df_bta1.head())

''' ~Full backfill '''
# a full backfill may be obtained by calling jf.calc_betas_rolling with df_ref 
# takes about 860 s
# t0 = time.time()
# df_bta = jf.calc_betas_rolling(df_ref, 
#                                window=trailing_calendar_days, 
#                                beta_func=jf.calc_beta_fuller)
# time_full = time.time() - t0
# print('For the whole sample...')
# names = len(df_bta.index.get_level_values(0).unique())
# dates = len(df_bta.index.get_level_values(1).unique())
# print('Backfilling {0} names over {1} dates took {2:.2f} seconds'.\
#       format(names,
#              dates,
#              time_full))
# print(df_bta.head())

# I saved output from a previous run in a pkl
df_bta = pd.read_pickle('full_beta_backfill.pkl')

''' Next look at some figures based on the data '''
# Consider a time series for a particular name
tkr = 'AAPL'
jfig.time_series_plots(df_bta, tkr)

# Or a look into a single day's fit
# # uses the same dataframe as backfill (i.e., it would call same database)
# # or, you could store output from backfill ... either way
calc_date = pd.to_datetime('2020-11-01')
jfig.fit_diagnostic_plots(df_ref, tkr, calc_date, window=180)

''' Finally, consider an alternative to OLS '''
import jump_ml as jml
(hr_mdl, X, s) = jml.hr_example(df_ref, 'AAPL', calc_date, window=500)
fit_reg = hr_mdl.best_estimator_.named_steps['hr']

print('Coefficients from a huber regressor fit using k-fold cross validation...')
print('Intercept: {0:.4f}, Slope: {1:.4f}'.format(fit_reg.intercept_, fit_reg.coef_[0]))
print('Linear band: {0:.4f} s.d.'.format(fit_reg.epsilon))

''' Make a figure with an interpretation of the regressor '''
jfig.huber_regressor_plot_poc(hr_mdl, X, s, tkr, calc_date)
















