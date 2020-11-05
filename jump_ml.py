# -*- coding: utf-8 -*-
''' Quick and dirty ml pipeline using huber regressor...
data veracity is not checked.  Intention is to show process'''

import datetime as dt
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import HuberRegressor

def hr_example(df_ref, tkr, calc_date, window=500):
    # example of an alternative formulation for least squares fit
    # using some ml
    
    print('Fitting a Huber Regressor model')
    calc_date = pd.to_datetime(calc_date, errors='coerce')
    
    date_index = df_ref.index.get_level_values(1)
    # prevents having to call it all the time
    
    msk = ( date_index>=calc_date-dt.timedelta(days = window) ) & ( date_index<=calc_date)
    
    df_res = df_ref[msk].copy()
    
    df_ts = df_res.xs(tkr, level=0) # time series dataframe for given ticker
    
    s = df_ts['Returns'].values 
    mkt = df_ts['Market'].values
    
    X =  mkt.reshape(-1,1)
    
    # construct a pipeline
    mdl = Pipeline([('scaler',None), ('hr', HuberRegressor(fit_intercept=True))])
    
    parameters = {'hr__epsilon':np.linspace(1,4,20),
                  'hr__alpha':np.logspace(-4,-2,3)}
    
    mdl = GridSearchCV(mdl,
                       param_grid=parameters,
                       n_jobs=-1,
                       cv=KFold(n_splits=10, shuffle=True, random_state=0),
                       scoring='neg_median_absolute_error',
                       return_train_score=True,
                       refit=True,
                       error_score=np.nan)
    
    mdl.fit(X,s)
    
    return (mdl, X, s)
