#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime as dt
import os
import pandas as pd
import requests
import sys
import zipfile

import quandl
import yfinance as yf

import config_files as config
import mod_utils as util

def unzip_file(zip_location, target_location):
    if not zipfile.is_zipfile(zip_location):
        print('ERROR: The file given is not a zip file.')
        file = open(zip_location, 'r')
        print(file.read())
    else:
        unzipped_file = zipfile.ZipFile(zip_location)
        print('Extracting file to {0:s}'.format(target_location))
        unzipped_file.extractall(target_location)
        unzipped_file.close()


def download_zip_file(url, filename):
    r = requests.get(url, stream=True)
    total_length = int(r.headers.get('content-length'))
    zip_path = os.path.join(util.get_script_dir(), 'ZIP')
    zip_filename = util.create_unique_filename(
        filename,
        'zip',
        zip_path,
    )
    with open(zip_filename, 'wb') as f:
        print('Downloading {0:s}'.format(filename))
        print('Downloading to {0:s}'.format(zip_filename))
        print('File size: {0:.2f} MB'.format((total_length / 1024 / 1024)))
        num_chunks = 0
        total_chunks = total_length / 4096
        for chunk in r.iter_content(chunk_size=4096):
            if chunk:
                num_chunks += 1
                f.write(chunk)
                done = int(50 * (num_chunks / total_chunks))
                percent_progress = '%.0f' % (num_chunks / total_chunks * 100)
                sys.stdout.write('\r[%s%s] %s%%' % ('#' * done, ' ' * (50 - done), percent_progress))
                sys.stdout.flush()

    print('')
    raw_path = os.path.join(util.get_script_dir(), 'RAW')
    unzip_file(zip_filename, raw_path)
    os.remove(zip_filename)


def bulk_files():
    
    print('Downloading zip {0:s} from {1:s}'.format(config.QUANDL_TICKERS_DETAILS['file'], config.QUANDL_TICKERS_DETAILS['uri']))
    r = requests.get(config.QUANDL_TICKERS_DETAILS['uri'])
    file_data = r.json()
    zip_uri = file_data['datatable_bulk_download']['file']['link']
    download_zip_file(zip_uri, config.QUANDL_TICKERS_DETAILS['file'])
    
def ticker_file():
    # create dataframe from bulk download
    
    raw_path = os.path.join(util.get_script_dir(), 'RAW')
    ticker_filename = util.get_recent_file(raw_path + '\\SHARADAR_TICKERS*.csv')
    
    fields = ['table', 'ticker', 'cusips', 'isdelisted', 'exchange', 'category', 'famaindustry', 'sector' ]
    df_tkr = pd.read_csv(ticker_filename, usecols=fields)
    # clean this up
    # Want to only look at SF1 table (matches fundamentals from vendor)
    # Want to remove any delisted names
    # Only want to look at US common stocks
    # that are not OTC
    df_tkr = df_tkr[ df_tkr['table'] == 'SF1' ]
    df_tkr = df_tkr[ df_tkr['isdelisted'] == 'N' ]
    df_tkr = df_tkr[ df_tkr['category'] == 'Domestic Common Stock' ]
    df_tkr = df_tkr[ df_tkr['exchange'] != 'OTC' ]
    # drop the stuff that you won't use anymore
    df_tkr.drop(['table', 'isdelisted', 'category', 'exchange'], axis=1, inplace=True)
    # set the index and take out any dupes (just in case)
    df_tkr.set_index('ticker', inplace=True)
    df_tkr = df_tkr[~df_tkr.index.duplicated(keep='first')]
    
    return df_tkr    

def yahoo_prices(ticker_list, start_date='2018-01-01', end_date=None):
    print('Downloading pricing information from Yahoo')
    df_prc = yf.download(ticker_list, start=start_date, end=end_date)
    
    # massage this dataframe so that it has an index of ticker x time,
    # and columns with Adj Close and Volume
    
    print('Creating multiindex dataframe from download')
    if type(ticker_list) is list:
        # first set multtiindex
        df_prc = df_prc.unstack().reset_index()
        df_prc.rename({'level_1':'Ticker'}, axis=1, inplace=True)
        df_prc.set_index(['Ticker','Date'], inplace=True)
        
        # next, make this tall table into a df with columns of values you care about
        df_prc = df_prc.pivot(index=df_prc.index, columns='level_0').droplevel(0,axis=1)
        df_prc.columns.name=None # would be level_0 and not necessary
        df_prc = df_prc[['Adj Close', 'Volume']]
        
        # finally, clean up those tickers that have no price data returned
        df_prc = df_prc.groupby(level=0).\
                          apply(lambda x: x.dropna(how='all', subset=['Adj Close'])).\
                               droplevel(level=0, axis=0)    
    else:
        df_prc.reset_index(inplace=True)
        df_prc['Ticker'] = ticker_list
        df_prc.set_index(['Ticker','Date'], inplace=True)
        df_prc = df_prc[['Adj Close', 'Volume']]
    
    return df_prc

def quandl_recent_market_caps(ticker_list):
    # this is a quick and dirty example with a low-frequency measure of market cap
    # the intention is to show proof of concept work
    # this requires the quandl package to be installed
    
    print('Pulling market cap data from Quandl')
    # this is my quandl api key
    quandl.ApiConfig.api_key = config.quandl_api_key
    
    # the data is low frequency, based in part on filings; take a quarter look back
    date_lb = (dt.datetime.today() - dt.timedelta(days=120) ).strftime('%Y-%m-%d')
    qd = quandl.get_table('SHARADAR/SF1', qopts = {'columns': ['ticker', 'datekey', 'marketcap'] }, 
                 ticker=ticker_list, paginate=True, datekey = { 'gte': date_lb })
    
    # TODO restrict based on staleness of data
    # take the most recent data ... there should be checks on how stale the data is here
    # but, again, this is proof of concept
    qd = qd.groupby('ticker').apply(lambda x: x.sort_values(by='datekey')).droplevel(level=1, axis=0)
    qd = qd.drop_duplicates('ticker', keep='last')
    qd.drop(['ticker'], axis=1, inplace=True)
    qd.rename({'marketcap':'market_cap'}, axis=1, inplace=True)
    
    return qd

if __name__ == '__main__':
    bulk_files()
