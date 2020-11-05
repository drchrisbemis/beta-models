quandl_api_key = 'kKj_zNobZLnLCzbs6WEJ' # this is mine

QUANDL_TICKERS_DETAILS = {
    'file': 'QUANDL_TICKERS',
    'uri': 'https://www.quandl.com/api/v3/datatables/SHARADAR/TICKERS.json?qopts.export=true&api_key={0}'.format(quandl_api_key)
}



TICKERS_RAW_COLUMNS = [
        'TICKER',
        'NAME',
        'ISDELISTED',
        'SECTOR',
        'LASTPRICEDATE',]


# Market reference ticker
MKT_TICKER = 'SPY'

# calendar days for beta and vol calculations
TRAILING_WINDOW = 180


