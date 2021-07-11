import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import pandas_datareader.data as web

##################
## data section ##
##################
def csv_extractor(ticker,apath):
    # a. reading csv as pandas dataframe, renaming and formating date column
    temp_df = pd.read_csv(apath+ticker+'.csv', usecols = ['Date', 'Price'])
    temp_df = temp_df.rename(columns = {'Price': ticker})
    temp_df['Date'] = pd.to_datetime(temp_df['Date'])
    
    # b. sort and index by date
    temp_df = temp_df.sort_values(by = 'Date')
    temp_df = temp_df.set_index('Date', drop = True)
    return temp_df


def yahoo_extractor(ticker, start, end):
    # a. loading total dataset from yahoo
    temp_df = web.DataReader(ticker, 'yahoo', start, end)
    # here one could do web.DataReader()['Adj Close']
    
    # b. filtering and renaming columns
    temp_df = pd.DataFrame(temp_df['Adj Close']).rename(columns = {'Adj Close': ticker})
    return temp_df


def df_generator(tickers, method, apath = None, start = None, end = None):
    
    # i. temporarily storing each df in a list
    temp_store = []
    
    # making sure datetime-message only occurs 1 time
    msg = 0
    # ii. creating a df for each ticker
    for i in tickers:

        # ii.a. extracting data from yahoo database
        if method == 'yahoo':
            # Message regarding datatime input
            if start == None and msg == 0:
                y5_ago = (dt.datetime.now()-dt.timedelta(days=5*365)).strftime('%Y-%m-%d')
                msg += 1
                print(f'No starttime selected, has therefore chosen default closest to (5 years before today) {y5_ago}')
                
            if end == None and msg < 2:
                today = dt.datetime.now().strftime('%Y-%m-%d')
                msg += 2
                print(f'No endtime selected, has therefore chosen default closest to (today) {today}')

            vector = yahoo_extractor(i, start, end)
        # ii.b extracting data from investing.com csv-files
        elif method == 'csv':
            assert apath != None, 'Please denote the absolute path to csv-file(s)'
            vector = csv_extractor(i, apath)
        
        # ii.c doesn't recognize data-extraction-method
        else:
            print('did not recognize method, please use either yahoo or csv')
        temp_store.append(vector)
    
    # iii. combining the dfs
    df = pd.concat(temp_store, axis = 1)
    df = df.fillna(method = 'ffill') # forwards fills
    
    # iv. creating a second df with pct.-returns
    rdf = (df/df.shift(1)-1)*100
    
    # v. creating a third df with cumulative return
    cdf = (df/df.shift(1)-1)
    for col in cdf.columns:
        cdf[col] = np.cumprod(1+cdf[col])*100
    
    # vi. describes data
    des_df(df)

    return df, rdf, cdf


def des_df(dfs):
    print('Will eventually output description of each ticker in df')


####################
## describe ticks ##
####################
def desc_ticks(data,rdata,cdata,rfree=0):

    ###########
    ## plots ##
    ###########
    # 1st plot: heatmap of correlations 
    plt.subplots(figsize=(8,6))
    sns.heatmap(rdata.corr(),
                vmin=-1, vmax=1, center=0,
                annot=True,
                cmap='RdGy').set_title('Correlation heatmap')
    plt.tight_layout()
    
    # 2nd plot: cummulative returns
    cdata.plot(figsize=(10,7), title = 'Cummulative return');

    # 3rd plot(s): distribution of returns for each ticker
    for i in rdata.columns:
        plt.figure(figsize = (10,7))
        sns.distplot(rdata[i], kde=True, bins=50).set_title(f'Distribution of returns for {i}')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.show()
    
    
    ###########
    ## stats ##
    ###########
    # i. initializing storage
    sol_shape = len(rdata.columns)
    a_returns = np.nan*np.zeros(sol_shape)
    a_volatilities = np.nan*np.zeros(sol_shape)
    a_sr = np.nan*np.zeros(sol_shape)
    a_cr = np.nan*np.zeros(sol_shape)
    
    # ii. calculating stats for each ticker
    for i, col in enumerate(rdata.columns):
        # a. storing number of days the ticker has been around
        start = data[col].first_valid_index()
        end = data[col].last_valid_index()
        no_days = (end - start).days

        # b. calculating stats
        absolute_return = data[col][data[col].last_valid_index()]/data[col][data[col].first_valid_index()]-1
        ar = round(((1+absolute_return)**(1/(no_days/252))-1)*100,2)
        vol = round(rdata[col].std()*np.sqrt(252),2)
        sr = round((ar - rfree) / vol,2)
        cr = round(ar / abs(rdata[col].min()),2)

        # c. storing stats
        a_returns[i] = ar
        a_volatilities[i] = vol
        a_sr[i] = sr
        a_cr[i] = cr

    # iii. creating dataframe with stats
    stats = pd.DataFrame({'Ticker': rdata.columns,
                          'annualized_return': a_returns,
                          'volatility': a_volatilities,
                          'sharpe_ratio': a_sr,
                          'calmar_ratio': a_cr})
    
    # iv. display stats
    return stats


######################
## optimal porfolio ##
######################


##########################################
## analytical solution: 2 funds theorem ##
##########################################









############################
## monte carlo simulation ##
############################





###########
## to-do ##
###########
# histplot vs. distplot
# one is deprecated, make sure you have relevant version of seaborn


# optimal portfolio
## analytical solution (mutual funds theorem)
## simulated results (in parallel)
## Get stats for the portfolio


# include in stats: 
## Stats for how much of a normal distribution it is 
## Value at Risk (see financial econometrics), both normally distributed and skewed
## Expected Shortfall (see financial econometrics), both normally distributed and skewed

# inlcude in plots:
## Simulated path

# add docstrings

# important
# should not use fillnas when generating dataframe.
## data will look stable when estimating as ARCH(1)
## however having na's will probably fuck up the percentage returns?
## perhaps one should use log returns all the way through.



###############
## graveyard ##
###############
# def df_generator_csv(tickers, apath):
#     # i. temporarily storing each df in a list
#     temp_store = []
    
#     # ii. creating a df for each ticker
#     for i in tickers:
#         vector = csv_extractor(i, apath)
#         temp_store.append(vector)
    
#     # iii. combining the dfs
#     df = pd.concat(temp_store, axis = 1)
#     df = df.fillna(method = 'ffill')
    
#     # iv. creating a second df with pct.-returns
#     rdf = (df/df.shift(1)-1)*100
    
#     # v. creating a third df with cumulativ return
#     cdf = (df/df.shift(1)-1)
#     for col in cdf.columns:
#         cdf[col] = np.cumprod(1+cdf[col])*100
    
#     return df, rdf, cdf
# def df_generator(tickers, method = 'yahoo', apath = None):
    
#     # i. extracts data
#     if method == 'yahoo':
#         df, rdf, cdf = df_generator_yahoo(tickers)
    
#     elif method == 'csv':
#         assert len(apath)>2, 'when using csv-files the absolute path of the folder in which data lies, must be specified'
#         df, rdf, cdf = df_generator_csv(tickers, apath)
        
#     else:
#         print('doesnt recognize method')
    
#     # ii. describes data
#     des_df(df)
    
#     return df, rdf, cdf

# def df_generator_yahoo(tickers, start, end):
#     # i. temporarily storing each df in a list
#     temp_store = []
    
#     # ii. creating a df for each ticker
#     for i in tickers:
#         vector = yahoo_extractor(i, start, end)
#         temp_store.append(vector)
    
#     # iii. combining the dfs
#     df = pd.concat(temp_store, axis = 1)
#     df = df.fillna(method = 'ffill')
    
#     # iv. creating a second df with pct.-returns
#     rdf = (df/df.shift(1)-1)*100
    
#     # v. creating a third df with cumulativ return
#     cdf = (df/df.shift(1)-1)
#     for col in cdf.columns:
#         cdf[col] = np.cumprod(1+cdf[col])*100
    
#     return df, rdf, cdf

# def df_generator_yahoo(tickers):
#     print('Yahoo dataextractor will be implemented later')











