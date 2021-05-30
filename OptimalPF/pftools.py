import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web

##################
## data section ##
##################


def csv_extractor(ticker,apath):
    temp_df = pd.read_csv(apath+ticker+'.csv', usecols = ['Date', 'Price'])
    temp_df = temp_df.rename(columns = {'Price': ticker})
    temp_df['Date'] = pd.to_datetime(temp_df['Date'])
    temp_df = temp_df.sort_values(by = 'Date')
    temp_df = temp_df.set_index('Date', drop = True)
    return temp_df

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

def yahoo_extractor(ticker, start, end):
    temp_df = web.DataReader(ticker, 'yahoo', start, end)
    # temp_df = pd.DataFrame(temp)
    temp_df = temp_df['Adj Close']
    temp_df[ticker] = temp_df
    # temp_df.rename(columns = {'Adj Close': ticker})
    # temp_df = temp_df.rename(columns = {'Adj Close': ticker})
    return temp_df


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


def df_generator(tickers, method, start = None, end = None, apath = None):
    # i. temporarily storing each df in a list
    temp_store = []
    
    # ii. creating a df for each ticker
    for i in tickers:
        if method == 'yahoo':
            # assert start == None, 'please denote a start-time (dt-format) when extracting data'
            # assert end == None, 'please denote a end-time (dt-format) when extracting data'
            vector = yahoo_extractor(i, start, end)
        elif method == 'csv':
            assert apath == None, 'please denote an absolute path to your local csv-files'
            vector = csv_extractor(i, apath)
        else:
            print('did not recognize method, please use either yahoo or csv')
        temp_store.append(vector)
    
    # iii. combining the dfs
    df = pd.concat(temp_store, axis = 1)
    df = df.fillna(method = 'ffill')
    
    # iv. creating a second df with pct.-returns
    rdf = (df/df.shift(1)-1)*100
    
    # v. creating a third df with cumulativ return
    cdf = (df/df.shift(1)-1)
    for col in cdf.columns:
        cdf[col] = np.cumprod(1+cdf[col])*100
    
    # ii. describes data
    des_df(df)

    return df, rdf, cdf



# def df_generator_yahoo(tickers):
#     print('Yahoo dataextractor will be implemented later')


def des_df(dfs):
    print('Will eventually output description of each ticker in df')


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
        sns.histplot(rdata[i], kde=True, bins=50).set_title(f'Distribution of returns for {i}')
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

