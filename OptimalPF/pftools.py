import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from dateutil.relativedelta import relativedelta
import seaborn as sns
import pandas_datareader.data as web
from scipy import optimize


##################
## data section ##
##################
def csv_extractor(ticker,path):
    '''
    Loads csv-file, on a selected local path, for a given ticker and spits out series of ajusted close prices
    
    Args:
    ticker (string): should be exact name of csv-file - oftentimes ticker name
    path (string): denotes the absolute or relative path to the given csv-file
    
    Returns:
    (dataframe): dataframe containing adjusted close for given ticker along with an index of dates    
    '''
    # a. reading csv as pandas dataframe, renaming and formating date column
    temp_df = pd.read_csv(path+ticker+'.csv', usecols = ['Date', 'Price'])
    temp_df = temp_df.rename(columns = {'Price': ticker})
    temp_df['Date'] = pd.to_datetime(temp_df['Date'])
    
    # b. sort and index by date
    temp_df = temp_df.sort_values(by = 'Date')
    temp_df = temp_df.set_index('Date', drop = True)
    return temp_df



def yahoo_extractor(ticker, start, end):
    '''
    Loads ticker data from yahoo database within a selected time period and spits out series of ajusted close prices
    
    Args:
    ticker (string): should be exact name of ticker as denoted in the yahoo database.
    start (datetime): first date of ticker data
    end (datetime): last date of ticker data
    
    Returns:
    (dataframe): dataframe containing adjusted close for given ticker along with an index of dates    
    '''
    # a. loading dataset from yahoo and only keeping adjusted close column
    temp_df = web.DataReader(ticker, 'yahoo', start, end)['Adj Close']
    
    # b. renaming column
    temp_df = pd.DataFrame(temp_df).rename(columns = {'Adj Close': ticker})
    return temp_df



def df_generator(tickers, method, path = None, start = None, end = None):
    '''
    Creates adjusted close dataframe from a list of tickers based on some method (being yahoo database or csv-files).
    
    Args:
    tickers (list): list of strings containing either yahoo databse tickers, or csv-file names.
    method (string): either 'yahoo' for yahoo extractir or 'csv' for local csv extractor
    path (string): denotes the absolute path of csv-files. Necessary when using csv extractor (method = 'csv')
    start (datetime): first date of ticker data. Can be included when using yahoo extractor (method = 'yahoo')
    end (datetime): last date of ticker data. Can be included when using yahoo extractor (method = 'yahoo')
    
    Returns:
    (dataframe): dataframe containing adjusted close for a list of tickers with an index of dates    
    '''
    # i. temporarily storing each df in a list
    temp_store = []
    
    # making sure datetime-message only occurs 1 time
    msg = 0
    # ii. creating a df for each ticker
    for i in tickers:

        # ii.a. extracting data from yahoo database
        if method.lower() == 'yahoo':
            # Message regarding datatime input
            if start == None and msg == 0:
                y5_ago = (dt.datetime.now()-dt.timedelta(days=5*365.2425)).strftime('%Y-%m-%d')
                msg += 1
                print(f'No starttime selected, has therefore chosen default closest to (5 years before today) {y5_ago}')
                
            if end == None and msg < 2:
                today = dt.datetime.now().strftime('%Y-%m-%d')
                msg += 2
                print(f'No endtime selected, has therefore chosen default which is the latest trading day as of {today}\n')

            vector = yahoo_extractor(i, start, end)
        # ii.b extracting data from investing.com csv-files
        elif method.lower() == 'csv':
            assert path != None, 'Please denote the absolute or relative path to csv-file(s)'
            vector = csv_extractor(i, path)
        
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
    cdf = (1+cdf).cumprod()*100
    
    # vi. describes data
    desc_df(df)

    return df, rdf, cdf



def desc_df(df):
    '''
    Will print description for the tickers which lack data compared to the rest.
    
    Args:
    df (dataframe): dataframe containing adjusted close for some tickers.
    
    Prints:
    Prints tickers that have less data compared to the rest along with how much less data is the case.    
    '''
    # i. prints first observation in the dataframe
    first_obs = df.first_valid_index()
    first_obs_date = df.first_valid_index().strftime('%d-%m-%Y')
    print(f'First observation in dataframe is         {first_obs_date}')
    
    # ii. prints ticker name and when it started (implyin how much less data), if it started later than df
    for i in df.columns:
        first_i_obs = df[i].first_valid_index()
        
        if first_obs < first_i_obs:
            rdelta = relativedelta(first_i_obs,first_obs)
            years = round((rdelta.years+rdelta.months/12+rdelta.days/365.2425),2)
            tradingdays = ((first_i_obs - first_obs)/365.2425*252).days
            first_i_obs_date = df[i].first_valid_index().strftime('%d-%m-%Y')
            print(f'Note that {i:10} first appears at the {first_i_obs_date}')
            print(f'                     This means theres is missing data from the first {years} years, corresponding to approximately {tradingdays} trading days/observations\n')
        


####################
## describe ticks ##
####################
def desc_ticks(df,rdf,cdf,rfree=0):
    '''
    Will describe tickers of dataframes with correlation plot, cumulative return plot and distribution of returns for each ticker. Furthermore stats such as annual return, volatility, Sharpe-ratio, 1m max drawdown and Calmar ratio are produced and displayed.
    
    Args:
    df (dataframe): dataframe containing adjusted close for some tickers.
    rdf (dataframe): dataframe containing returns for some tickers.
    cdf (dataframe): dataframe containing cumulative return for some tickers.
    rfree (float): risk-free interest rate (by default = 0)
    
    Returns:
    plots correlation heatmap, cumulative return and return distribution for each ticker 
    displays dataframe containing annual return, volatility, Sharpe-ratio, 1m max drawdown and Calmar ratio for each ticker. 
    (array): 1d array of annual return for each ticker
    '''
    ###########
    ## plots ##
    ###########
    # 1st plot: heatmap of correlations 
    plt.subplots(figsize=(8,6))
    sns.heatmap(rdf.corr(),
                vmin=-1, vmax=1, center=0,
                annot=True,
                cmap='RdGy').set_title('Correlation heatmap')
    plt.tight_layout()
    
    # 2nd plot: cummulative returns
    cdf.plot(figsize=(10,7), title = 'Cummulative return');

    # 3rd plot(s): distribution of returns for each ticker
    for i in rdf.columns:
        plt.figure(figsize = (10,7))
        sns.histplot(rdf[i], kde=True, bins=50).set_title(f'Distribution of returns for {i}')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.show()
    
    
    ###########
    ## stats ##
    ###########
    # i. initializing storage
    sol_shape = len(rdf.columns)
    a_returns = np.nan*np.zeros(sol_shape)
    a_volatilities = np.nan*np.zeros(sol_shape)
    a_sr = np.nan*np.zeros(sol_shape)
    m_mdd = np.nan*np.zeros(sol_shape)
    a_cr = np.nan*np.zeros(sol_shape)
    
    # ii. calculating stats for each ticker
    for i, col in enumerate(rdf.columns):
        # a. storing number of days the ticker has been around
        start = df[col].first_valid_index()
        end = df[col].last_valid_index()
        no_days = (end - start).days

        # b. calculating stats
        absolute_return = df[col][df[col].last_valid_index()]/df[col][df[col].first_valid_index()]-1
        ar = round(((1+absolute_return)**(1/(no_days/365))-1)*100,2)
        vol = round(rdf[col].std()*np.sqrt(252),2)
        sr = round((ar - rfree) / vol,2)
        monthlyreturn = (df[col]/df[col].shift(21)-1)*100
        drawdown = np.abs(monthlyreturn.min())
        cr = round(ar/drawdown,2)

        # c. storing stats
        a_returns[i] = ar
        a_volatilities[i] = vol
        a_sr[i] = sr
        m_mdd[i] = round(drawdown,2)
        a_cr[i] = cr

    # iii. creating dataframe with stats
    stats = pd.DataFrame({'Ticker': rdf.columns,
                          'Annualized return': a_returns,
                          'Volatility': a_volatilities,
                          'Sharpe ratio': a_sr,
                          '1m max drawdown': m_mdd,
                          'Calmar ratio': a_cr})
    
    stats = stats.set_index('Ticker')
    
    # iv. display stats
    display(stats)
    
    # return mu-vector consisting of the annualized return for each ticker
    return np.array(a_returns)



######################
## optimal porfolio ##
######################
####################################################
## 1. analytical solution (short-selling allowed) ##
####################################################
def mvar_analytical(df,sigma,mu,printres = True):
    '''
    Will calculate the analytical solution for a minimum variance portfolio. That is the allocation and the respective stats (annual return, volatility, Sharpe ratio, 1m max drawdown and Calmar ratio)
    
    Args:
    df (dataframe): dataframe containing adjusted close for tickers.
    sigma (dataframe): dataframe of the variance/covariance matrix for tickers.
    mu (array): array consisting of annual return for each ticker.
    printres (boolean): if True will print the results (by default = True).
    
    Returns:
    (dataframe, dataframe): first dataframe contains weights/allocation, while the second dataframe containts the portfolios stats (annual return, volatility, Sharpe ratio, 1m max drawdown and Calmar ratio)
    '''
    
    # i. calculating the optimal weights for a minimum variance portfolio
    weights = np.linalg.inv(sigma)@np.ones((len(sigma.columns)))
    weights = weights / sum(weights)
    
    # ii. calculating variance, standard deviation and expected return
    variance = weights@sigma@np.transpose(weights)
    sd = np.sqrt(variance)
    pfreturn = weights@mu
    drawdown, cr = calmar(weights,df,mu)
    
    # iii. creating dataframe containing weights and another with its stats  
    wopt = weights*100
    wopt = [round(w,2) for w in wopt]
    names = sigma.columns
    wpfdf = pd.DataFrame({'Ticker':names, 'weight':wopt})
    wpfdf = wpfdf.set_index('Ticker')
    statspfdf = pd.DataFrame.from_dict({'Annualized return':round(pfreturn,2), 'Volatility':round(sd,2),
                                        'Sharpe ratio':round(pfreturn/sd,2),'1m max drawdown':round(drawdown,2),
                                        'Calmar ratio':round(cr,2)},
                                       orient = 'index', columns = ['stats'])
    
    # iv. printing statement 
    if printres:
            print(f'-----------------------------------------------------------------------------------------------')
            print(f'\nThe analytical solution for the minimum variance portfolio (allows for shorting) resulted in:\n')
            display(wpfdf)
            print(f'With portfolio characteristics:\n')
            display(statspfdf)
            print(f'-----------------------------------------------------------------------------------------------')
    
    return wpfdf, statspfdf



# Analytical solution for the efficient tangent portfolio
def tan_analytical(df,sigma,mu,printres = True):
    '''
    Will calculate the analytical solution for the efficient tangent portfolio. That is the allocation and the respective stats (annual return, volatility, Sharpe ratio, 1m max drawdown and Calmar ratio)
    
    Args:
    df (dataframe): dataframe containing adjusted close for tickers.
    sigma (dataframe): dataframe of the variance/covariance matrix for tickers.
    mu (array): array consisting of annual return for each ticker.
    printres (boolean): if True will print the results (by default = True).
    
    Returns:
    (dataframe, dataframe): first dataframe contains weights/allocation, while the second dataframe containts the portfolios stats (annual return, volatility, Sharpe ratio, 1m max drawdown and Calmar ratio)
    '''
    # i. calculating the optimal weights for the efficient tangent portfolio
    weights = np.linalg.inv(sigma) @ mu
    weights = weights / sum(weights)
    
    # ii. calculating variance, standard deviation and expected return
    variance = weights@sigma@np.transpose(weights)
    sd = np.sqrt(variance)
    pfreturn = weights@mu
    drawdown, cr = calmar(weights,df,mu)
    
    # iii. creating dataframe containing weights and another with its stats  
    wopt = weights*100
    wopt = [round(w,2) for w in wopt]
    names = sigma.columns
    wpfdf = pd.DataFrame({'Ticker':names, 'weight':wopt})
    wpfdf = wpfdf.set_index('Ticker')
    statspfdf = pd.DataFrame.from_dict({'Annualized return':round(pfreturn,2),'Volatility': round(sd,2),
                                        'Sharpe ratio':round(pfreturn/sd,2),'1m max drawdown':round(drawdown,2),
                                        'Calmar ratio':round(cr,2)},
                                       orient = 'index', columns = ['stats'])
    
    # iv. printing statement 
    if printres:
            print(f'-----------------------------------------------------------------------------------------------')
            print(f'\nThe analytical solution for the efficient tangent portfolio (allows for shorting) resulted in:\n')
            display(wpfdf)
            print(f'With portfolio characteristics:\n')
            display(statspfdf)
            print(f'-----------------------------------------------------------------------------------------------')
    
    return wpfdf, statspfdf



def ana_optimal_portfolios(df,rdf,mu):
    '''
    Will call upon the analytical solution of the minimum variance portfolio and the efficient tangent portfolio.
    
    Args:
    df (dataframe): dataframe containing adjusted close for tickers.
    rdf (dataframe): dataframe containing returns.
    mu (array): array consisting of annual return for each ticker.
    
    Returns:
    displays the allocation of the minimum variance portfolio and the efficient tangent portfolio along with their stats (being annual return, volatility, Sharpe ratio, 1m max drawdown and Calmar ratio)
    '''
    # i. calculating relevant data for optimal portfolios
    sigma = rdf.cov()*252
    
    # ii. calling analytical optimizers
    wmvdf,statmvdf = mvar_analytical(df,sigma,mu)
    wtandf,stattandf = tan_analytical(df,sigma,mu)
    print(f'\nCannot analytically solve the maximum calmar ratio portfolio')



###########################
## 2. numerical solution ##
###########################
def mvar(w,sigma):
    '''
    Will calculate variance for some weights (w) and a variance/covariance matrix (sigma)
    
    Args:
    w (array): 1d array containing weights/allocations for a given portfolio.
    sigma (dataframe): dataframe of the variance/covariance matrix for tickers.
    
    Returns:
    (float): variance of given portfolio.
    '''
    # i. normalizing weights
    nw = w/sum(w)
    
    # ii. calculating variance based on given weights
    variance = nw @ sigma @ nw
    
    return variance



def tangent(w,sigma,mu):
    '''
    Will calculate Sharpe ratio for some weights (w), variance/covariance matrix (sigma) and annual returns (mu)
    
    Args:
    w (array): 1d array containing weights/allocations for a given portfolio.
    sigma (dataframe): dataframe of the variance/covariance matrix for tickers.
    mu (array): array consisting of annual return for each ticker.
    
    Returns:
    (float): Sharpe ratio of given portfolio.
    '''
    # i. normalizing weights
    nw = w/sum(w)

    # ii. calculating yearly variance+standard deviation based on given weights
    variance = nw @ sigma @ nw
    sd = np.sqrt(variance)

    # iii. calculating yearly return
    r = nw @ mu
    
    # iv. calculating respective sharpe ratio (with risk-free return = 0)
    sharperatio = r/sd
    
    return sharperatio



# helping function for the max calmar ration portfolio
def calmar(w,df,mu):
    '''
    Will calculate Calmar ratio, based on 1 month max drawdown and a risk-free rate of 0, for some weights (w), max drawdowns (based on df) and annual returns (mu)
    
    Args:
    w (array): 1d array containing weights/allocations for a given portfolio.
    df (dataframe): dataframe of adjusted close of the variance/covariance matrix for tickers.
    mu (array): array consisting of annual return for each ticker.
    
    Returns:
    (float): Calmar ratio of given portfolio.
    '''
    # i. normalizing weights
    nw = w/sum(w)

    # ii. calulating the max monthly drawdown of portfolio based on weights
    pfprice = nw@df.T # note that this will only be calculated when there is no nans
    monthlyreturn = (pfprice/pfprice.shift(21)-1)*100
    drawdown = np.abs(monthlyreturn.min())
    
    # iii. calculating yearly return
    r = nw @ mu
        
    # iv. calculating the portfolios respective calmar ratio ratio (with risk-free return = 0)
    calmarratio = r/drawdown
    
    return drawdown, calmarratio



def optimize_pf(pftype,sigma,mu,df,N,shorting = False):
    '''
    Will optimize some type of portfolio (pftype) based on available ticker and whether shorting is allowed.
    
    Args:
    pftype (string): denotes what type of portfolio to optimize, can be 'minvar', 'tangent' or 'calmar'.
    sigma (dataframe): dataframe of the variance/covariance matrix for tickers.
    mu (array): array consisting of annual return for each ticker.
    df (dataframe): dataframe of adjusted close of the variance/covariance matrix for tickers.
    N (integer): number of times the minimization should be executed (by default = 50)
    shorting (boolean): denotes whether shorting is allowed, handles boundaries of optimization (by default = False)
    
    Returns:
    (dataframe, dataframe): first dataframe contains weights of the optimal portfolio, while the other dataframe contains said portfolios stats (being annual return, volatility, Sharpe ratio, 1m max drawdown and Calmar ratio)
    '''
    # i. establishing names, number of assets and initiating storage
    names = df.columns
    M = len(names)
    fopt = np.inf # initialize optimal value
    wopt = np.nan # initialize optimal weights
    ws = [] # storing all optimal weights in list
    fs = [] # storing all optimal values of interest in list
    attempt = [] # storing the speficic no attempt
    
    # ii. creating bounds based on whether short-selling is allowed or not
    rng = np.random.default_rng(1995)
    if shorting == True:
        w0s = rng.uniform(-1+1e-8,1-1e-8,size = (N,M))
        bound = (-1+1e-8,1-1e-8)    
        # a. Calmar ration cannot be solved with shorting allowed??
        if pftype.lower() == 'calmar':
            print(f'Currently no numerical solution for maximum calmar ratio portfolio with shorting - will therefore be solved for\nSHORTING IS NOT ALLOWED')
            # bounds, weights cannot be negative, i.e. short-selling is not allowed
            w0s = rng.uniform(1e-8,1-1e-8,size = (N,M))
            bound = (1e-8,1-1e-8)
    else:
        # bounds, weights cannot be negative, i.e. short-selling is not allowed
        w0s = rng.uniform(1e-8,1-1e-8,size = (N,M))
        bound = (1e-8,1-1e-8)
    bounds = ((bound, ) * M)

    print(f'-'*55)
    
    # iii. objective functions
    if pftype.lower() == 'minvar':
        optimizing = 'variance'
        obj = lambda x: mvar(x,sigma)
        print(f'Will numerically solve the minimum variance portfolio')
    elif pftype.lower() == 'tangent':
        optimizing = 'Sharpe Ratio'
        obj = lambda x: -tangent(x,sigma,mu)
        print(f'Will numerically solve the efficient tangent portfolio')
    elif pftype.lower() == 'calmar':
        optimizing = 'Calmar Ratio'
        obj = lambda x: -calmar(x,df,mu)[1]
        print(f'Will numerically solve the calmar portfolio')
    else:
        print(f'Can only optimize portfolios: minimum variance (pftype = minvar), efficient tangent (pftype = tangent) and calmar ratio (pftype = calmar)')
    
    print(f'Multistart optimization')
    print(f'-'*55)
    
    # iv. multistart using SLSQP (bounded) minimizer
    for i, w0 in enumerate(w0s):
        # a. bounded optimization for given initial weights
        result = optimize.minimize(obj,w0,method = 'SLSQP',
                                  bounds=bounds)
        
        # b. storing solution (optimized value and its weights)
        f = result.fun

        # c. storing the optimal solution and eventual better optimal solutions
        if f < fopt:
            # 1. extracting weights from the optimal solution
            weights = result.x
            
            # 2. storing optimal value, its weights and corresponding return
            fopt = f
            wopt = weights/sum(weights)
            ropt = wopt @ mu
            
            # 3. making sure the sign is proper for the given portfolio
            if pftype.lower() == 'minvar':
                f = f
            else:
                f = -f
            
            # 4. storing all improved solutions for eventual print statement
            fs.append(f)
            ws.append(wopt)
            attempt.append(i)
            

    # v. summarize the performed optimizations
    print(f'Performed multistart optimization with N = {N} total attempts')
    print(f'The optimal solution improved {len(ws)} times')
    
    
    # vi. print statement depending on whether optimal solutions have changed noticably during multistart optimization
    wsols = np.around(np.array(ws),2)
    testsols = np.all(wsols == wsols[0], axis=1)
    if np.sum(testsols) < len(testsols):
        print(f'The improvements of the optimal solution resulted in a deviation of allocation of more than 1 pct. for at least 1 asset.')
        print(f'All optimal solution are as follows:')
        print(f'-'*22)
        for i,w in enumerate(wsols):
            print(f'Attempt {attempt[i]} had weights: {w*100} with {optimizing} = {fs[i]:.2f}\n')
        print(f'-'*22)
    else:
        print(f'The improvements of the optimal solution resulted in no notable deviations of allocation.')
    
    
    # vii. storing optimal portfolio weights and other stats
    if pftype.lower() == 'minvar':
        vopt = fopt
        sropt = ropt/np.sqrt(vopt)
        mdd, cropt = calmar(wopt,df,mu)
    elif pftype.lower() == 'tangent':
        vopt = wopt@sigma@wopt
        sropt = -fopt
        mdd, cropt = calmar(wopt,df,mu)
    else:
        vopt = wopt@sigma@wopt
        sropt = ropt/np.sqrt(vopt)
        mdd, cropt = calmar(wopt,df,mu)
    
    nwopt = np.around(wopt*100,2)
    wpfdf = pd.DataFrame({'Ticker':names, 'weight':nwopt})
    wpfdf = wpfdf.set_index('Ticker')
    statspfdf = pd.DataFrame.from_dict({'Annualized return':round(ropt,2),'Volatility':round(np.sqrt(vopt),2),
                                         'Sharpe ratio':round(sropt,2),'1m max drawdown':round(mdd,2),
                                        'Calmar ratio':round(cropt,2)},
                                       orient = 'index', columns = ['stats'])
    
    # viii. summarize the best optimal solution
    print(f'\nThe optimal portfolio allocation ended up being:')
    display(wpfdf)
    print(f'With portfolio characteristics:')
    display(statspfdf)
    
    return wpfdf, statspfdf



def num_optimal_portfolios(df,mu,N=50,shorting = False):
    '''
    Will numerically, using multistart optimization, solve the minimum variance portfolio, efficient tangent portfolio and the maximum calmar ratio portfolio.
    
    Args:
    df (dataframe): dataframe of adjusted close of the variance/covariance matrix for tickers.
    mu (array): array consisting of annual return for each ticker.
    N (integer): number of times the minimization should be executed (by default = 50)
    shorting (boolean): denotes whether shorting is allowed, handles boundaries of optimization (by default = False)
    
    Returns:
    (dataframe,dataframe,dataframe): 3 dataframes containing weights for the minimum variance portfolio, efficient tangent portfolio and the maximum Calmar ration portfolio respectively.
    '''
    # i. preparing data
    rdf = (df/df.shift(1)-1)*100
    sigma = rdf.cov()*252
    
    # ii. Stating whether shorting is allowed or not
    if shorting:
        print(f'SHORTING IS ALLOWED')
    else:
        print(f'SHORTING IS NOT ALLOWED')
    
    
    # ii. storing results for each of the portfolios
    mvwdf, mwstatsdf = optimize_pf('minvar',sigma,mu,df,N,shorting)
    twdf, tstatsdf = optimize_pf('tangent',sigma,mu,df,N,shorting)
    cwdf, cstatsdf = optimize_pf('calmar',sigma,mu,df,N,shorting)
    
    return mvwdf, twdf, cwdf



##################################
## simple portfolio simulations ##
##################################
def sim_pf(rdf, weights, N = 1000, t = 63):
    '''
    Will simulate N return paths over t days (or t/21 months). The simulated returns paths are drawn such that t days of return are selected randomly from the return dataframe. There is no rebalancing during the t days, meaning only initially each ticker is weighted properly and ultimately rise of falls according to said tickers return path. One could say that the return draws are IID over time (no autocorrelation) but tickers will still exhibit correlation with one another.
    
    Args:
    rdf (dataframe): dataframe of returns.
    weights (array): 1d array consisting of portfolio weights/allocation.
    N (integer): number of returns paths to draw (by default = 1000)
    t (integer): number of days for each return path, note there are roughly 21 trading days each month (by default = 63)
    
    Returns:
    plots the N simulated paths along with the 2.5th percentile, the median and the 97.5th percentil cumulated returns.
    (dataframe): dataframe containing the N simulated paths with weights (weights).
    '''
    # i. preparing storage of simulations and setting rng
    simulations = np.zeros((N,t+1))
    rng = np.random.default_rng(1995)
    
    # ii. drawing t random days N times, cumulating return and summing portfolio return
    matrixrdf = np.array(rdf)
    I = random_choice_noreplace(range(matrixrdf.shape[0]),N,t,rng)
    matrdf = matrixrdf[I]
    matrdf = matrdf/100+1
    matcdf = np.cumprod(matrdf, axis = 1)
    simulations = np.inner(weights,matcdf)-100
    dfsimulations = pd.DataFrame(simulations).T
    
    # iii. plotting simulated return paths
    fig = plt.figure(figsize = (12,8))
    plt.plot(dfsimulations, linewidth = 1, alpha = .1, color = 'blue')
    plt.plot(dfsimulations.quantile([.025,.5,.975], axis = 1).T, label = ['2.5th percentile', '50th percentile (median)','97.5th percentile'])
    plt.legend(frameon = True)
    plt.xlabel('Days', fontsize = 12)
    plt.ylabel('pct. return',fontsize = 12)
    plt.show();
    
    return dfsimulations



def simulated_portfolios(mvw, tw, cw, rdf, N = 1000, t = 63, shorting = False):
    '''
    Will simulate N return paths over t days (or t/21 months) for 3 types of portfolios (minimum variance, efficient tangent and maximum Calmar ratio portfolios). The simulated returns paths are drawn such that t days of return are selected randomly from the return dataframe. There is no rebalancing during the t days, meaning only initially each ticker is weighted properly and ultimately rise of falls according to said tickers return path. One could say that the return draws are IID over time (no autocorrelation) but tickers will still exhibit correlation with one another.
    
    Args:
    mvw (dataframe): dataframe of weights for the minimum variance portfolio
    tw (dataframe): dataframe of weights for the efficient tangent portfolio
    cw (dataframe): dataframe of weights for the maximum Calmar ratio portfolio
    rdf (dataframe): dataframe of returns.
    N (integer): number of returns paths to draw (by default = 1000)
    t (integer): number of days for each return path, note there are roughly 21 trading days each month (by default = 63)
    shorting (boolean): boolean indication whether shorting is allowed (by default = False)
    
    Returns:
    plots the N simulated paths along with the 2.5th percentile, the median and the 97.5th percentil cumulated returns for each of the portfolios.
    '''
    # i. preparing data and weights
    rdfclean = rdf.dropna()
    mvw = np.array(mvw).reshape(-1)
    tw = np.array(tw).reshape(-1)
    cw = np.array(cw).reshape(-1)
    
    if shorting:
        print(f'Simulating {N} timeseries for each type of portfolio where shorting is allowed')
    else:
        print(f'Simulating {N} timeseries for each type of portfolio where shorting is NOT allowed')
    
    print(f'-'*80)
    
    # ii. simulate and plot each type of portfolio
    print(f'\nSimulates the minimum variance portfolio {N} times over {t} days')
    mvsim = sim_pf(rdfclean,mvw,N,t)
    print(f'-'*100)
    print(f'\nSimulates the efficient tangent portfolio {N} times over {t} days')
    tsim = sim_pf(rdfclean,tw,N,t)
    print(f'-'*100)
    print(f'Simulates the optimal calmar ratio portfolio (shorting is not allowed) {N} times over {t} days')
    csim = sim_pf(rdfclean,cw,N,t)



def random_choice_noreplace(rdf, N, t,rng):
    '''
    Theres a possibility of np.random.choice having replicates for each N draw. This means a 1-time event could occur multiple times in the same Nth draw. This function will not have that problem. Here all indexes for each N are unique, meaning 1-time events doesn't have a chance of occuring in the same Nth draw. This function spits out relevant indexes.
    
    Args:
    rdf (dataframe): dataframe of returns.
    N (integer): number of returns paths to draw (by default = 1000)
    t (integer): number of days for each return path, note there are roughly 21 trading days each month (by default = 63)
    rng (generator): which default rng to have, makes sure results are replicable.
    
    Returns:
    (array): array containing indexes where all indexes are unique within each N draw (no duplicate index is picked for any Nth simulation).
    '''
    # t, N are the number of rows, cols of output, i.e. number of observations in each sample and number of samples
    return rng.random((N,len(rdf))).argsort(axis=-1)[:,:t]




###########
## to-do ##
###########
# pararellization possible?
# to histogram: add 5 percentile line