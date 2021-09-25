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
    # a. loading dataset from yahoo and only keeping adjusted close column
    temp_df = web.DataReader(ticker, 'yahoo', start, end)['Adj Close']
    
    # b. renaming column
    temp_df = pd.DataFrame(temp_df).rename(columns = {'Adj Close': ticker})
    return temp_df


def df_generator(tickers, method, apath = None, start = None, end = None):
    
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
    desc_df(df)

    return df, rdf, cdf


def desc_df(df):
    first_obs = df.first_valid_index()
    first_obs_date = df.first_valid_index().strftime('%d-%m-%Y')
    print(f'First observation in dataframe is         {first_obs_date}')
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
    m_mdd = np.nan*np.zeros(sol_shape)
    a_cr = np.nan*np.zeros(sol_shape)
    
    # ii. calculating stats for each ticker
    for i, col in enumerate(rdata.columns):
        # a. storing number of days the ticker has been around
        start = data[col].first_valid_index()
        end = data[col].last_valid_index()
        no_days = (end - start).days

        # b. calculating stats
        absolute_return = data[col][data[col].last_valid_index()]/data[col][data[col].first_valid_index()]-1
        ar = round(((1+absolute_return)**(1/(no_days/365))-1)*100,2)
        vol = round(rdata[col].std()*np.sqrt(252),2)
        sr = round((ar - rfree) / vol,2)
        monthlyreturn = (data[col]/data[col].shift(21)-1)*100
        drawdown = np.abs(monthlyreturn.min())
        cr = round(ar/drawdown,2)

        # c. storing stats
        a_returns[i] = ar
        a_volatilities[i] = vol
        a_sr[i] = sr
        m_mdd[i] = round(drawdown,2)
        a_cr[i] = cr

    # iii. creating dataframe with stats
    stats = pd.DataFrame({'Ticker': rdata.columns,
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
# Analytical solution for minimum variance portfolio
def mvar_analytical(df,sigma,mu,printres = True):
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
    # i. calculating relevant data for optimal portfolios
    sigma = rdf.cov()*252
    
    # ii. calling analytical optimizers
    wmvdf,statmvdf = mvar_analytical(df,sigma,mu)
    wtandf,stattandf = tan_analytical(df,sigma,mu)
    print(f'\nCannot analytically solve the maximum calmar ratio portfolio')
    


###########################
## 2. numerical solution ##
###########################
# helping function for minimum variance portfolio
def mvar(w,sigma):    
    # i. normalizing weights
    nw = w/sum(w)
    
    # ii. calculating variance based on given weights
    variance = nw @ sigma @ nw
    
    return variance


# helping function for efficient tangent portfolio
def tangent(w,sigma,mu):
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
    # i. normalizing weights
    nw = w/sum(w)

    # ii. calulating the max monthly drawdown of portfolio based on weights
    pfprice = nw@df.T # this will only be calculated when there is no nans
    monthlyreturn = (pfprice/pfprice.shift(21)-1)*100
    drawdown = np.abs(monthlyreturn.min())
    
    # iii. calculating yearly return
    r = nw @ mu
        
    # iv. calculating the portfolios respective calmar ratio ratio (with risk-free return = 0)
    calmarratio = r/drawdown
    
    return drawdown, calmarratio


def num_optimal_portfolios(df,mu,N=50,shorting = False):
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


def optimize_pf(pftype,sigma,mu,df,N,shorting = False):
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



##################################
## simple portfolio simulations ##
##################################
def sim_pf(rdf, weights, N = 1000, t = 60):
    # i. preparing storage of simulations
    simulations = np.zeros((N,t+1))
    
    # ii. drawing t random days N times 
    for i in range(N):
        # a. sampling data and calculating the cumulative return
        rdfcleani = rdf.sample(t)
        rdfcleani = rdfcleani/100+1
        rdfcleani = rdfcleani.cumprod()
        
        # b. calculating portfolio cumulated return for the 60 days as an array
        simulations[i,1:] = np.array(weights@rdfcleani.T)-100
    
    # iii. converging simulations to dataframe
    dfsimulations = pd.DataFrame(simulations).T
    
    # iv. plotting simulated return paths
    fig = plt.figure(figsize = (12,8))
    plt.plot(dfsimulations, linewidth = 1, alpha = .1, color = 'blue')
    plt.plot(dfsimulations.quantile([.025,.5,.975], axis = 1).T, label = ['2.5th percentile', '50th percentile (median)','97.5th percentile'])
    plt.legend(frameon = True)
    plt.xlabel('Days', fontsize = 12)
    plt.ylabel('pct. return',fontsize = 12)
    plt.show();
    
    return dfsimulations



def simulated_portfolios(mvw, tw, cw, rdf, N = 1000, t = 60, shorting = False):
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








###########
## to-do ##
###########
# histplot vs. distplot
# one is deprecated, make sure you have relevant version of seaborn


# optimal portfolio
## simulated results (in parallel)
## Get stats for the portfolio


# include in stats: 
## Stats for how much of a normal distribution it is 

# inlcude in plots:
## Simulated path



# important
# should not use fillnas when generating dataframe.
## data will look stable when estimating as ARCH(1)
## however having na's will probably fuck up the percentage returns?
## perhaps one should use log returns all the way through.



###############
## graveyard ##
###############