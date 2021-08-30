import numpy as np
from scipy import optimize
import pandas as pd
from scipy.stats import norm
from numba import njit, prange
# import matplotlib.pyplot as plt
# import datetime as dt
# import seaborn as sns
# import pandas_datareader.data as web

####################
## data unpacking ##
####################
def data_prep(df,ticker):
    '''
    Function which takes a dataframe and spits out the relevant array
    
    Args:
    df (dataframe): dataframe containing the adjusted close price of ticker
    ticker (string): string being the relevant ticker code (name of column in dataframe)

    Returns:
    (tuple): firest element is an array of dates while second element is an array with its corresponding log-returns
    '''
    # i. remove na's as they compromise the log-likelihood
    tempdf = df[~np.isnan(df[ticker])]
    
    # ii. storing dates
    dates = tempdf.index.values
    
    # iii. from df column to array + calculating log returns
    x = np.array(tempdf[ticker])
    logR = np.diff(np.log(x))*100
    
    return dates, logR







#################################
## estimate ARCH(1) parameters ##
#################################
def ARCH1_ll(theta,logR):
    '''
    Estimates the loglikelihood of an ARCH(1)-model for some log-returns data and a theta (omega,alpha)

    Args:
    theta (tuple): tuple containing omega and alpha parameters
    logR (array): array containing the log returns for some ticker

    Returns:
    (float): returns the log-likelihood of the ARCH(1)-model for the data (logR) and the parameters (theta)
    '''
    
    # i. unpacking parameter values
    omega = theta[0]
    alpha = theta[1]
    
    # ii. retrieving max periods
    T = len(logR)
    
    # iii. initializing sigma2 series
    sigma2 = np.empty(T)
    sigma2[0] = np.var(logR) # starting at simple variance of returns
    sigma2[0] = 1 # starting at 1
    
    # iv. estimating sigma2 in ARCH(1,1) process
    for t in range(1,T):
        sigma2[t] = omega+alpha*logR[t-1]**2
    
    # v. calculating the log-likelihood to be optimized (negative)
    LogL = -np.sum(-np.log(sigma2)-logR**2/sigma2)
    
    return LogL


def ARCH1_est(ticker, df, printres = True):
    '''
    Estimates a given ticker as a ARCH(1)-model

    Args:
    ticker (string): string being the relevant ticker code (name of column in dataframe)
    df (dataframe): dataframe containing the adjusted close price of ticker
    printres (boolean): whether or not to print the results, by default True

    Returns:
    (tuple): tuple containing 3 elements. First the estimated omega, next the estimates alpha and lastly the maximized log-likelihood
    '''
    
    
    # i. retrieving data
    _, logR = data_prep(df,ticker)
    
    # ii. initial guess
    theta0 = ([0.25,2])
    
    # iii. optimizing
    sol = optimize.minimize(lambda x: ARCH1_ll(x,logR), theta0, bounds = ((1e-8,None),(0,None)))
    
    # iv. unpacking estimates
    omega_hat = sol.x[0]
    alpha_hat = sol.x[1]
    logl = -sol.fun
    
    # v. computing standard errors
    # negative inverse hessian matrix (we have minimized negative log-likelihood)
    v_hessian = sol.hess_inv.todense()
    se_hessian = np.sqrt(np.diagonal(v_hessian))
    
    
    # vi. printing result
    if printres == True:
        print(f'Estimating {ticker} as a ARCH(1)-model resulted in:')
        print(f'--------------------------------------------------------------------------------------')
        print(f'Omega^hat                       --> {omega_hat:.4f} with std. errors ({se_hessian[0]:.4f}) and t-val {omega_hat/se_hessian[0]:.4f}')
        print(f'alpha^hat                       --> {alpha_hat:.4f} with std. errors ({se_hessian[1]:.4f}) and t-val {alpha_hat/se_hessian[1]:.4f}')
        print(f'Maximized log-likelihood        --> {logl:.3f}')
        print(f'--------------------------------------------------------------------------------------')
    
    
    return omega_hat, alpha_hat, logl



################
## VaR and ES ##
################
def VaRES(omega,a,df,ticker,alpha,h,M,printres = True):
    '''
    Calculates gaussian and ARCH(1) Value at Risk and Expected shortfall for a given ticker at a given risk level (alpha).

    Args:
    omega (float): the estimated omega of the ticker (using an ARCH(1)-model) 
    a (float): the estimated alpha of the ticker (using an ARCH(1)-model)
    df (dataframe): dataframe containing the adjusted close price of ticker
    ticker (string): string being the relevant ticker code (name of column in dataframe)
    alpha (float): is the risk level being in (0,1)
    h (integer): is the h period loss
    M (integer): denotes the number of simulations for each time period
    printres (boolean): whether or not to print the results, by default True

    Returns:
    (dataframe): dataframe containing the 2 period loss, 2 period VaR and ES for both gaussian and ARCH(1) returns 
    '''
    
    # i. initializing
    dates, logR = data_prep(df,ticker)
    T = len(logR)
    
    # ii. h period loss
    loss = np.zeros(T-h+1)
    j = 0
    for i in reversed(range(h)):
        loss -= logR[i:T-j]
        j += 1
              
    # iii. calculating gaussian VaR and ES
    # Value at Risk
    VaR_gauss = np.zeros(T-h+1)
    VaR_gauss.fill(-np.sqrt(np.var(logR))*np.sqrt(h)*norm.ppf(alpha))
    # Expected Shortfall
    ES_gauss = np.empty(T-h+1)
    ES_gauss.fill(np.sqrt(np.var(logR))*np.sqrt(h)*norm.pdf(norm.ppf(1-alpha))/alpha)
    
    # iv. initializing relevant arrays for ARCH(1) simulation
    VaR_ARCH1 = np.zeros(T-h+1)
    ES_ARCH1 = np.zeros(T-h+1)
    simr = np.zeros((T-h+1,M))
    
    # v. drawing innovations
    z = np.random.normal(loc = 0, scale = 1, size = (T-h+1,M,h))
    
    # v. calculating VaR and ES for ARCH(1) process
    # in this case VaR and ES has no closed form solution -> therefore calculated using simulations
    assert h < 4, f'VaR and ES only supported for 1, 2 and 3 period losses, not h={h}.'
    if h == 1:
        simr = -np.sqrt(omega+a*logR**2)*np.transpose(z[:,:,0])
    elif h == 2:
        r1 = np.sqrt(omega+a*logR[:-1]**2)*np.transpose(z[:,:,0])
        r2 = np.sqrt(omega+a*r1**2)*np.transpose(z[:,:,1])
        simr = -(r1+r2)
    elif h == 3:
        r1 = np.sqrt(omega+a*logR[:-2]**2)*np.transpose(z[:,:,0])
        r2 = np.sqrt(omega+a*r1**2)*np.transpose(z[:,:,1])
        r3 = np.sqrt(omega+a*r2**2)*np.transpose(z[:,:,2])
        simr = -(r1+r2+r3)
    
    VaR_ARCH1 = np.quantile(simr,1-alpha, axis = 0) # approximate VaR based on simulated losses
    # the two lines below apply for a GARCH process where r_t+h isn't conditionally gaussian (conditional on I_t)
#     loss_exceeds_VaR = simr > VaR_ARCH1[None,]
#     ES_ARCH1 = np.mean(simr*loss_exceeds_VaR, axis = 0)/alpha # approximate ES based on simulated losses and VaR
    # However in an ARCH(1)-process, this isn't a problem
    ES_ARCH1 =  np.sqrt(np.var(simr, axis = 0))*norm.pdf(norm.ppf(1-alpha))/alpha
    
    # vi. calculating the coverage of gaussian and ARCH(1) VaR
    hit = loss > VaR_ARCH1
    coverage = np.mean(hit)    
    
    # vii. create dataframe with VaR
    colnames = str(h)+'_period_'
    df=pd.DataFrame(loss, columns=[colnames+'loss'], index = dates[h:])
    df[colnames+'VaR_gauss']=VaR_gauss
    df[colnames+'ES_gauss']=ES_gauss
    df[colnames+'VaR_ARCH']=VaR_ARCH1
    df[colnames+'ES_ARCH']=ES_ARCH1
    
    # viii. print result
    if printres == True:
        print(f'Risk measures for {ticker} at {h} period losses with a {alpha} risk level is')
        print(f'-----------------------------------------')
        print(f'Gauss')
        print(f'-----')
        print(f'VaR                             --> {VaR_gauss[0]:.2f}')
        print(f'ES                              --> {ES_gauss[0]:.2f}')
        print(f'-----------------------------------------')
        print(f'ARCH(1)')
        print(f'------')
        print(f'VaR (average)                   --> {np.mean(VaR_ARCH1):.2f}')
        print(f'ES  (average)                   --> {np.mean(ES_ARCH1):.2f}')
        print(f'-----------------------------------------')

    return df


###########
## plots ##
###########



####################
## call functions ##
####################




###########
## to-do ##
###########
# create GARCH(1,1) model
## Vary whether you want to estimate an ARCH(1) model or a GARCH(1,1) model.
# load data properly from github
# create figure for VaR and ES
# calculate simple unconditional valuation of VaR E[loss>VaR] should be (close to) alpha
## this one fits poorly for some reason. Should be investigated at some point

