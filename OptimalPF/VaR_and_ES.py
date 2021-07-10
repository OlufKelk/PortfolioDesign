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
def data_prep(df,col):
    # i. from df column to array
    x = np.array(df[col])
    # ii. remove na's as they compromise the log-likelihood
    x = x[~np.isnan(x)]
    # iii. calculating log returns
    logR = np.diff(np.log(x))*100
    return logR




#################################
## estimate ARCH(1) parameters ##
#################################
def ARCH1_ll(theta,logR):
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
    # i. retrieving data
    logR = data_prep(df,ticker)
    
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
        print(f'Omega^hat                       --> {omega_hat:.4f} with std. errors ({se_hessian[0]:.4f}) and t-val {omega_hat/se_hessian[0]:.4f}')
        print(f'alpha^hat                       --> {alpha_hat:.4f} with std. errors ({se_hessian[1]:.4f}) and t-val {alpha_hat/se_hessian[1]:.4f}')
        print(f'Maximized log-likelihood        --> {logl:.3f}')
        print(f'--------------------------------------------------------------------------------------')
    
    
    return omega_hat, alpha_hat, logl



################
## VaR and ES ##
################
def VaRES(omega,a,df,ticker,alpha,h,M,printres = True):
    # i. initializing
    logR = data_prep(df,ticker)
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
    ES_gauss.fill(np.sqrt(np.var(logR))*np.sqrt(2)*norm.pdf(norm.ppf(1-alpha))/alpha)
    

    # iv. initializing VaR and ES array for ARCH(1) simulation
    VaR_ARCH1 = np.zeros(T-h+1)
    ES_ARCH1 = np.zeros(T-h+1)
    temp = np.zeros(M)
    
    
    # v. calculating VaR for ARCH(1) process
    # VaR has no closed form solution -> VaR is estimated using simulations
    for i in range(T-h+1):
        # v.a. simulate (negative) returns for each time period 
        for j in range(M):
            # v.b. M simulations of (negative) return for each time period 
            z = np.random.normal(loc = 0, scale = 1, size = h) # drawing innovations
            r1=np.sqrt(omega+a*logR[i]**2)*z[0] # compute return at time i+1
            r2=np.sqrt(omega+a*r1**2)*z[1] # compute return at time i+2
            temp[j]=-(r1+r2)# compute two-period loss
        
        VaR_ARCH1[i] = np.quantile(temp,1-alpha)
        ES_ARCH1[i] =  np.sqrt(np.var(temp))*norm.pdf(norm.ppf(1-alpha))/alpha
    # parallelize the above and adapt to varying h
    
    
    # vi. create dataframe with VaR
    colnames = str(h)+'_period_'
    df=pd.DataFrame(loss, columns=[colnames+'loss'])
    df[colnames+'VaR_gauss']=VaR_gauss
    df[colnames+'ES_gauss']=ES_gauss
    df[colnames+'VaR_ARCH']=VaR_ARCH1
    df[colnames+'ES_ARCH']=ES_ARCH1
    
    # vii. print result
    if printres == True:
        print(f'Risk measures for {ticker}')
        print(f'------------------------')
        print(f'Gauss')
        print(f'-----')
        print(f'VaR                             --> {VaR_gauss[0]:.2f}')
        print(f'ES                              --> {ES_gauss[0]:.2f}')
        print(f'-----------------------------------------')
        print(f'ARCH(1)')
        print(f'------')
        print(f'VaR                             --> {np.mean(VaR_ARCH1):.2f}')
        print(f'ES                              --> {np.mean(ES_ARCH1):.2f}')
        print(f'-----------------------------------------')

    return df


def VaRESz(omega,a,df,ticker,alpha,h,M,printres = True):
    # i. initializing
    logR = data_prep(df,ticker)
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
    ES_gauss.fill(np.sqrt(np.var(logR))*np.sqrt(2)*norm.pdf(norm.ppf(1-alpha))/alpha)
    

    # iv. initializing VaR and ES array for ARCH(1) simulation
    VaR_ARCH1 = np.zeros(T-h+1)
    ES_ARCH1 = np.zeros(T-h+1)
    temp = np.zeros(M)
    
    # v. drawing innovations
    z = np.random.normal(loc = 0, scale = 1, size = (T-h+1,M,h)) # drawing innovations
    
    # v. calculating VaR for ARCH(1) process
    # VaR has no closed form solution -> VaR is estimated using simulations
    for i in range(T-h+1):
        # v.a. simulate (negative) returns for each time period 
        for j in range(M):
            # v.b. M simulations of (negative) return for each time period 
            r1=np.sqrt(omega+a*logR[i]**2)*z[i,j,0] # compute return at time i+1
            r2=np.sqrt(omega+a*r1**2)*z[i,j,1] # compute return at time i+2
            temp[j]=-(r1+r2)# compute two-period loss
        
        VaR_ARCH1[i] = np.quantile(temp,1-alpha)
        ES_ARCH1[i] =  np.sqrt(np.var(temp))*norm.pdf(norm.ppf(1-alpha))/alpha
    # parallelize the above and adapt to varying h
    
    
    # vi. create dataframe with VaR
    colnames = str(h)+'_period_'
    df=pd.DataFrame(loss, columns=[colnames+'loss'])
    df[colnames+'VaR_gauss']=VaR_gauss
    df[colnames+'ES_gauss']=ES_gauss
    df[colnames+'VaR_ARCH']=VaR_ARCH1
    df[colnames+'ES_ARCH']=ES_ARCH1
    
    # vii. print result
    if printres == True:
        print(f'Risk measures for {ticker}')
        print(f'------------------------')
        print(f'Gauss')
        print(f'-----')
        print(f'VaR                             --> {VaR_gauss[0]:.2f}')
        print(f'ES                              --> {ES_gauss[0]:.2f}')
        print(f'-----------------------------------------')
        print(f'ARCH(1)')
        print(f'------')
        print(f'VaR                             --> {np.mean(VaR_ARCH1):.2f}')
        print(f'ES                              --> {np.mean(ES_ARCH1):.2f}')
        print(f'-----------------------------------------')

    return df


def VaRESzmj(omega,a,df,ticker,alpha,h,M,printres = True):
    # i. initializing
    logR = data_prep(df,ticker)
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
    ES_gauss.fill(np.sqrt(np.var(logR))*np.sqrt(2)*norm.pdf(norm.ppf(1-alpha))/alpha)
    
    # iv. initializing relevant arrays for ARCH(1) simulation
    VaR_ARCH1 = np.zeros(T-h+1)
    ES_ARCH1 = np.zeros(T-h+1)
    r1 = np.zeros(M)
    r2 = np.zeros(M)
    r1r2 = np.zeros((T-h+1,M))
    
    # v. drawing innovations
    z = np.random.normal(loc = 0, scale = 1, size = (T-h+1,M,h))
    
    # v. calculating VaR for ARCH(1) process
    # VaR has no closed form solution -> VaR is calculated using simulations
    for i in range(T-h+1):
        # v.a. simulate (negative) returns for each time period 
        r1 = np.sqrt(omega+a*logR[i]**2)*z[i,:,0]
        r2 = np.sqrt(omega+a*r1**2)*z[i,:,1]
        r1r2[i,:] = -(r1+r2)
        VaR_ARCH1[i] = np.quantile(r1r2[i],1-alpha)
        ES_ARCH1[i] =  np.sqrt(np.var(r1r2[i]))*norm.pdf(norm.ppf(1-alpha))/alpha

    # vi. create dataframe with VaR
    colnames = str(h)+'_period_'
    df=pd.DataFrame(loss, columns=[colnames+'loss'])
    df[colnames+'VaR_gauss']=VaR_gauss
    df[colnames+'ES_gauss']=ES_gauss
    df[colnames+'VaR_ARCH']=VaR_ARCH1
    df[colnames+'ES_ARCH']=ES_ARCH1
    
    # vii. print result
    if printres == True:
        print(f'Risk measures for {ticker}')
        print(f'------------------------')
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


def VaRESzmji(omega,a,df,ticker,alpha,h,M,printres = True):
    # i. initializing
    logR = data_prep(df,ticker)
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
    ES_gauss.fill(np.sqrt(np.var(logR))*np.sqrt(2)*norm.pdf(norm.ppf(1-alpha))/alpha)
    
    # iv. initializing relevant arrays for ARCH(1) simulation
    VaR_ARCH1 = np.zeros(T-h+1)
    ES_ARCH1 = np.zeros(T-h+1)
    r1 = np.zeros(M)
    r2 = np.zeros(M)
    r1r2 = np.zeros((T-h+1,M))
    
    # v. drawing innovations
    z = np.random.normal(loc = 0, scale = 1, size = (T-h+1,M,h))
    
    # v. calculating VaR for ARCH(1) process
    # VaR has no closed form solution -> VaR is calculated using simulations
    for i in range(T-h+1):
        # v.a. simulate (negative) returns for each time period 
        r1 = np.sqrt(omega+a*logR[i]**2)*z[i,:,0]
        r2 = np.sqrt(omega+a*r1**2)*z[i,:,1]
        r1r2[i,:] = -(r1+r2)
        VaR_ARCH1[i] = np.quantile(r1r2[i],1-alpha)
        ES_ARCH1[i] =  np.sqrt(np.var(r1r2[i]))*norm.pdf(norm.ppf(1-alpha))/alpha

    # vi. create dataframe with VaR
    colnames = str(h)+'_period_'
    df=pd.DataFrame(loss, columns=[colnames+'loss'])
    df[colnames+'VaR_gauss']=VaR_gauss
    df[colnames+'ES_gauss']=ES_gauss
    df[colnames+'VaR_ARCH']=VaR_ARCH1
    df[colnames+'ES_ARCH']=ES_ARCH1
    
    # vii. print result
    if printres == True:
        print(f'Risk measures for {ticker}')
        print(f'------------------------')
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


#########
## ES ##
#########




###########
## plots ##
###########




####################
## call functions ##
####################




###########
## to-do ##
###########
# adapt VAR and ES to varying h
# find some way to include datetime for each ticker (in plot of VaR and ES)
# create GARCH(1,1) model
# Vary whether you want to estimate an ARCH(1) model or a GARCH(1,1) model.
# load data properly from github
# create figure for VaR and ES
# add docstrings
# calculate simple unconditional valuation of VaR E[loss>VaR] should be (close to) alpha

