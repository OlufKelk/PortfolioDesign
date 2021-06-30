import numpy as np
from scipy import optimize
import pandas as pd
# import matplotlib.pyplot as plt
# import datetime as dt
# import seaborn as sns
# import pandas_datareader.data as web

####################
## data unpacking ##
####################

def data_prep(df,col):
    x = np.array(df[col])
    # log returns times 100!!!
    
    
    return x




#################################
## estimate ARCH(1) parameters ##
#################################
def ARCH11_ll(df,ticker,theta):
    # i. preparing relevant data
    x_vals = data_prep(df,ticker)
    
    # ii. unpacking parameter values
    omega = theta[0]
    alpha = theta[1]
    
    # iii. retrieving max periods
    T = len(x_vals)
    
    # iv. initializing sigma2 series
    sigma2 = np.empty(T)
    sigma2[0] = np.var(x_vals) # starting at simple variance of returns
    sigma2[0] = 1 # starting at 1
    
    # v. estimating sigma2 in ARCH(1,1) process
    for t in range(1,T):
        sigma2[t] = omega+alpha*x_vals[t-1]**2
    
    # vi. calculating the log-likelihood to be optimized (negative)
    LogL = -np.sum(-np.log(sigma2)-x_vals**2/sigma2)
    
    return LogL 


def ARCH11_est(ticker, printres = True):
    # i. initial guess
    theta0 = ([0.25,2])
    
    # ii. optimizing
    sol = optimize.minimize(lambda x: ARCH11_ll(x,ticker), theta0, bounds = ((1e-8,None),(0,None)))
    
    # iii. unpacking estimates
    omega_hat = sol.x[0]
    alpha_hat = sol.x[1]
    logl = -sol.fun
    
    # iv. computing standard errors
    # negative inverse hessian matrix (we have minimized negativ log-likelihood)
    v_hessian = sol.hess_inv.todense()
    se_hessian = np.sqrt(np.diagonal(v_hessian))
    
    
    # iv. printing result
    if printres == True:
        print(f'Estimating {ticker} as a ARCH(1,1) model resulted in:')
        print(f'Omega^hat                       --> {omega_hat:.4f} with std. errors ({se_hessian[0]:.4f}) and t-val {omega_hat/se_hessian[0]:.4f}')
        print(f'alpha^hat                       --> {alpha_hat:.4f} with std. errors ({se_hessian[1]:.4f}) and t-val {alpha_hat/se_hessian[1]:.4f}'')
        print(f'Resultes in a log-likelihood of --> {logl:.3f}')
    
    
    return omega_hat, alpha_hat, logl



#########
## VaR ##
#########


def VaR(omega,a,xvals,alpha,h,M):
    # i. initializing
    x = xvals.T[0,:]
    T = len(x)
    
    # a. h period loss
    loss = np.zeros(T-h)
    j = 0
    for i in reversed(range(h)):
        loss -= x[i,T-j]
        j += 1
              
    
    # b. initializing h period VaR for ARCH(1,1) and gaussian returns
    temp1 = np.zeros(M)
    temp2 = -np.sqrt(np.var(x))*np.sqrt(h)*norm.ppf(alpha) # check np.sqrt(h) is correct here!!!

    # c. initializing VaR variable
    VaR_ARCH11 = np.zeros(T-2)
    VaR_gauss = np.zeros(T-2)
              
    # ii. calculating VaR for ARCH(1,1) process
    for ii in range(T-h): # check h is correct here as well!!!
        for jj in range(M):
            z = np.random.normal(loc = 0, scale = 1, size = 2) # drawing innovations (check if size should be h)
            r1 = np.sqrt(omega+a*x[i]**2)*z[0] # find way to adopt this properly to h time periods
            r2 = np.sqrt(omega+a*r1**2)*z[1]
            temp1[j] = -(r1+r2)
        
        VaR_ARCH11[i] = np.quantile(temp1,1-alpha)
        VaR_gauss[i] = temp2
    # can any of the above be done using matrix-notation or in parallel??
    
    
    # iii. create dataframe with VaR
    colnames = str(h)+'_period_'
    df=pd.DataFrame(loss, columns=[colnames+'loss'])
    df[colnames+'VaR_gauss']=VaR_gauss
    df[colnames+'VaR_ARCH']=VaR






start_time = time.time()
folder='Insert your path here! xD)'
data=np.genfromtxt(folder+'SP500_from2012.csv', delimiter=',') #loading in dataset
x = data[1:, 4:5] #100 times log-returns of the S&P 500 index from 2012
x=x.T[0,:] #unpacking numpy array
T=len(x)
omega = 0.497556 #MLE of omega 
a = 0.208533 #//MLE of alpha
alpha = 0.05
loss=np.zeros(T-2)
loss=-(x[1:T-1] + x[2:T]) #//future two-period loss
M=10000 #number of monte carlo simulations
temp1=np.zeros(M)
temp2=-np.sqrt(np.var(x))*np.sqrt(2)*norm.ppf(alpha) #two period VaR for iid Gaussian returns

VaR=np.zeros(T-2) #VaR variable
VaR_Gauss=np.zeros(T-2) #VaR gaussian variable

for i in range(T-2):
    for j in range(M):
        z=np.random.normal(loc=0.0, scale=1.0, size=2) #draw innovations from N(0,1)
        r1=np.sqrt(omega+a*x[i]**2)*z[0] #//compute return at time i+1
        r2=np.sqrt(omega+a*r1**2)*z[1] #//compute return at time i+2
        temp1[j]=-(r1+r2)#//compute two-period loss
    VaR[i]=np.quantile(temp1,1-alpha) #// compute (1-alpha) percentile of losses
    VaR_Gauss[i]=temp2 #VaR under Gaussianity
print('Elapsed time (all computations)+'"--- %s seconds ---" % (time.time() - start_time))
              
              
              
#exporting to CSV file
df=pd.DataFrame(loss, columns=["2_period_loss"])
df["2_period_VaR_Gauss"]=VaR_Gauss
df["2_period_VaR_ARCH"]=VaR


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





