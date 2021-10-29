# Creating the Optimal Portfolio(s)

Stock picking and portfolio composition can be chaotic. This simple script aim at using historic return data for some selected list of stocks (or other assets) in order to create the 3 optimal portfolios. This way one can quickly create a portfolio based on the (historic) data.<br/>
The 3 optimal portfolios are the *i)* minimum variance portfolio, *ii)* the efficient tangent portfolio as well as the *iii)* maximum calmar ratio portfolio. The first 2 portfolios have an optimal allocation with and without short-selling and are solved both analytically and numerically. <br/>
The [notebook](Porfolio_notebook.ipynb), being an example of the available function's capabilities, is constructed such that:
1. Data for a given list of assets are imported and manipulated properly, either using local csv-files or the Yahoo-database.
2. The list of assets are then described both visually (correlation heatmap, indexed returns and return distribution) and numeri (annualized return, volatility, 1 month max drawdown, Sharpe ratio, Calmar ratio)
3. Assets are then analyzed by 
    1. Quantifying the Value-at-Risk and Expected Shortfall (both gaussian and as an ARCH(1)-process) for a given stock
    2. Solving the minimum variance portfolio and efficient tangent portfolio **analytically** (*allowing for shorting*)
    3. Solving the minimum variance portfolio and efficient tangent portfolio **numerically** (*allowing for shorting*)
    4. Solving the minimum variance portfolio, efficient tangent portfolio and maximum calmar-ratio portfolio **numerically** (*no shorting*)
    5. Simulating 3 months of return for a given set of assets and corresponding weights, that is e.g. for the efficient tangent portfolio. This simulation picks random historic daily returns, meaning that the simulation assumes that there's no autocorrelation in stock returns. This is a very harsh (and faulty) assumption, meaning this simulation should be regarded with a certain degree of skepticism.


**Dependencies:** Apart from a standard Anaconda Python 3 installation, the project requires the following installations:
``pip install numpy``
``pip install DateTime``
``pip install scipy``
``pip install pandas``
``pip install numpy``
``pip install matplotlib``
``pip install python-dateutil``
``pip install seaborn``
``pip install pandas-datareader``
